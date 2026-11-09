"""Offline Git/evidence verification of the frozen pre-M4 delivery snapshot."""
import argparse
import json
from pathlib import Path
import re
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def require(condition, message):
    if not condition:
        raise ValueError('Delivery status mismatch: ' + message)


def git(root, *args):
    return subprocess.check_output(['git', '-C', str(root), *args], text=True).strip()


def validate_status(data, root):
    """Validate a dated snapshot, not current live GitHub approval/merge state."""
    root = Path(root).resolve()
    require(data['schema_version'] == 1, 'schema version')
    require(git(root, 'rev-parse', '--is-shallow-repository') == 'false', 'full Git history required')
    cache = {}

    def ancestors(sha):
        require(isinstance(sha, str) and re.fullmatch('[0-9a-f]{40}', sha) is not None, 'full commit SHA')
        if sha not in cache:
            cache[sha] = set(git(root, 'rev-list', sha).splitlines())
        return cache[sha]

    def evidence(paths):
        for path in paths:
            target = (root / path).resolve()
            require(target.is_relative_to(root) and target.is_file(), 'missing/escaping evidence: ' + path)

    baseline = data['baseline']['commit']; main = data['main']['commit']
    baseline_ancestors = ancestors(baseline); main_ancestors = ancestors(main)
    require(main in baseline_ancestors, 'main is not an ancestor of baseline')
    delta = baseline_ancestors - main_ancestors
    require(len(delta) == data['snapshot']['commits_ahead_of_main'], 'ahead count')
    require(data['baseline']['tested_code_commit'] in baseline_ancestors, 'tested SHA outside baseline')
    require(not data['baseline']['is_default_branch'] or baseline == main, 'false default-branch claim')
    evidence(data['baseline']['evidence'])
    pulls = data['pull_requests']; by_number = {p['number']: p for p in pulls}
    require(len(by_number) == len(pulls), 'duplicate PR number')
    require(len(pulls) == data['snapshot']['pull_request_count'], 'PR snapshot count')
    require(sum(p['state'] == 'open' for p in pulls) == data['snapshot']['open_pull_request_count'], 'open PR count')
    issue_rows = data['issues']; issues = {i['number']: i for i in issue_rows}
    require(len(issues) == len(issue_rows) == data['snapshot']['issue_count'], 'Issue snapshot count')
    require(not set(issues) & set(by_number), 'PR incorrectly registered as Issue')
    visiting, visited = set(), set()

    def visit(number):
        require(number in by_number, 'unknown PR dependency')
        require(number not in visiting, 'PR dependency cycle')
        if number in visited:
            return
        visiting.add(number)
        parent = by_number[number]['base_pr']
        if parent is not None:
            visit(parent)
        visiting.remove(number); visited.add(number)

    for number in by_number:
        visit(number)
    covered = set()
    for p in pulls:
        head_ancestors = ancestors(p['head_sha']); base_ancestors = ancestors(p['base_sha'])
        require(p['included_in_baseline'] == (p['head_sha'] in baseline_ancestors), 'PR inclusion claim')
        require(p['head_sha'] in baseline_ancestors, 'PR head outside baseline')
        require(p['base_sha'] in head_ancestors, 'PR base is not in its head history')
        require(p['state'] in ('open', 'closed'), 'PR state')
        require(not p['merged'] or p['state'] == 'closed', 'open PR claimed merged')
        if p['merged']:
            require(p['head_sha'] in main_ancestors, 'merged claim absent from main snapshot')
        if p['base_pr'] is not None:
            parent = by_number[p['base_pr']]
            require(p['base_ref'] == parent['head_ref'] and p['base_sha'] == parent['head_sha'], 'PR dependency ref/SHA')
        require(set(p['issues']) <= issues.keys(), 'PR references unknown Issue')
        covered |= head_ancestors - base_ancestors
    bridges = data['bridge_commits']
    require(len({b['sha'] for b in bridges}) == len(bridges), 'duplicate bridge commit')
    require({b['sha'] for b in bridges} == delta - covered, 'bridge coverage')
    for bridge in bridges:
        require(bridge['sha'] in baseline_ancestors, 'bridge outside baseline')
        require(set(bridge['issues']) <= issues.keys(), 'bridge references unknown Issue')
        evidence(bridge['evidence'])
    for issue in issue_rows:
        require(issue['state'] in ('open', 'closed'), 'Issue state')
        expected = sorted(p['number'] for p in pulls if issue['number'] in p['issues'])
        require(issue['linked_prs'] == expected, 'Issue/PR cross-reference')
        expected = [b['sha'] for b in bridges if issue['number'] in b['issues']]
        require(issue['bridge_commits'] == expected, 'Issue/bridge cross-reference')
    modules = data['modules']; ids = [m['id'] for m in modules]
    require(len(ids) == len(set(ids)), 'duplicate module ID')
    module_map = root / 'docs/architecture/module-map.json'
    if module_map.exists():
        expected_ids = {m['id'] for m in json.loads(module_map.read_text())['modules']}
        require(expected_ids <= set(ids), 'module index entry missing from ledger')
    for module in modules:
        require(module['implementation'] in ('planned', 'partial', 'delivered'), 'implementation status')
        require(module['engineering'] in ('pending', 'accepted', 'partial', 'not_applicable'), 'engineering status')
        require(module['empirical'] in ('pending', 'not_applicable', 'pilot_no_stable_gain', 'development_candidate'), 'empirical status')
        require(set(module['prs']) <= by_number.keys() and set(module['issues']) <= issues.keys(), 'module tracking reference')
        if module['engineering'] == 'accepted':
            require(module['implementation'] != 'planned' and module['evidence'], 'acceptance without implementation/evidence')
        require(module['integration'] == 'pending_main_review', 'unsupported module merge claim')
        evidence(module['evidence'])
    require(data['tracking']['main_integration_state'] == 'not_merged', 'unreviewed main integration claim')
    require(data['tracking']['next_module'] == 'M4' and not data['tracking']['main_merge_required_before_m4'], 'M4 queue contract')
    current = git(root, 'rev-parse', 'HEAD')
    require(baseline in ancestors(current), 'checkout does not contain baseline')
    changed = subprocess.check_output(['git', '-C', str(root), 'diff', '--name-only', '-z', baseline, current]).decode().rstrip('\0')
    paths = changed.split('\0') if changed else []
    allowed = ('scripts/verify_delivery_status.py', 'tests/test_delivery_status.py',
               'tests/test_run_provenance.py', 'README.md')
    require(all(p in allowed or p.startswith(('docs/', 'research/')) for p in paths), 'algorithm/training change after functional baseline')
    return {'status': 'verified', 'snapshot_date': data['snapshot']['date'], 'baseline': baseline,
            'main_snapshot': main, 'pull_requests': len(pulls), 'bridge_commits': len(bridges),
            'issues': len(issues), 'modules': len(modules), 'ahead_of_main': len(delta),
            'checkout': current, 'next_module': data['tracking']['next_module']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--expected-commit', required=True)
    parser.add_argument('--out-dir', type=Path, required=True)
    args = parser.parse_args(); root = Path(__file__).resolve().parents[1]
    out = args.out_dir.resolve()
    require(not out.is_relative_to(root) and not root.is_relative_to(out), 'output overlaps checkout')
    out.mkdir(parents=True, exist_ok=False)
    receipt = {'status': 'running', 'command': [sys.executable, *sys.argv]}
    try:
        require(git(root, 'rev-parse', 'HEAD') == args.expected_commit, 'unexpected checkout SHA')
        require(not git(root, 'status', '--porcelain'), 'dirty checkout')
        data = json.loads((root / 'docs/architecture/delivery-status.json').read_text())
        receipt.update(validate_status(data, root))
        require(not git(root, 'status', '--porcelain'), 'checkout changed during verification')
    except BaseException as error:
        receipt.update(status='failed', error=repr(error)); raise
    finally:
        (out / 'manifest.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2))
    print(json.dumps(receipt, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
