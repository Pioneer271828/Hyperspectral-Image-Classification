"""One current status projection and read-only integration acceptance checks."""
import ast
import json
from pathlib import Path
import re
import subprocess

STATUS_PATH = 'docs/architecture/current-status.json'
SUMMARY_PATH = 'research/当前搭建状态与审查整改清单_20261102.md'
PLAN_PATHS = (
    'research/研究内容整合与详细搭建测试计划_2026-10-28.md',
    'research/模块化搭建与测试消融实施计划_2026-10-28.md',
    'research/全量研究模块必测清单_20261104.md',
)


def render_status(value):
    lines = [
        '# 当前搭建状态与审查整改清单', '',
        '> 自动投影自 [current-status.json](../docs/architecture/current-status.json)。',
        '> 请只修改该JSON，再在远程运行 `python scripts/check_repository.py --write-status`。',
        '> 历史过程见[归档快照](历史状态快照_20261105.md)；归档中的下一步不作为当前执行队列。', '',
        '更新日期：' + value['as_of'] + '。工程通过、真实效果与主分支交付分别判断。', '',
        '- 开发整合分支：`' + value['development_branch'] + '`。',
        '- main修复候选：`' + value['main_repair_branch'] + '`；合并状态：`' + value['main_merge'] + '`。',
        '- 自动验收运行器：`' + value['ci_runner'] + '`；未注册不等于CI已经运行。',
        '- 本次整改不启动真实训练或query计分；实验实时进度以各冻结批次回执为准。', '',
        '| 模块 | 当前状态 | 已有证据 | 剩余工作 |',
        '|---|---|---|---|',
    ]
    for row in value['modules']:
        lines.append('| [' + row['id'] + '](' + row['report'].removeprefix('research/') + ') | '
                     + row['state'] + ' | ' + row['evidence'] + ' | ' + row['remaining'] + ' |')
    lines.extend(['', '三条主线均以提升准确率为目标；负结果可完成实验，不自动进入最终模型。',
                  '传感器身份、波长质量、独立支持集/空间重复等缺口仍开放；整合验收不替代这些研究。', ''])
    return '\n'.join(lines)


def validate_status(value, root):
    if (value.get('schema_version') != 1 or value.get('main_merge') != 'pending_review'
            or value.get('ci_runner') not in ('registration_required', 'registered')
            or value.get('real_training_started_by_this_task') is not False):
        raise ValueError('Delivery scope/status differs; register an explicit new status policy')
    root = Path(root).resolve()
    modules = value['modules']
    if not modules or len({m['id'] for m in modules}) != len(modules):
        raise ValueError('Unique nonempty module status records required')
    for sha in [value['main_snapshot'], value['frozen_runtime_reference'], *value['required_ancestors'].values()]:
        if not re.fullmatch('[0-9a-f]{40}', sha):
            raise ValueError('Full immutable SHA required')
    for relative in [*value['required_files'], *(m['report'] for m in modules)]:
        path = (root / relative).resolve()
        if root not in path.parents or not path.is_file():
            raise ValueError('Missing or escaping delivery path: ' + relative)
    for row in modules:
        for field in ('id', 'state', 'evidence', 'remaining'):
            if not isinstance(row[field], str) or not row[field] or any(c in row[field] for c in '|\n\r'):
                raise ValueError('Single-line explicit status cells required')


def check_repository(root, expected_commit=None):
    root = Path(root).resolve()
    def git(*args):
        return subprocess.check_output(['git', '-C', str(root), *args], text=True).strip()
    head = git('rev-parse', 'HEAD')
    if expected_commit is not None and head != expected_commit:
        raise ValueError('Unexpected checkout SHA')
    if git('status', '--porcelain', '--untracked-files=no'):
        raise ValueError('Tracked checkout modifications present')
    if git('rev-parse', '--is-shallow-repository') != 'false':
        raise ValueError('Full history required for regression fixtures and delivery evidence')
    value = json.loads((root / STATUS_PATH).read_text(encoding='utf-8'))
    validate_status(value, root)
    for name, sha in value['required_ancestors'].items():
        if subprocess.run(['git', '-C', str(root), 'merge-base', '--is-ancestor', sha, head],
                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode:
            raise ValueError('Required repair/module not included: ' + name)
    for path in value['frozen_runtime_paths']:
        if path not in ('model', 'loadData', 'configs/experiments') and not path.endswith('.py'):
            raise ValueError('Unexpected runtime guard path')
    if git('diff', '--name-only', value['frozen_runtime_reference'], '--', *value['frozen_runtime_paths']):
        raise ValueError('Frozen model/training/data defaults changed during delivery integration')
    actual = (root / SUMMARY_PATH).read_text(encoding='utf-8')
    if actual != render_status(value):
        raise ValueError('Current status Markdown is stale; render it remotely from the JSON')
    for path in PLAN_PATHS:
        text = (root / path).read_text(encoding='utf-8')
        if text.count('<!-- CURRENT-PROGRESS-START -->') != 1:
            raise ValueError('One current-status pointer required in ' + path)
        pointer = text.split('<!-- CURRENT-PROGRESS-START -->', 1)[1].split('<!-- CURRENT-PROGRESS-END -->', 1)[0]
        if '当前搭建状态与审查整改清单_20261102.md' not in pointer or '| 任务 |' in pointer:
            raise ValueError('Duplicated current status table in ' + path)
    files = git('ls-files', '-z').split('\0')
    python_count = json_count = 0
    for name in files:
        path = root / name
        if path.suffix == '.py':
            ast.parse(path.read_text(encoding='utf-8-sig'), filename=name)
            python_count += 1
        elif path.suffix == '.json':
            json.loads(path.read_text(encoding='utf-8-sig'))
            json_count += 1
    return {'status': 'passed', 'commit': head, 'branch': git('branch', '--show-current'),
            'python_files': python_count, 'json_files': json_count,
            'included_modules': list(value['required_ancestors']),
            'frozen_runtime_unchanged': True, 'main_merged': False,
            'ci_runner': value['ci_runner'], 'new_real_training': False}
