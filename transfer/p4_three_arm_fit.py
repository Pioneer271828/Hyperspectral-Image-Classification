"""Register, execute or review the explicit HongHu three-arm fitting scope."""
import argparse
import json
from pathlib import Path

from research_runtime import code_snapshot, hash_file
from transfer import p4_coverage_batch as batch
from transfer import p4_full_budget_campaign as campaign
from transfer import p4_full_budget_registration as registration
from transfer import p4_runtime_device as device
from transfer import p4_three_arm_contract as contract, p4_three_arm_review


def prepare(parent, prior, out, execution, code, *, variant='real', classes=None, gpus=(), qualifications=(),
            reference=None, release=None):
    synthetic = variant != 'real'
    fit = contract.contract(variant, classes)
    destination = batch.sibling(execution, parent)
    device.require(destination != Path(out).resolve() and not destination.exists(),
                   'Fitting requires its own new execution directory')
    fit['execution'] = str(destination)
    fit['release'] = release
    steps, interval, _ = contract.STEPPING[variant]
    return registration.prepare(parent, hash_file(parent) if synthetic else batch.PARENT_SHA,
        prior, hash_file(prior) if synthetic else batch.PRIOR_SHA, out, code,
        protocol=device.CUDA_FP64_WEIGHT if gpus else device.CPU, gpus=gpus, synthetic=synthetic,
        steps=steps, interval=interval, purpose=contract.PURPOSE, qualifications=qualifications,
        fit=fit, fit_reference=reference)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest='mode', required=True)
    prep = modes.add_parser('prepare', help='Freeze the real 9-case preflight or an approved launch')
    for key in ('parent', 'prior', 'reference', 'registration-dir', 'execution-dir'):
        prep.add_argument('--' + key, type=Path, required=True)
    prep.add_argument('--gpus', required=True, nargs='+')
    prep.add_argument('--qualification', required=True, type=Path, action='append')
    prep.add_argument('--release', type=Path)
    prep.add_argument('--release-sha256')
    run = modes.add_parser('run')
    run.add_argument('--resume', action='store_true')
    accept = modes.add_parser('accept')
    review = modes.add_parser('review')
    for mode in (run, accept, review):
        mode.add_argument('--scope', required=True, type=Path)
        mode.add_argument('--scope-sha256', required=True)
        mode.add_argument('--execution-dir', required=True, type=Path)
    review.add_argument('--acceptance', required=True, type=Path)
    review.add_argument('--acceptance-sha256', required=True)
    review.add_argument('--out', required=True, type=Path)
    for mode in modes.choices.values():
        mode.add_argument('--expected-commit', required=True)
    args = parser.parse_args()
    code = code_snapshot()
    device.require(code['commit'] == args.expected_commit, 'Unexpected fitting producer commit')

    if args.mode == 'prepare':
        device.require((args.release is None) == (args.release_sha256 is None),
                       'Provide the separately reviewed fitting release and hash together')
        release = ({'path': str(args.release.resolve()), 'sha256': args.release_sha256} if args.release else None)
        path = prepare(args.parent, args.prior, args.registration_dir, args.execution_dir, code,
                       reference=args.reference, gpus=args.gpus, qualifications=args.qualification, release=release)
    else:
        scope, _ = registration.load(args.scope, args.scope_sha256, code)
        device.require(scope.get('purpose') == contract.PURPOSE, 'Expected the three-arm fitting scope')
        call = (args.scope, args.scope_sha256, args.execution_dir)
        if args.mode == 'run':
            print(json.dumps(campaign.run(*call, code, resume=args.resume)), flush=True)
            return
        if args.mode == 'accept':
            path = campaign.accept(*call, code)
        else:
            path = p4_three_arm_review.review(*call, args.acceptance, args.acceptance_sha256, args.out, code)
    print(json.dumps({'path': str(path), 'sha256': hash_file(path)}), flush=True)


if __name__ == '__main__':
    main()
