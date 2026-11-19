"""Import only fully pinned WP05–WP07 scopes and keep original WP12 table bytes."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation import wp12_mechanism_admission as evidence
from research_runtime import code_snapshot, external_directory, require_frozen_code


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--runs", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    kinds = ("crash", "learning", "bn", "region")
    for key in (*kinds, "authorities"):
        parser.add_argument("--" + key, nargs=2)
    extension = ("calibration", "legacy-region")
    for key in extension:
        parser.add_argument("--" + key, nargs=2)
    parser.add_argument("--band-graph", nargs=2, metavar=("DIRECTORY", "REGISTRATION_SHA256"))
    parser.add_argument("--donor-retrieval", nargs=2, metavar=("DIRECTORY", "REGISTRATION_SHA256"))
    parser.add_argument("--band-scale", nargs=2, metavar=("DIRECTORY", "REGISTRATION_SHA256"))
    args = parser.parse_args()
    if args.band_scale:
        if args.band_graph or args.donor_retrieval or any(getattr(args,key.replace("-", "_")) for key in (*kinds,"authorities",*extension)):
            parser.error("Band-scale intake requires its own pinned registration")
        from evaluation.wp12_band_scale import publish
        code = code_snapshot()
        require_frozen_code(code)
        root = args.runs.resolve(strict=True)
        folder = Path(args.band_scale[0]).resolve(strict=True)
        out = external_directory(args.out)
        evidence.require(code["commit"]==args.expected_commit and root in out.parents and root in folder.parents,
                         "Band-scale intake must stay under the approved run directory")
        print(publish(root,folder,out,code,args.band_scale[1]),flush=True)
        return
    if args.donor_retrieval:
        if args.band_graph or any(getattr(args,key.replace("-", "_")) for key in (*kinds,"authorities",*extension)):
            parser.error("Donor retrieval cannot be mixed with other pinned evidence")
        from evaluation.wp12_donor_retrieval import publish
        code = code_snapshot()
        require_frozen_code(code)
        root = args.runs.resolve(strict=True)
        folder = Path(args.donor_retrieval[0]).resolve(strict=True)
        out = external_directory(args.out)
        evidence.require(code["commit"] == args.expected_commit and root in out.parents and root in folder.parents,
                         "Donor evidence must stay within the authorized run directory")
        print(publish(folder,out,code,args.donor_retrieval[1]),flush=True)
        return
    if args.band_graph:
        if any(getattr(args, key.replace("-", "_")) for key in (*kinds, "authorities", *extension)):
            parser.error("Band-graph registration cannot be mixed with legacy receipts")
        from evaluation.wp12_band_graph import publish
        code = code_snapshot()
        require_frozen_code(code)
        root = args.runs.resolve(strict=True)
        folder = Path(args.band_graph[0]).resolve(strict=True)
        out = external_directory(args.out)
        evidence.require(code["commit"] == args.expected_commit and root in out.parents and root in folder.parents,
                         "Band-graph evidence must stay within the authorized run directory")
        print(publish(folder,out,code,args.band_graph[1]),flush=True)
        return
    if any(getattr(args,key) is None for key in (*kinds,"authorities")):
        parser.error("Legacy WP12 requires --crash, --learning, --bn, --region and --authorities")
    if any(getattr(args, key.replace("-", "_")) for key in extension):
        if not all(getattr(args, key.replace("-", "_")) for key in extension):
            parser.error("Real calibration and legacy region receipts must both be pinned")
        kinds += extension
    code = code_snapshot()
    require_frozen_code(code)
    root, out = args.runs.resolve(strict=True), external_directory(args.out)
    evidence.require(code["commit"] == args.expected_commit and root in out.parents, "Invalid adapter revision or output path")
    pin = lambda key: dict(zip(("path", "sha256"), getattr(args, key.replace("-", "_")), strict=True))
    result = evidence.admit(out, root, {key: pin(key) for key in kinds}, pin("authorities"),
                            {k: code[k] for k in ("commit", "source_hash")})
    print(result, flush=True)


if __name__ == "__main__":
    main()
