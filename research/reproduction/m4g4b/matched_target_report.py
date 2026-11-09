"""Export full54 train/validation curves only after independent complete acceptance."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from g4_matched_target_review import export_review
from research_runtime import code_snapshot


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("expected-commit", "run", "launch-sha256", "acceptance-sha256", "audit", "audit-sha256", "out"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--resource-audit")
    parser.add_argument("--resource-audit-sha256")
    args = parser.parse_args()
    code = code_snapshot()
    if sys.platform != "linux" or not __debug__ or code["dirty"] or code["commit"] != args.expected_commit:
        raise RuntimeError("Clean expected remote review exporter required")
    export_review(args.run, args.launch_sha256, args.acceptance_sha256,
        args.audit, args.audit_sha256, args.out, code,
        resource_audit_path=args.resource_audit, resource_audit_sha256=args.resource_audit_sha256)


if __name__ == "__main__":
    main()
