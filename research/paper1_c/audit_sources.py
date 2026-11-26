"""Describe fixed-source history without importing unlicensed external code."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import subprocess

from pilot import git_clean, write_new

CEGCN = "e5bc8a007a7179d2092aa1cc1c0993ac1a171d4c"
BASE = "a339ce8d7e5f0a600363b1a8826b96d8a16ae794"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cegcn", required=True)
    p.add_argument("--expected-commit", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    root = Path(__file__).resolve().parents[2]
    git_clean(root, args.expected_commit)
    upstream = subprocess.check_output(["git", "-C", args.cegcn, "show", CEGCN+":CEGCN.py"])
    tree = subprocess.check_output(["git", "-C", args.cegcn, "ls-tree", "-r", "--name-only", CEGCN], text=True).splitlines()
    upstream_ast = ast.parse(upstream)
    methods = {(c.name, m.name): ast.dump(m, include_attributes=False) for c in upstream_ast.body
               if isinstance(c, ast.ClassDef) for m in c.body if isinstance(m, ast.FunctionDef)}
    rows = {}
    for path in ("model/WFCG.py", "model/SAHGCN.py", "model/VCASAHGCN.py"):
        previous = subprocess.check_output(["git", "show", BASE+":"+path], cwd=root)
        raw = (root/path).read_bytes()
        if raw != previous:
            raise ValueError("C changed a historical project model")
        current_ast = ast.parse(raw)
        exact = []
        for c in current_ast.body:
            if not isinstance(c, ast.ClassDef):
                continue
            for m in c.body:
                if isinstance(m, ast.FunctionDef):
                    hits = [".".join(k) for k,v in methods.items() if v == ast.dump(m, include_attributes=False)]
                    if hits:
                        exact.append({"local": c.name+"."+m.name, "upstream_methods": hits})
        rows[path] = {"sha256": hashlib.sha256(raw).hexdigest(), "unchanged_from_base": True,
                      "exact_method_ast_matches": exact,
                      "embedded_source_links": [s.strip() for s in raw.decode().splitlines() if "github.com" in s]}
    result = {"code": args.expected_commit, "cegcn": {"commit": CEGCN,
              "model_sha256": hashlib.sha256(upstream).hexdigest(),
              "license_paths": [s for s in tree if any(t in s.lower() for t in ("license", "copying", "notice"))],
              "executed": False, "newly_copied_into_project": False}, "historical_project_files": rows,
              "interpretation": "Exact syntax or embedded attribution describes historical overlap, not complete provenance or permission. No historical license gap is cured by this adapter.",
              "external_incorporation": "DSNet model is loaded only from separately pinned GPL3 author checkout; no external file is vendored."}
    write_new(args.output, result)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
