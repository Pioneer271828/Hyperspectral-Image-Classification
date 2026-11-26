"""Read only image headers, delivered README and external source metadata in LXC.

No label image, sealed query, pixel spectrum, or classification GT is decoded.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import zipfile

from scipy.io import whosmat


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for data in iter(lambda: stream.read(1024**2), b""):
            h.update(data)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--project-root", required=True)
    p.add_argument("--references", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    if sys.platform != "linux" or subprocess.check_output(["systemd-detect-virt"], text=True).strip() != "lxc":
        raise RuntimeError("Use authorized LXC")
    root, refs = Path(args.project_root).resolve(), Path(args.references).resolve()
    images = {
        "paviaU": "datasets/current/PaviaU/PaviaU.mat",
        "xuzhou": "datasets/current/Xuzhou/xuzhou.mat",
        "whu-honghu": "datasets/incoming/base97a-honghu-issue97-20261103/WHU_Hi_HongHu.mat",
        "whu-longkou": "datasets/incoming/base97a-issue97-20261103/WHU_Hi_LongKou.mat",
        "whu-hanchuan": "datasets/current/WHU-Hi/WHU-Hi-HanChuan/WHU_Hi_HanChuan.mat",
    }
    rows = {}
    for name, relative in images.items():
        path = root / relative
        rows[name] = {"path": relative, "exists": path.exists()}
        if path.exists():
            rows[name].update(sha256=digest(path), bytes=path.stat().st_size,
                              image_variables=[{"name": n, "shape": list(s), "dtype": t} for n, s, t in whosmat(path)])
    readme = root/"datasets/current/Xuzhou/Xuzhou_Readme.txt"
    dsnet = refs/"dsnet"
    author = {"commit": subprocess.check_output(["git", "-C", str(dsnet), "rev-parse", "HEAD"], text=True).strip(),
              "files": {name: digest(dsnet/name) for name in ("model.py", "demo.py", "dataset.py", "utils.py", "README.md", "LICENSE")}}
    archive = refs/"dcg-fixed.zip"
    with zipfile.ZipFile(archive) as z:
        names = z.namelist()
        source = {}
        for name in names:
            if name.endswith((".py", ".cpp", ".m", "README.md", "README.txt")):
                data = z.read(name)
                text = data.decode("utf-8", errors="replace")
                source[name] = {"sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data),
                                "license_or_attribution_lines": [line for line in text.splitlines()
                                    if any(k in line.lower() for k in ("license", "copyright", "github", "borrow", "adapted"))]}
        dcg = {"commit": "2eb4e7b9fef602c2899cb2f6d596b95655b5bf3f", "zip_sha256": digest(archive),
               "members": len(names), "source": source,
               "license_members": [n for n in names if any(k in n.lower() for k in ("license", "copying", "notice"))],
               "array_members_decoded": 0, "programs_executed": 0}
    result = {"schema": "paper_c_readonly_asset_source_inventory_v1", "images": rows,
              "xuzhou_readme": {"sha256": digest(readme), "text": readme.read_text()},
              "dsnet": author, "dcg": dcg,
              "montello_pdf_sha256": digest(refs/"montello-dynamic-review.pdf"),
              "raw_gt_files_opened": 0, "query_files_opened": 0, "new_labels_released": 0}
    out = Path(args.output).resolve()
    if not out.is_relative_to(root/"runs"):
        raise ValueError("Evidence must stay in authorized runs")
    with out.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"output": str(out), "sha256": digest(out), "images": rows}))


if __name__ == "__main__":
    main()
