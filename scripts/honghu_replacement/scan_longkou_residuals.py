from __future__ import annotations

import re
import sys
import zipfile
from pathlib import Path
from xml.etree import ElementTree

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from honghu_common import RESULTS_DIR, ROOT, ensure_dirs, write_csv  # noqa: E402


TERMS = [
    "WHU-Hi LongKou",
    "WHU-Hi-LongKou",
    "LongKou",
    "WHUL",
]

TEXT_SUFFIXES = {".txt", ".md", ".csv", ".py", ".json", ".tex"}
DOCX_SUFFIXES = {".docx"}
SKIP_PARTS = {".git", "__pycache__"}


def should_skip(path: Path) -> bool:
    if path.resolve() == (RESULTS_DIR / "residual_longkou_terms.csv").resolve():
        return True
    return any(part in SKIP_PARTS for part in path.parts)


def context(text: str, start: int, end: int, width: int = 72) -> str:
    lo = max(0, start - width)
    hi = min(len(text), end + width)
    return re.sub(r"\s+", " ", text[lo:hi]).strip()


def scan_text(path: Path) -> list[dict]:
    try:
        content = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        try:
            content = path.read_text(encoding="gbk")
        except Exception:
            return []
    except Exception:
        return []
    rows = []
    lines = content.splitlines()
    for line_no, line in enumerate(lines, start=1):
        for term in TERMS:
            for match in re.finditer(re.escape(term), line, flags=re.IGNORECASE):
                rows.append(
                    {
                        "File": str(path.relative_to(ROOT)),
                        "Container": "",
                        "Line": line_no,
                        "Term": term,
                        "Context": context(line, match.start(), match.end()),
                    }
                )
    return rows


def docx_text_parts(path: Path) -> list[tuple[str, str]]:
    parts = []
    try:
        with zipfile.ZipFile(path) as archive:
            for name in archive.namelist():
                if not name.startswith("word/") or not name.endswith(".xml"):
                    continue
                try:
                    xml = archive.read(name)
                    root = ElementTree.fromstring(xml)
                except Exception:
                    continue
                texts = []
                for elem in root.iter():
                    if elem.tag.endswith("}t") and elem.text:
                        texts.append(elem.text)
                if texts:
                    parts.append((name, " ".join(texts)))
    except Exception:
        return []
    return parts


def scan_docx(path: Path) -> list[dict]:
    rows = []
    for container, text in docx_text_parts(path):
        for term in TERMS:
            for match in re.finditer(re.escape(term), text, flags=re.IGNORECASE):
                rows.append(
                    {
                        "File": str(path.relative_to(ROOT)),
                        "Container": container,
                        "Line": "",
                        "Term": term,
                        "Context": context(text, match.start(), match.end()),
                    }
                )
    return rows


def main() -> None:
    ensure_dirs()
    rows = []
    for path in sorted(ROOT.rglob("*")):
        if not path.is_file() or should_skip(path):
            continue
        suffix = path.suffix.lower()
        if suffix in TEXT_SUFFIXES:
            rows.extend(scan_text(path))
        elif suffix in DOCX_SUFFIXES:
            rows.extend(scan_docx(path))
    write_csv(RESULTS_DIR / "residual_longkou_terms.csv", rows, ["File", "Container", "Line", "Term", "Context"])
    print(f"Wrote {RESULTS_DIR / 'residual_longkou_terms.csv'} ({len(rows)} matches)")


if __name__ == "__main__":
    main()
