"""Static integrity and naming audit for the public CMPB release."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import re
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path


FORBIDDEN = re.compile(
    "|".join(
        [
            r"J" + r"BI",
            r"Journal of Bio" + r"medical Informatics",
            r"MKG-" + r"J" + r"BI",
        ]
    ),
    re.IGNORECASE,
)
TEXT_SUFFIXES = {
    "", ".cff", ".csv", ".gitignore", ".json", ".md", ".ps1", ".py",
    ".txt", ".yaml", ".yml",
}
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def tracked_files(root: Path) -> list[Path]:
    result = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=root,
        check=True,
        capture_output=True,
    )
    return [root / item.decode("utf-8") for item in result.stdout.split(b"\0") if item]


def audit(root: Path) -> dict:
    files = tracked_files(root)
    failures: list[dict[str, str]] = []
    counts = {
        "tracked_files": len(files),
        "text_files": 0,
        "json_files": 0,
        "csv_files": 0,
        "pdf_files": 0,
        "png_files": 0,
        "svg_files": 0,
    }

    for path in files:
        rel = path.relative_to(root).as_posix()
        if FORBIDDEN.search(rel):
            failures.append({"file": rel, "check": "filename", "detail": "legacy journal token"})
        if not path.is_file():
            failures.append({"file": rel, "check": "existence", "detail": "missing tracked file"})
            continue
        data = path.read_bytes()
        if not data:
            failures.append({"file": rel, "check": "size", "detail": "zero-byte file"})
            continue

        suffix = path.suffix.lower()
        if suffix in TEXT_SUFFIXES:
            counts["text_files"] += 1
            try:
                text = data.decode("utf-8-sig")
            except UnicodeDecodeError as exc:
                failures.append({"file": rel, "check": "utf8", "detail": str(exc)})
                continue
            if FORBIDDEN.search(text):
                failures.append({"file": rel, "check": "content", "detail": "legacy journal token"})

        if suffix == ".json":
            counts["json_files"] += 1
            try:
                json.loads(data.decode("utf-8-sig"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                failures.append({"file": rel, "check": "json", "detail": str(exc)})
        elif suffix == ".csv":
            counts["csv_files"] += 1
            try:
                rows = [
                    row
                    for row in csv.reader(io.StringIO(data.decode("utf-8-sig"), newline=""))
                    if row
                ]
                if not rows:
                    raise ValueError("CSV has no rows")
                width = len(rows[0])
                if width == 0 or any(len(row) != width for row in rows[1:]):
                    raise ValueError("inconsistent CSV column count")
            except (UnicodeDecodeError, csv.Error, ValueError) as exc:
                failures.append({"file": rel, "check": "csv", "detail": str(exc)})
        elif suffix == ".pdf":
            counts["pdf_files"] += 1
            if not data.startswith(b"%PDF-") or b"%%EOF" not in data[-2048:]:
                failures.append({"file": rel, "check": "pdf", "detail": "invalid header or missing EOF"})
        elif suffix == ".png":
            counts["png_files"] += 1
            if not data.startswith(PNG_SIGNATURE) or data[-8:-4] != b"IEND":
                failures.append({"file": rel, "check": "png", "detail": "invalid signature or missing IEND"})
        elif suffix == ".svg":
            counts["svg_files"] += 1
            try:
                ET.fromstring(data.decode("utf-8-sig"))
            except (UnicodeDecodeError, ET.ParseError) as exc:
                failures.append({"file": rel, "check": "svg", "detail": str(exc)})

    required = [
        "code/run_cmpb_lock_pipeline.py",
        "code/conformal_ipcw_cmpb.py",
        "config/MKG_CMPB_SUBMISSION_LOCK_manifest.json",
        "results/conformal_ipcw_cmpb_results.json",
        "results/submission_lock/Table_CMPB_LOCK_weights.csv",
        "results/source_tables/Table_CMPB_LOCK_external_cindex.csv",
    ]
    for rel in required:
        if not (root / rel).is_file():
            failures.append({"file": rel, "check": "required", "detail": "missing CMPB release file"})

    for rel in [
        "config/MKG_CMPB_SUBMISSION_LOCK_manifest.json",
        "config/MKG_CMPB_SUBMISSION_ADDENDUM_manifest.json",
    ]:
        manifest_path = root / rel
        if not manifest_path.is_file():
            continue
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for target, expected in manifest.get("sha256", {}).items():
            target_path = root / target
            if not target_path.is_file():
                failures.append({"file": target, "check": "manifest hash", "detail": f"missing target from {rel}"})
                continue
            actual = hashlib.sha256(target_path.read_bytes()).hexdigest()
            if actual != expected:
                failures.append({"file": target, "check": "manifest hash", "detail": f"mismatch in {rel}"})

    return {
        "release": "MKG-CMPB-reproducibility",
        "counts": counts,
        "naming_check": "no legacy journal tokens in tracked paths or text files",
        "failures": failures,
        "passed": not failures,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = audit(args.root.resolve())
    rendered = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    print(rendered, end="")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
