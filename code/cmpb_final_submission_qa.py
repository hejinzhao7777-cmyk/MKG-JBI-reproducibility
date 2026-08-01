"""Deterministic static QA for the flat CMPB submission directory."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


WORD = re.compile(r"[A-Za-z0-9]+(?:[.@'-][A-Za-z0-9]+)*")


def without_comments(text: str) -> str:
    return re.sub(r"(?<!\\)%.*", "", text)


def prose(text: str) -> str:
    text = without_comments(text)
    text = re.sub(r"\\begin\{(?:figure\*?|table\*?|equation\*?|align\*?)\}.*?\\end\{(?:figure\*?|table\*?|equation\*?|align\*?)\}", " ", text, flags=re.S)
    text = re.sub(r"\$\$.*?\$\$|\\\[.*?\\\]|\$.*?\$", " ", text, flags=re.S)
    text = re.sub(r"\\(?:cite|ref|eqref|url|path|includegraphics)(?:\[[^\]]*\])?\{[^}]*\}", " ", text)
    text = re.sub(r"\\[A-Za-z@]+\*?(?:\[[^\]]*\])?", " ", text)
    text = re.sub(r"[{}~&]", " ", text)
    return text


def count_words(text: str) -> int:
    return len(WORD.findall(prose(text)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    package = args.package.resolve()
    main_path = package / "mkg_cmpb.tex"
    supplement_path = package / "mkg_cmpb_supplement.tex"
    bib_path = package / "mkg_cmpb.bib"
    main = main_path.read_text(encoding="utf-8")
    supplement = supplement_path.read_text(encoding="utf-8")
    bib = bib_path.read_text(encoding="utf-8")

    abstract = main.split("\\begin{abstract}", 1)[1].split("\\end{abstract}", 1)[0]
    abstract = re.sub(r"\\textbf\{(?:Background and Objectives|Methods|Results|Conclusions):\}", "", abstract)
    core = main.split("\\section{Introduction}", 1)[1].split("\\section*{Ethics statement}", 1)[0]
    cite_keys = set()
    for source in (main, supplement):
        for group in re.findall(r"\\cite\{([^}]+)\}", source):
            cite_keys.update(key.strip() for key in group.split(","))
    bib_keys = set(re.findall(r"@\w+\{\s*([^,\s]+)", bib))
    figures = []
    for source_name, source in ((main_path.name, main), (supplement_path.name, supplement)):
        for name in re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", source):
            figures.append({"source": source_name, "name": name, "exists": (package / name).is_file()})

    highlights = [line.strip() for line in (package / "highlights.txt").read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    combined = "\n".join((main, supplement))
    forbidden = [
        "TODO", "TBD", "PLACEHOLDER", "INSERT HERE",
        "bootstrap stability", "Bootstrap stability", "150 projected-gradient",
        "0.288 for MKG", "MKG had the highest mean", "MKG clearly exceeded Cox-Lasso",
        "CITATION.cff} version 0.3.0", "8106e721f9d2",
    ]
    stale_hits = [term for term in forbidden if term in combined]
    log_checks = {}
    for stem in ("mkg_cmpb", "mkg_cmpb_supplement", "cover_letter_cmpb", "mkg_cmpb_title_page"):
        path = package / f"{stem}.log"
        if not path.exists():
            log_checks[stem] = {"exists": False}
            continue
        log = path.read_text(encoding="utf-8", errors="replace")
        patterns = ["Undefined control sequence", "LaTeX Error", "Citation `", "Reference `", "Overfull \\hbox", "Overfull \\vbox"]
        log_checks[stem] = {"exists": True, "hits": [pattern for pattern in patterns if pattern in log]}

    report = {
        "package": str(package),
        "abstract_words_excluding_labels": count_words(abstract),
        "main_words_introduction_through_conclusions_excluding_display_environments": count_words(core),
        "bib_entries": len(bib_keys),
        "unique_cited_keys": len(cite_keys),
        "missing_bib_keys": sorted(cite_keys - bib_keys),
        "uncited_bib_keys": sorted(bib_keys - cite_keys),
        "figures": figures,
        "missing_figures": [item for item in figures if not item["exists"]],
        "highlights_count": len(highlights),
        "highlight_lengths": [len(line) for line in highlights],
        "stale_text_hits": stale_hits,
        "log_checks": log_checks,
    }
    output = json.dumps(report, indent=2, ensure_ascii=False)
    print(output)
    if args.output:
        args.output.write_text(output + "\n", encoding="utf-8")
    failed = (
        report["abstract_words_excluding_labels"] > 250
        or report["main_words_introduction_through_conclusions_excluding_display_environments"] > 3500
        or len(bib_keys) > 50
        or bool(report["missing_bib_keys"])
        or bool(report["missing_figures"])
        or len(highlights) not in {3, 4, 5}
        or any(length > 85 for length in report["highlight_lengths"])
        or bool(stale_hits)
        or any(check.get("hits") for check in log_checks.values())
    )
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
