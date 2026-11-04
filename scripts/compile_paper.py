#!/usr/bin/env python3
"""Automated LaTeX compilation and verification script for the SKARM manuscript.

Compiles paper/paper.tex using MiKTeX pdflatex twice, verifying:
- 0 fatal LaTeX errors
- 0 undefined citations
- 0 multiply-defined labels
- Detection and audit of any overfull \\hbox occurrences
- Successful generation of paper/paper.pdf
"""

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

DEFAULT_PDFLATEX_PATH = Path(
    r"C:\Users\ak612\AppData\Local\Programs\MiKTeX\miktex\bin\x64\miktex-pdftex.exe"
)


def find_pdflatex():
    """Locate pdflatex executable."""
    # 1. Specified environment variable
    env_path = os.environ.get("PDFLATEX_BIN")
    if env_path and Path(env_path).is_file():
        return Path(env_path)

    # 2. Known MiKTeX installation path on Windows
    if DEFAULT_PDFLATEX_PATH.is_file():
        return DEFAULT_PDFLATEX_PATH

    # 3. System PATH lookup for pdflatex or miktex-pdftex
    which_pdftex = shutil.which("miktex-pdftex") or shutil.which("pdflatex")
    if which_pdftex:
        return Path(which_pdftex)

    raise FileNotFoundError(
        "Could not find pdflatex or miktex-pdftex.exe. "
        "Set PDFLATEX_BIN or install MiKTeX/TeX Live."
    )


def compile_file(tex_file: Path, repo_root: Path, passes: int = 2) -> bool:
    paper_dir = repo_root / "paper"
    pdf_file = paper_dir / f"{tex_file.stem}.pdf"
    log_file = paper_dir / f"{tex_file.stem}.log"

    if not tex_file.is_file():
        print(f"Error: LaTeX source file not found at {tex_file}", file=sys.stderr)
        return False

    pdflatex_bin = find_pdflatex()
    print(f"\n================ Compiling: {tex_file.name} ================")
    print(f"Using LaTeX engine: {pdflatex_bin}")
    print(f"Working directory:  {paper_dir}")
    print(f"Passes:             {passes}")

    cmd = [
        str(pdflatex_bin),
        "-undump=pdflatex",
        "-interaction=nonstopmode",
        tex_file.name,
    ]

    for p in range(1, passes + 1):
        print(f"--- Pass {p} of {passes} ---")
        res = subprocess.run(
            cmd,
            cwd=str(paper_dir),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        print(f"Pass {p} process exited with code {res.returncode}")
        if res.returncode != 0:
            print(f"Warning: Pass {p} returned non-zero exit code {res.returncode}")

    if not log_file.is_file():
        print(f"Error: Log file not found at {log_file}", file=sys.stderr)
        return False

    log_content = log_file.read_text(encoding="utf-8", errors="replace")

    # Analyze log file
    fatal_errors = re.findall(r"^!\s+(.*?)$", log_content, flags=re.M)
    undef_citations = re.findall(
        r"LaTeX Warning: Citation `(.*?)' on page \d+ undefined", log_content
    )
    mult_labels = re.findall(
        r"LaTeX Warning: Label `(.*?)' multiply defined", log_content
    )
    # Match all Overfull \hbox occurrences (paragraphs, alignments, page splits)
    overfull_boxes_raw = re.findall(
        r"Overfull \\hbox \(([0-9.]+)pt too wide\)",
        log_content,
    )
    overfull_boxes = [float(w) for w in overfull_boxes_raw]
    max_overfull_pt = max(overfull_boxes, default=0.0)

    # Layout criteria: reject severe overfull boxes that spill into margins (> 3.0 pt)
    MAX_OVERFULL_THRESHOLD_PT = float(os.environ.get("MAX_OVERFULL_PT", "3.0"))
    overfull_severe = [w for w in overfull_boxes if w > MAX_OVERFULL_THRESHOLD_PT]
    overfull_ok = (len(overfull_severe) == 0)

    print("\n================ COMPILATION AUDIT REPORT ================")
    print(f"Document:                  {tex_file.name}")
    print(f"Fatal LaTeX Errors:        {len(fatal_errors)}")
    for err in fatal_errors:
        print(f"  [ERROR] {err}")

    print(f"Undefined Citations:       {len(undef_citations)}")
    for cite in undef_citations:
        print(f"  [WARNING] Undefined citation: {cite}")

    print(f"Multiply-Defined Labels:   {len(mult_labels)}")
    for lbl in mult_labels:
        print(f"  [WARNING] Multiply-defined label: {lbl}")

    print(f"Overfull \\hbox Warnings:   {len(overfull_boxes)} (Max: {max_overfull_pt:.2f} pt)")
    if overfull_severe:
        print(f"  [ALERT] {len(overfull_severe)} overfull boxes exceed layout threshold ({MAX_OVERFULL_THRESHOLD_PT:.1f} pt):")
        for w in overfull_severe[:10]:
            print(f"    - {w:.2f} pt too wide")
        if len(overfull_severe) > 10:
            print(f"    ... and {len(overfull_severe) - 10} more.")

    pdf_ok = pdf_file.is_file() and pdf_file.stat().st_size > 0
    if pdf_ok:
        size_kb = pdf_file.stat().st_size / 1024
        print(f"Generated PDF:             {pdf_file} ({size_kb:.1f} KB)")
    else:
        print(f"Generated PDF:             FAILED (missing or empty)")

    print("==========================================================")

    passed = (
        len(fatal_errors) == 0
        and len(undef_citations) == 0
        and len(mult_labels) == 0
        and overfull_ok
        and pdf_ok
    )

    if passed:
        print(f"[SUCCESS] {tex_file.name} compiled and verified cleanly with 0 defects.")
    else:
        print(f"[FAILURE] {tex_file.name} verification failed (errors, citations, or overfull layout defects).")

    return passed


def main():
    repo_root = Path(__file__).resolve().parent.parent
    paper_dir = repo_root / "paper"
    # Primary canonical manuscript
    ok1 = compile_file(paper_dir / "paper.tex", repo_root, passes=2)
    # IEEE transaction manuscript
    ok2 = True
    if (paper_dir / "paper-ieee.tex").is_file():
        ok2 = compile_file(paper_dir / "paper-ieee.tex", repo_root, passes=2)
    sys.exit(0 if (ok1 and ok2) else 1)


if __name__ == "__main__":
    main()
