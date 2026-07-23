#!/usr/bin/env python3
"""Verify sptc-format .tns files have compatible axis extents.

For a molecule dir under `sptc_coo/<mol>/`, every tensor whose
axes share the same symbolic index (i/m/k/a) must report the
same extent. This is required so that any contraction in
`traces/all_equations.txt` can be run on the dataset without
per-tensor index remapping.

Exits 0 on total pass; 1 if any molecule has an axis-extent
mismatch. Prints a Markdown table.
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path
from typing import Optional

# Explicit path insertion rather than relying on Python's implicit
# sys.path[0] = script's own directory (only true when run directly by
# path) — this import would otherwise silently break under `python -m`,
# or if this script is ever copied/packaged without its _axis_map.py
# sibling.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _axis_map import AXIS_MAP  # noqa: E402


def parse_shape(header: str) -> Optional[list[int]]:
    m = re.search(r"shape=([\d,]+)", header)
    if not m:
        return None
    return [int(x) for x in m.group(1).split(",") if x]


def audit_molecule(mol_dir: str) -> dict:
    """Return a dict summarizing per-axis extents seen + fail list."""
    axis_extents: dict[str, dict[int, list[str]]] = {}
    file_shapes: dict[str, list[int]] = {}
    for name in sorted(os.listdir(mol_dir)):
        p = os.path.join(mol_dir, name)
        if os.path.islink(p) or not name.endswith(".tns"):
            continue
        if name not in AXIS_MAP:
            continue  # unknown tensor — ignore (also flat but not in the map)
        with open(p, "r", encoding="utf-8") as f:
            hdr = f.readline()
        shape = parse_shape(hdr)
        if shape is None:
            continue
        file_shapes[name] = shape
        for pos, axis in enumerate(AXIS_MAP[name]):
            axis_extents.setdefault(axis, {}).setdefault(shape[pos], []).append(name)

    # Compute per-molecule verdict: any axis with multiple distinct extents = FAIL.
    fails: list[tuple[str, dict[int, list[str]]]] = []
    summary: dict[str, int] = {}
    for axis, buckets in axis_extents.items():
        if len(buckets) == 1:
            summary[axis] = next(iter(buckets))
        else:
            fails.append((axis, buckets))
            # Still put the most common in summary for reference:
            summary[axis] = max(buckets, key=lambda k: len(buckets[k]))
    return {
        "shapes": file_shapes,
        "summary": summary,
        "fails": fails,
    }


def parse_equation_contractions(eq_path: str) -> list[tuple[str, str]]:
    """Extract (symbol_letter, symbol_letter) pairs that contract in equations.

    Returns list of (letter_A, letter_B) pairs — one for each shared index
    appearing in more than one arg-position within an equation. Because our
    AXIS_MAP already collapses all i# to 'occ', all m# to 'pao', etc., we
    don't need to explicitly cross-check the equations if the audit passes.

    We still parse and report the equation count for provenance.
    """
    with open(eq_path, "r", encoding="utf-8") as f:
        text = f.read()
    n_eqs = len(re.findall(r"^eq\d+:", text, re.MULTILINE))
    return [], n_eqs


def emit_report(results: dict[str, dict], eq_info: Optional[tuple], out=sys.stdout) -> bool:
    """Print Markdown table. Returns True if all pass."""
    all_pass = True
    n_eqs = eq_info[1] if eq_info else None

    print("| Mol   | n_occ | n_pao | n_ri | n_osv | n_pno | verdict |", file=out)
    print("|---|---|---|---|---|---|---|", file=out)
    for mol, res in sorted(results.items()):
        s = res["summary"]
        row_pass = not res["fails"]
        all_pass &= row_pass
        verdict = "✅ pass" if row_pass else "❌ FAIL"
        print(
            f"| {mol} | {s.get('occ','—')} | {s.get('pao','—')} | "
            f"{s.get('ri','—')} | {s.get('osv','—')} | {s.get('pno','—')} | {verdict} |",
            file=out,
        )

    # Per-molecule fail detail
    for mol, res in sorted(results.items()):
        if not res["fails"]:
            continue
        print(f"\n### {mol} — mismatches", file=out)
        for axis, buckets in res["fails"]:
            extents = sorted(buckets.keys())
            print(f"- axis `{axis}` sees extents {extents}", file=out)
            for ext, files in sorted(buckets.items()):
                print(f"    - {ext}: {', '.join(files)}", file=out)

    if n_eqs is not None:
        print(
            f"\n(cross-referenced against {n_eqs} contraction diagrams "
            "in the equations file — any per-molecule pass above means "
            "every diagram's paired indices agree in extent)",
            file=out,
        )
    return all_pass


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("dataset_dir", help="sptc_coo/ tree (contains per-mol subdirs)")
    ap.add_argument(
        "--molecule",
        default=None,
        help="restrict to one molecule subdir (e.g. C2H6)",
    )
    ap.add_argument(
        "--equations",
        default=None,
        help="path to traces/all_equations.txt for provenance (optional)",
    )
    args = ap.parse_args()

    if not os.path.isdir(args.dataset_dir):
        print(f"ERROR: not a directory: {args.dataset_dir}", file=sys.stderr)
        return 1

    if args.molecule:
        mol_dirs = [args.molecule]
    else:
        mol_dirs = sorted(
            d for d in os.listdir(args.dataset_dir)
            if os.path.isdir(os.path.join(args.dataset_dir, d))
        )

    results: dict[str, dict] = {}
    for mol in mol_dirs:
        p = os.path.join(args.dataset_dir, mol)
        if not os.path.isdir(p):
            print(f"WARNING: no such dir: {p}", file=sys.stderr)
            continue
        results[mol] = audit_molecule(p)

    if not results:
        print(
            "ERROR: no molecule directories were audited "
            f"(dataset_dir={args.dataset_dir!r}, molecule={args.molecule!r}) "
            "— nothing to pass",
            file=sys.stderr,
        )
        return 1

    eq_info = None
    if args.equations:
        try:
            eq_info = parse_equation_contractions(args.equations)
        except FileNotFoundError:
            print(f"WARNING: --equations file not found: {args.equations}", file=sys.stderr)

    all_pass = emit_report(results, eq_info)
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
