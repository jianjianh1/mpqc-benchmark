#!/usr/bin/env python3
"""For each catalog equation, find the (iter, term_idx) occurrence in a
trace with the most steps, and report whether it reaches the catalog's
full stage_count.

Why this exists: MPQC's CSE shares intermediate sub-expressions not
just across CCSD iterations (an equation's own repeat), but across
*different* catalog equations within the same iteration — e.g. if
diagram A's computation graph contains a sub-expression that diagram B
also needs, and A's term_idx executes first, B's term_idx will skip
recomputing that shared piece. A given equation's cheapest-looking
occurrence in a trace may just be the one that arrived after its
sharable sub-expressions were already computed by something else — not
a real steady-state cost. This script finds, per equation, the
occurrence closest to (or matching) the full unshared computation, and
flags whether that ceiling was actually reached anywhere in the trace.

CLI: python3 traces/best_full_occurrence.py --equations traces/all_equations.txt
     --log <path to raw MPQC log> --steps <path to <mol>-traced.steps.csv>
     --output-csv <path>
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

from term_begin_mapper import extract_term_begins, parse_catalog_trees, per_term_wall


def row_counts_per_group(steps_csv: Path) -> dict[tuple[int, int], int]:
    """Count steps.csv rows per (iter, term_idx) — the raw stage count
    actually executed for that occurrence (post-CSE, whatever MPQC
    decided didn't need recomputing)."""
    counts: dict[tuple[int, int], int] = defaultdict(int)
    with steps_csv.open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            try:
                it = int(r["iter"])
                tid = int(r["term_idx"])
            except (KeyError, ValueError):
                continue
            counts[(it, tid)] += 1
    return counts


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--equations", type=Path, required=True)
    ap.add_argument("--log", type=Path, required=True)
    ap.add_argument("--steps", type=Path, required=True)
    ap.add_argument("--output-csv", type=Path, required=True)
    args = ap.parse_args()

    catalog, stage_counts = parse_catalog_trees(args.equations)
    by_key: dict[tuple[int, str], list[str]] = defaultdict(list)
    for eq_id, (coef, key) in catalog.items():
        by_key[(coef, key)].append(eq_id)

    # A (coef, tree) key shared by >1 eq_id means every occurrence of that
    # key resolves to the numerically-lowest eq_id below (see the
    # `sorted(eqs, key=lambda x: int(x[2:]) ...)[0]` pick in the loop below
    # — sorted by the integer after "eq", NOT string order, so e.g. eq5
    # sorts after eq30) — every OTHER eq_id sharing the key gets zero
    # occurrences and looks indistinguishable from a diagram that never
    # fires. term_begin_mapper.py already flags this same collision; do
    # the same here instead of resolving it silently.
    dup_keys = [(k, eqs) for k, eqs in by_key.items() if len(eqs) > 1]
    if dup_keys:
        print(
            f"WARNING: {len(dup_keys)} (coef, tree) keys are shared by "
            "multiple eq_ids; all but the numerically-lowest eq_id in "
            "each group will show 0 occurrences below:",
            file=sys.stderr,
        )
        for k, eqs in dup_keys:
            print(
                f"    {sorted(eqs, key=lambda x: int(x[2:]) if x[2:].isdigit() else 999)}",
                file=sys.stderr,
            )

    term_begins = extract_term_begins(args.log)
    counts = row_counts_per_group(args.steps)
    walls = per_term_wall(args.steps)  # disk-load excluded, every iter/occurrence

    # For each matched eq_id, track every occurrence's (row_count, iter, tid).
    occurrences: dict[str, list[tuple[int, int, int]]] = defaultdict(list)
    for (it, tid), (coef, tree_key) in term_begins.items():
        eqs = by_key.get((coef, tree_key), [])
        if not eqs:
            continue
        eq_id = sorted(eqs, key=lambda x: int(x[2:]) if x[2:].isdigit() else 999)[0]
        occurrences[eq_id].append((counts.get((it, tid), 0), it, tid))

    rows = []
    n_complete = 0
    for eq_id in sorted(catalog, key=lambda x: int(x[2:]) if x[2:].isdigit() else 999):
        expected = stage_counts.get(eq_id, 0)
        occs = occurrences.get(eq_id, [])
        if not occs:
            rows.append([eq_id, expected, "", "", 0, False, ""])
            continue
        best_count, best_it, best_tid = max(occs)
        complete = best_count >= expected and expected > 0
        wall_ns = walls.get((best_it, best_tid), 0)
        if complete:
            n_complete += 1
        rows.append(
            [eq_id, expected, best_it, best_tid, best_count, complete, round(wall_ns / 1e6, 4)]
        )

    n_matched = len(occurrences)
    print(
        f"{n_matched} eqs matched at least once; "
        f"{n_complete}/{n_matched} reach full stage_count somewhere in the trace",
        file=sys.stderr,
    )

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            ["eq_id", "stage_count", "best_iter", "best_term_idx", "best_row_count", "complete", "wall_ms"]
        )
        w.writerows(rows)
    print(f"wrote {args.output_csv}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
