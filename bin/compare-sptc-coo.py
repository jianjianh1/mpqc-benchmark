#!/usr/bin/env python3
"""Three-tier comparison of two sptc-bench-format COO files.

Used to validate that our `bin/tns-to-sptc-coo.py` output matches
sptc-bench's reference dataset.

Tiers:
  1. shape — total dimensions agree
  2. sparsity — the set of nonzero coordinate tuples agrees
  3. values — at common coordinates, values agree within --tol

The tiers are layered: tier-2 only runs if tier-1 passes; tier-3 only
on common coordinates if tier-2 is partial. This is informative for
the (likely) case where the two pipelines (MPQC vs sptc-bench's
unknown generator) emit consistent SHAPES but divergent VALUES due to
basis-orientation, MO-phase, or PNO-truncation differences.

Output: markdown to stdout. Pipe to traces/SPTC_BENCH_COMPARISON.md.
"""
from __future__ import annotations

import argparse
import math
import os
import sys


def load_coo(path: str) -> tuple[list[int], dict[tuple[int, ...], float]]:
    """Parse a text-COO `.txt`. Returns (shape, {coord: value}).

    `shape` is taken from a `# shape=d0,d1,...` header line if present;
    otherwise computed as max+1 per axis (matches sptc-bench's
    `load_coo` semantics).
    """
    shape = None
    data: dict[tuple[int, ...], float] = {}
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            if line.startswith("#"):
                if "shape=" in line:
                    # `# shape=4,4,200 nnz=... ...`
                    for tok in line.lstrip("#").split():
                        if tok.startswith("shape="):
                            shape = [int(x) for x in tok[len("shape="):].split(",") if x]
                            break
                continue
            tokens = line.split()
            if len(tokens) < 2:
                raise ValueError(f"{path}: short data line: {line!r}")
            val = float(tokens[-1])
            coord = tuple(int(t) for t in tokens[:-1])
            data[coord] = val
    if shape is None and data:
        rank = len(next(iter(data)))
        shape = [max(c[d] for c in data) + 1 for d in range(rank)]
    return (shape or []), data


def fp_sums(data: dict[tuple[int, ...], float]) -> tuple[float, float]:
    """Return (Σ value, Σ value²) over the data — both order-invariant.

    Σ value² (Frobenius norm²) is also invariant under unitary
    rotations of any tensor axis, so it should match between two
    runs that differ only in MO-localization gauge.
    """
    s = 0.0
    s2 = 0.0
    for v in data.values():
        s += v
        s2 += v * v
    return s, s2


def compare_one(ours_path: str, theirs_path: str, tol: float) -> dict:
    """Compare two COO files and return a result dict for the report."""
    res = {"ours": ours_path, "theirs": theirs_path, "tol": tol}
    try:
        ours_shape, ours_data = load_coo(ours_path)
    except FileNotFoundError:
        res["status"] = "OURS_MISSING"
        return res
    try:
        theirs_shape, theirs_data = load_coo(theirs_path)
    except FileNotFoundError:
        res["status"] = "THEIRS_MISSING"
        return res

    res["ours_shape"] = ours_shape
    res["theirs_shape"] = theirs_shape
    res["ours_nnz"] = len(ours_data)
    res["theirs_nnz"] = len(theirs_data)

    # Tier 0: scalar checksums (order-invariant; cheap; informative
    # even when tiers 2/3 diverge). Σ value² is unitary-invariant so
    # it should match across MO-localization gauges; Σ value catches
    # missing rows when the bench-side data is suspect.
    ours_sum, ours_sum2 = fp_sums(ours_data)
    theirs_sum, theirs_sum2 = fp_sums(theirs_data)
    res["ours_sum"] = ours_sum
    res["theirs_sum"] = theirs_sum
    res["ours_sum2"] = ours_sum2
    res["theirs_sum2"] = theirs_sum2
    res["sum_diff"] = abs(ours_sum - theirs_sum)
    res["sum2_diff"] = abs(ours_sum2 - theirs_sum2)
    res["sum2_rel_diff"] = (res["sum2_diff"] / abs(theirs_sum2)
                            if theirs_sum2 != 0 else float("inf"))

    # Tier 1: shape
    if ours_shape != theirs_shape:
        res["tier1_shape"] = "FAIL"
        res["status"] = "SHAPE_DIVERGE"
        return res
    res["tier1_shape"] = "PASS"

    # Tier 2: sparsity (set of coordinate tuples)
    ours_coords = set(ours_data.keys())
    theirs_coords = set(theirs_data.keys())
    inter = ours_coords & theirs_coords
    only_ours = ours_coords - theirs_coords
    only_theirs = theirs_coords - ours_coords
    res["common_coords"] = len(inter)
    res["only_ours"] = len(only_ours)
    res["only_theirs"] = len(only_theirs)
    if only_ours == 0 and only_theirs == 0:
        res["tier2_sparsity"] = "PASS"
    else:
        res["tier2_sparsity"] = "PARTIAL"

    # Tier 3: values at common coordinates
    if not inter:
        res["tier3_values"] = "N/A (no common coords)"
        res["status"] = "NO_COMMON_COORDS"
        return res
    diffs = []
    max_abs = 0.0
    max_rel = 0.0
    for c in inter:
        a, b = ours_data[c], theirs_data[c]
        abs_diff = abs(a - b)
        max_abs = max(max_abs, abs_diff)
        if abs(b) > 0:
            max_rel = max(max_rel, abs_diff / abs(b))
        if abs_diff > tol:
            diffs.append((c, a, b, abs_diff))
    res["max_abs_diff"] = max_abs
    res["max_rel_diff"] = max_rel
    res["n_above_tol"] = len(diffs)
    if not diffs:
        res["tier3_values"] = "PASS"
    elif len(diffs) < 0.01 * len(inter):
        res["tier3_values"] = "MOSTLY-PASS (<1% above tol)"
    else:
        res["tier3_values"] = "DIVERGE"
    res["status"] = "OK"
    # Keep up to 5 example differences for the report
    diffs.sort(key=lambda d: -d[3])
    res["example_diffs"] = diffs[:5]
    return res


def write_markdown(results: list[dict], tol: float) -> None:
    print(f"# sptc-bench vs MPQC COO comparison\n")
    print(f"Tolerance for tier-3 value agreement: `{tol}` (abs diff).")
    print(f"Σv² (Frobenius norm²) is unitary-invariant and should match")
    print(f"across MO-localization gauges even when tier-3 diverges.\n")
    print("| tensor | shape | tier-1 | tier-2 | tier-3 | Σv (ours/theirs) | Σv² rel-diff | notes |")
    print("|---|---|---|---|---|---|---|---|")
    for r in results:
        name = os.path.basename(r.get("ours", "?"))
        if r.get("status") == "OURS_MISSING":
            print(f"| {name} | — | — | — | — | — | — | OURS_MISSING |")
            continue
        if r.get("status") == "THEIRS_MISSING":
            print(f"| {name} | {r.get('ours_shape')} | — | — | — | — | — | THEIRS_MISSING |")
            continue
        notes = []
        if r.get("only_ours"):
            notes.append(f"+{r['only_ours']} ours-only")
        if r.get("only_theirs"):
            notes.append(f"+{r['only_theirs']} theirs-only")
        if "max_rel_diff" in r:
            notes.append(f"max|Δ| {r['max_abs_diff']:.2e}")
        sums_cell = f"{r.get('ours_sum', 0):+.3e} / {r.get('theirs_sum', 0):+.3e}"
        sum2_rel = r.get('sum2_rel_diff', float('nan'))
        sum2_cell = f"{sum2_rel:.2e}"
        same_shape = r.get('ours_shape') == r.get('theirs_shape')
        shape_cell = (str(r.get('ours_shape'))
                      if same_shape
                      else f"ours={r.get('ours_shape')}, theirs={r.get('theirs_shape')}")
        print(
            f"| {name} | {shape_cell} | "
            f"{r.get('tier1_shape', '—')} | {r.get('tier2_sparsity', '—')} | "
            f"{r.get('tier3_values', '—')} | "
            f"{sums_cell} | {sum2_cell} | "
            f"{'; '.join(notes) or '—'} |"
        )

    # Append a "first divergent values" section if any
    interesting = [r for r in results if r.get("example_diffs")]
    if interesting:
        print("\n## Top divergent values (per tensor)\n")
        for r in interesting:
            print(f"### {os.path.basename(r['ours'])}")
            print("| coord | ours | theirs | |Δ| |")
            print("|---|---|---|---|")
            for c, a, b, d in r["example_diffs"]:
                print(f"| {c} | {a!r} | {b!r} | {d:.3e} |")
            print()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--ours",
        required=True,
        help="dir with our converted sptc-bench-format files (one molecule)",
    )
    ap.add_argument(
        "--theirs",
        required=True,
        help="sptc-bench dataset dir for the same molecule (e.g. .../C2H6/)",
    )
    ap.add_argument(
        "--tol",
        type=float,
        default=1e-7,
        help="absolute tolerance for tier-3 value agreement (default 1e-7)",
    )
    args = ap.parse_args()

    # Sptc-bench's released data uses `.txt`; this project's converter
    # emits `.tns`. Probe both suffixes per file so the comparator works
    # regardless of which dataset is on which side.
    bare_names = [
        "g_i_1_i_2_Κ_1",
        "g_i_1_m_1_Κ_1",
        "g_m_1_m_2_Κ_1",
        "C_m_1_a_1_i_1",
        "C_m_1_a_1_i_1_i_2",
    ]

    def pick(dirpath: str, bare: str) -> str:
        for ext in (".tns", ".txt"):
            p = os.path.join(dirpath, bare + ext)
            if os.path.exists(p):
                return p
        return os.path.join(dirpath, bare + ".txt")  # fall through to .txt for missing-message

    results = []
    for bare in bare_names:
        results.append(
            compare_one(
                pick(args.ours, bare),
                pick(args.theirs, bare),
                args.tol,
            )
        )
    write_markdown(results, args.tol)
    return 0


if __name__ == "__main__":
    sys.exit(main())
