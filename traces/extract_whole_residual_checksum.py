#!/usr/bin/env python3
"""Extract MPQC's own live whole-residual checksum(s) from a trace log.

Reads `Eval | WholeResidualChecksum | R=<r> | checksum=nnz,sum,sumsq,max_abs`
lines (added in mpqc-fork commit 1ef4e722f4). One line is printed per R per
call to evaluate_csv_closedshell -- i.e. once per CCSD iteration. The state
entering iteration 2 (T1=0, T2 from one deterministic pass off the zero
initial guess -- the only iteration boundary that's safe to compare across
separate MPQC invocations, since later iterations' PNO amplitudes have a
genuine run-to-run sign/gauge ambiguity) corresponds to the *second*
occurrence of each R's line, which is what this defaults to reporting.

Usage: extract_whole_residual_checksum.py <log_path> [--occurrence N]
"""
import argparse
import re

LINE_RE = re.compile(
    r"^Eval \| WholeResidualChecksum \| R=(?P<r>\d+) \| "
    r"checksum=(?P<nnz>-?\d+),(?P<sum>[^,]+),(?P<sumsq>[^,]+),(?P<max_abs>[^,]+)\s*$"
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log_path")
    ap.add_argument("--occurrence", type=int, default=2,
                     help="which occurrence (1-based) of each R's line to report (default: 2, "
                          "the state entering iteration 2 -- the only gauge-safe comparison point)")
    args = ap.parse_args()

    seen = {}
    with open(args.log_path, encoding="utf-8", errors="replace") as f:
        for line in f:
            m = LINE_RE.match(line.strip())
            if not m:
                continue
            r = int(m.group("r"))
            seen.setdefault(r, []).append({
                "nnz": int(m.group("nnz")),
                "sum": float(m.group("sum")),
                "sumsq": float(m.group("sumsq")),
                "max_abs": float(m.group("max_abs")),
            })

    for r in sorted(seen):
        occs = seen[r]
        print(f"R={r}: {len(occs)} occurrence(s) found")
        idx = args.occurrence - 1
        if 0 <= idx < len(occs):
            c = occs[idx]
            print(f"  occurrence {args.occurrence}: nnz={c['nnz']} sum={c['sum']!r} "
                  f"sumsq={c['sumsq']!r} max_abs={c['max_abs']!r}")
        else:
            print(f"  occurrence {args.occurrence} not found")


if __name__ == "__main__":
    main()
