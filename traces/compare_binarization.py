#!/usr/bin/env python3
"""
compare_binarization.py — compare the real contraction order
("binarization"/parenthesization) SeQuant chose for the same catalog
equation across several saved raw-trace occurrences.

Reuses `extract_trace_equations.py`'s `parse_raw_term()` (the proven
binary-tree parser over the raw `Term | Begin` grammar) — no new
parsing logic. For each occurrence this prints a molecule/instance-
independent "skeleton": the tree's real left/right shape, with every
leaf reduced to `label(sorted-class-letters)` (SeQuant instance numbers
and CSV-restriction pairarg specifics stripped) — so two structurally
identical trees produce byte-identical skeleton strings regardless of
which real dummy variables were used, and two differently-ordered
trees visibly differ.

Usage:
    python3 traces/compare_binarization.py ~/scratch/eq30_*.trace
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from extract_trace_equations import (  # noqa: E402
    TNode,
    _alias_greek,
    parse_raw_term,
)
from term_begin_mapper import _SYM_RE, _TERM_BEGIN_RE, _split_coef  # noqa: E402


def load_term_begin_text(path: Path) -> tuple[int, str]:
    """Read a saved `.trace` file's own `Term | Begin` line, strip the
    coefficient and symmetry tags — the same preprocessing
    `extract_raw_terms()` does before calling `parse_raw_term()`."""
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        m = _TERM_BEGIN_RE.match(line)
        if m:
            raw = m.group(1).strip()
            coef, tree_text = _split_coef(raw)
            return coef, _SYM_RE.sub("", tree_text)
    raise ValueError(f"no 'Term | Begin' line found in {path}")


def leaf_skeleton(node: TNode) -> str:
    classes = sorted(_alias_greek(base) for base, _inst in node.leaf.full_ids())
    return f"{node.leaf.label}({','.join(classes)})"


def skeleton(node: TNode) -> str:
    if node.kind == "leaf":
        return leaf_skeleton(node)
    return f"({skeleton(node.left)}*{skeleton(node.right)})"


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("traces", nargs="+", type=Path)
    args = ap.parse_args(argv)

    by_skeleton: dict[str, list[str]] = defaultdict(list)
    for path in args.traces:
        coef, text = load_term_begin_text(path)
        root = parse_raw_term(text)
        skel = skeleton(root)
        by_skeleton[skel].append(path.stem)
        print(f"{path.stem:28s} coef={coef:+d}  {skel}")

    print()
    if len(by_skeleton) == 1:
        print(f"[compare_binarization] ALL {sum(len(v) for v in by_skeleton.values())} occurrences share the SAME binarization.")
    else:
        print(f"[compare_binarization] {len(by_skeleton)} DISTINCT binarizations found across {sum(len(v) for v in by_skeleton.values())} occurrences:")
        for i, (skel, names) in enumerate(by_skeleton.items(), 1):
            print(f"  group {i} ({len(names)} occurrences): {', '.join(names)}")
            print(f"    {skel}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
