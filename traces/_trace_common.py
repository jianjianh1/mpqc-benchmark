"""Shared helpers for the trace analyzers.

Previously duplicated verbatim across `parse_trace.py`, `size_dump.py`,
`summarize_leaves.py`, and `summarize_tensors.py` (see code-review F14).
Consolidating here so the alias table + canonicalization logic has a single
source of truth — when a new index space (e.g. a new Greek label) gets
introduced, only one edit is needed.
"""

from __future__ import annotations

import csv
from pathlib import Path

# The 5-alkane series (ethane..hexane) used by build_alkanes_timing.py,
# build_multirank_scaling.py, and check_timing_consistency.py. Previously
# each of the three defined its own copy of MOLS/N_C and an identical
# eq_timing.csv int/float-casting loop (see code-review finding on
# build_multirank_scaling.py:27) — single source of truth here instead.
MOLS = ["ethane", "propane", "butane", "pentane", "hexane"]
N_C = {"ethane": 2, "propane": 3, "butane": 4, "pentane": 5, "hexane": 6}

_EQ_TIMING_INT_FIELDS = ("stage_count", "total_samples")
_EQ_TIMING_FLOAT_FIELDS = (
    "matches_per_iter",
    "median_ms",
    "mean_ms",
    "p90_ms",
    "min_ms",
    "max_ms",
    "total_ms_all_iters",
    "frac_of_iter_pct",
)


def load_eq_timing_csv(path: Path) -> dict[str, dict]:
    """Load a `<mol>[-npN]-traced.eq_timing.csv` into {eq_id -> row dict},
    with the numeric columns cast from str to int/float."""
    out: dict[str, dict] = {}
    with path.open() as f:
        for r in csv.DictReader(f):
            for k in _EQ_TIMING_INT_FIELDS:
                r[k] = int(r[k] or 0)
            for k in _EQ_TIMING_FLOAT_FIELDS:
                r[k] = float(r[k] or 0)
            out[r["eq_id"]] = r
    return out

# Eval-line glyph aliases (used in `Term | Begin` and `Eval | …` log lines)
# → TeX form used in the SeQuant index-space dimension table the parser
# extracts from the run header.
INDEX_LABEL_ALIASES = {
    "Κ": "K",  # Greek capital kappa (U+039A) ↔ Latin K used in dim table
    "μ": "\\mu",  # Greek lowercase mu (U+03BC) ↔ \mu
    "μ̃": "\\tilde{\\mu}",  # mu + combining tilde (U+03BC U+0303) ↔ \tilde{\mu}
}


def canonical(label: str) -> str:
    """Map an Eval-line glyph label to the dim-table TeX form."""
    return INDEX_LABEL_ALIASES.get(label, label)


def canonical_sig(
    target_expr: str, index_labels: str, csv_pair_indices: str
) -> str:
    """Anonymize SeQuant instance numbers so multiple uses of the same
    underlying tensor collapse to a single signature.

    Examples:
        target_expr="g(μ̃_19601,μ̃_19602,Κ_1)"          → "g(μ̃,μ̃,K)"
        target_expr="C(i_2,μ̃_19602;a_2i_2)"            → "C(i,μ̃;a<i>)"
        target_expr="t(i_1,i_2;a_3i_1i_2,a_2i_1i_2)"    → "t(i,i;a<i,i>,a<i,i>)"
    """
    if "(" not in target_expr or ")" not in target_expr:
        return target_expr
    label = target_expr.split("(", 1)[0]
    bases = [canonical(b) for b in index_labels.split(",") if b]
    csv_groups = csv_pair_indices.split(";") if csv_pair_indices else []
    csv_groups += [""] * (len(bases) - len(csv_groups))
    parts = []
    for base, csv_grp in zip(bases, csv_groups):
        if csv_grp.strip():
            csv_bases = [canonical(c.split("_")[0]) for c in csv_grp.split(",") if c]
            parts.append(f"{base}<{','.join(csv_bases)}>")
        else:
            parts.append(base)
    return f"{label}({','.join(parts)})"


# Back-compat alias (size_dump.py originally used this name).
canonical_tensor_signature = canonical_sig
