#!/usr/bin/env python3
"""Attribute per-diagram wall time by matching each trace's Term|Begin
expression against catalog eqs (each reconstructed from its stages to
a leaf-only tree).

Bypasses the leaf-multiset ambiguity that trips up map_eqs_to_timing.py:
`Term | Begin` lines carry the fully-expanded SeQuant expression SeQuant
is about to dispatch to TA::einsum — no CSE elision. The catalog's
staged form encodes exactly the same tree, just serialized as a DAG.
Reconstruct the catalog's tree from its stages, canonicalize both,
match.

CLI:
    python3 traces/term_begin_mapper.py
        --equations traces/all_equations.txt
        --log <path to raw MPQC log>
        --steps <path to <mol>-traced.steps.csv>
        --output-csv <path>
"""
from __future__ import annotations

import argparse
import csv
import re
import statistics
import sys
from collections import Counter, defaultdict
from itertools import groupby
from pathlib import Path
from typing import Iterable


# --- normalization -----------------------------------------------------------

# Bra/ket symmetry annotation stripped from SPTN tensor literals; the catalog
# doesn't carry these so we drop them uniformly.
_SYM_RE = re.compile(r":N[-]?[SNC][-]?[SNC](?::[SNC])?")  # :N-S-S, :N-N-S, :N-C-S
_SPACES = re.compile(r"\s+")


def _alias_greek(text: str) -> str:
    """Fold Greek glyph classes to Latin: μ̃/μ → m, Κ/K → k.

    μ̃ is two codepoints (μ + U+0303) — must be folded BEFORE μ or the
    combining tilde is left orphaned. Latin K also folds to k so bare-K
    catalog labels match Greek-Κ trace labels (bookkeeping symmetry with
    the μ̃/μ pair)."""
    text = text.replace("μ̃", "m")
    text = text.replace("μ", "m")
    text = text.replace("Κ", "k")
    text = text.replace("K", "k")
    return text


def canonicalize_expr(expr: str) -> str:
    """Normalize a SPTN/catalog tensor expression to a comparable form.

    Trace side: `{μ̃_19602;a_2<i_2>}` bra;ket with restriction brackets.
    Catalog side: flat `(a3,i2,m2)`.

    Both get reduced to `label(class_tuple)` where class_tuple is the
    tensor's per-slot index-class letters. Restriction indices are
    unioned into the tensor's index set IFF they aren't already there
    (SPTN's `t{a_2<i_2>;i_2}` reuses `i_2` in both ket-restriction and
    bra, so it counts once → 2-slot `t(a,i)`, matching catalog).

    Commutative sorting of `*` children happens at tree level.
    """
    # Strip symmetry annotations first — they don't have parens/braces.
    text = _SYM_RE.sub("", expr)
    # Fold Greek → Latin so μ̃/Κ match m/k class letters used by the catalog.
    text = _alias_greek(text)
    # Rewrite each tensor literal (SPTN or catalog) into `label(sorted_classes)`.
    text = _rewrite_tensors(text)
    text = _SPACES.sub(" ", text).strip()
    return text


# Match either SPTN `label{...}` or catalog `label(...)` at top level.
_TENSOR_LITERAL_RE = re.compile(r"([A-Za-z][A-Za-z0-9]*)\s*([\{\(])")


def _rewrite_tensors(text: str) -> str:
    """Walk `text`, replacing every tensor literal with `label(cls,cls,...)`.

    Preserves everything else (parens, `*`) so the resulting string is a
    valid mini-expression consumable by `parse_tree`.
    """
    out: list[str] = []
    i = 0
    while i < len(text):
        # Try to match a tensor literal at position i.
        m = _TENSOR_LITERAL_RE.match(text, i)
        if not m:
            out.append(text[i])
            i += 1
            continue
        label = m.group(1)
        open_ch = m.group(2)
        close_ch = "}" if open_ch == "{" else ")"
        # Find matching close, allowing nested (for restrictions) `<...>`.
        depth = 1
        j = m.end()
        while j < len(text) and depth > 0:
            c = text[j]
            if c == open_ch:
                depth += 1
            elif c == close_ch:
                depth -= 1
            j += 1
        body = text[m.end() : j - 1]
        # Bail out if this "tensor literal" is really a grouping paren followed
        # by something non-tensor (e.g. `(a * b)` where `a` is an already-
        # rewritten leaf). Heuristic: tensor literals never contain `*`.
        if "*" in body:
            # Not a tensor — emit the label, then recurse into the parenthesized
            # subexpression.
            out.append(label)
            out.append(open_ch if open_ch == "(" else "(")
            out.append(_rewrite_tensors(body))
            out.append(")")
            i = j
            continue
        cls_list = _slot_classes(body)
        out.append(f"{label}({','.join(cls_list)})")
        i = j
    return "".join(out)


def _slot_classes(body: str) -> list[str]:
    """Extract per-slot index class letters from a tensor body, sorted.

    Body is either SPTN (`bra1,bra2;ket1<restr>,ket2`) or catalog
    (`a3,i2,m2` — flat comma-separated). Each unique variable name
    contributes exactly one slot — so a variable that appears in both
    a restriction and elsewhere in the same tensor counts once.
    Class letter is derived per variable (`i_2` → `i`, `a2` → `a`).

    The returned list is SORTED alphabetically. Rationale: the catalog
    writes the same tensor with varying slot orders across eqs
    (`C(a3,i1,m1)` in some, `C(m1,a1,i1)` in others), while SeQuant's
    SPTN output picks one canonical bra;ket order. Without sorting,
    trace-side `C(m,a,i)` and catalog-side `C(a,i,m)` become different
    leaf tokens even though they refer to the same tensor — matching
    fails and ~20% of catalog eqs get zero samples (code-review finding
    #2, verified 2026-07-03).
    """
    # Flatten body: `<...>` restriction brackets become just extra commas
    # so restriction indices join the flat list. Dedup happens by variable
    # name below.
    flat = body.replace("<", ",").replace(">", "")
    # Then join `;` and `,` as a single delimiter.
    tokens = re.split(r"[,;]", flat)
    seen: set[str] = set()
    classes: list[str] = []
    for tok in tokens:
        tok = tok.strip()
        if not tok:
            continue
        var = _var_of(tok)
        if var in seen:
            continue
        seen.add(var)
        classes.append(_class_of(tok))
    classes.sort()
    return classes


def _var_of(token: str) -> str:
    """`i_2` → `i_2` (SPTN keeps the instance number); `a2` → `a2` (catalog).
    Returns the raw variable name for set-membership comparison."""
    return token.strip()


def _class_of(token: str) -> str:
    """`i_2` → `i`, `a2` → `a`, `μ̃_19602` → `m` (Greek pre-folded)."""
    tok = token.strip()
    m = re.match(r"([a-zA-Z])", tok)
    return m.group(1) if m else tok


# --- tree parsing (both trace and catalog land here after canonicalization) --


class Node:
    __slots__ = ("kind", "value", "children")

    def __init__(self, kind: str, value: str = "", children=None):
        self.kind = kind      # "leaf" or "mul"
        self.value = value    # for leaf: `C(a,i,m)` etc.
        self.children = children or []

    def canon(self) -> str:
        """Canonical string: commutative + associative flattening."""
        if self.kind == "leaf":
            return self.value
        # Flatten nested `*` chains.
        parts = []
        stack = [self]
        while stack:
            n = stack.pop()
            if n.kind == "mul":
                stack.extend(n.children)
            else:
                parts.append(n.canon())
        return "*".join(sorted(parts))


def parse_tree(text: str) -> Node:
    """Parse a canonicalized expression (leaves + `*` and parens) into a tree.

    Assumes the canonicalization has already unified delimiters and stripped
    the noisy bits. Grammar:
        expr    := term ('*' term)*
        term    := '(' expr ')' | leaf
        leaf    := label '(' index-list ')'    (parens must match)
    """
    tokens = _tokenize(text)
    pos = [0]

    def parse_expr() -> Node:
        children = [parse_term()]
        while pos[0] < len(tokens) and tokens[pos[0]] == "*":
            pos[0] += 1
            children.append(parse_term())
        if len(children) == 1:
            return children[0]
        return Node("mul", children=children)

    def parse_term() -> Node:
        if pos[0] >= len(tokens):
            raise ValueError(f"unexpected end of expression at {text!r}")
        tok = tokens[pos[0]]
        if tok == "(":
            pos[0] += 1
            inner = parse_expr()
            if pos[0] >= len(tokens) or tokens[pos[0]] != ")":
                raise ValueError(f"unbalanced ( in {text!r}")
            pos[0] += 1
            return inner
        # leaf token
        pos[0] += 1
        return Node("leaf", value=tok)

    root = parse_expr()
    if pos[0] != len(tokens):
        raise ValueError(f"trailing tokens in {text!r}: {tokens[pos[0]:]!r}")
    return root


def _tokenize(text: str) -> list[str]:
    """Tokenize into `(`, `)`, `*`, and leaf-strings like `C(a,i,m)`."""
    text = text.strip()
    tokens: list[str] = []
    i = 0
    while i < len(text):
        c = text[i]
        if c.isspace():
            i += 1
        elif c == "*":
            tokens.append("*")
            i += 1
        elif c == "(":
            # Distinguish "(" as a grouping paren from "(" inside a leaf.
            # A leaf-open follows an identifier char (label). A grouping
            # paren follows nothing, or '*', or another '('.
            prev = tokens[-1] if tokens else ""
            if tokens and _is_leaf_label(prev):
                # this "(" continues the leaf: read until matching ")"
                buf = tokens.pop()
                depth = 0
                while i < len(text):
                    ch = text[i]
                    buf += ch
                    if ch == "(":
                        depth += 1
                    elif ch == ")":
                        depth -= 1
                        if depth == 0:
                            i += 1
                            break
                    i += 1
                tokens.append(buf)
            else:
                tokens.append("(")
                i += 1
        elif c == ")":
            tokens.append(")")
            i += 1
        else:
            # Leaf label (identifier + optional following `(...)` next iter).
            j = i
            while j < len(text) and (text[j].isalnum() or text[j] in "_"):
                j += 1
            tokens.append(text[i:j])
            i = j
    return tokens


def _is_leaf_label(tok: str) -> bool:
    """Is `tok` a bare label like `g`, `C`, `t`, `I0`, `R` (before `(...)`)?"""
    return bool(tok) and tok[0].isalpha() and "(" not in tok


# --- Term|Begin extraction ---------------------------------------------------

_TERM_BEGIN_RE = re.compile(r"\s*Term \| Begin \| (.+?)$")
_ITER_HDR_RE = re.compile(r"^\s*iter\s+delta\s+residual\s+energy\s+total time/s\s*$")
_ITER_ROW_RE = re.compile(r"^\s*(\d+)\s+[\d.e+-]+\s+[\d.e+-]+\s+[-\d.]+\s+[\d.]+\s*$")


def iter_term_begins(log_path: Path) -> Iterable[tuple[int, int, str]]:
    """Walk a trace log, yielding (iter, term_idx, raw_term_text) for every
    `Term | Begin` line. Shared by extract_term_begins (below) and
    extract_trace_equations.py's extract_raw_terms — both need the exact
    same iter/term_idx bookkeeping over the same lines, and previously
    each had its own copy of this walk (including the hardcoded 86
    rollover count), so a bookkeeping fix could land in one and not the
    other. iter starts at 1 (matches steps.csv); term_idx starts at 1 per
    iter, increments on each `Term | Begin`.
    """
    current_iter = 0
    term_idx = 0
    with log_path.open(encoding="utf-8", errors="replace") as f:
        for line in f:
            # Detect start of a CCSD iteration by the `iter delta ...` header,
            # then the next `iter=k row` bumps us. Simpler: bump on each `Term`
            # after seeing the header; increment `current_iter` on each new
            # iter-body row (the CCk driver logs `iter delta ...` once, then
            # iter rows like `    1    3.3255e+00    ...`).
            m_hdr = _ITER_HDR_RE.match(line)
            if m_hdr:
                # Sequant emits ONE "iter delta" header at the start of the
                # CCk solver, then per-iter rows. But it also emits per-iter
                # cycle blocks — the Term|Begin lines appear BEFORE each iter
                # row is printed. To sync: bump `current_iter` right when we
                # see the FIRST Term|Begin after the header (or after the
                # prior iter's Term|End sequence).
                pass
            m_it = _ITER_ROW_RE.match(line)
            if m_it:
                # An iter row (`    N    delta   residual   energy   time`).
                # This is printed AFTER all Term|Begin/End for iter N. So it
                # doesn't itself trigger a new iter; the NEXT Term|Begin does.
                pass
            m_tb = _TERM_BEGIN_RE.match(line)
            if m_tb:
                # New Term|Begin. If we've already seen 86 terms in the
                # current iter, roll over.
                if term_idx == 0 or term_idx >= 86:
                    current_iter += 1
                    term_idx = 0
                term_idx += 1
                yield current_iter, term_idx, m_tb.group(1).strip()


def extract_term_begins(log_path: Path) -> dict[tuple[int, int], tuple[int, str]]:
    """{(iter, term_idx) -> (coef, canonicalized_tree_string)}."""
    out: dict[tuple[int, int], tuple[int, str]] = {}
    for current_iter, term_idx, raw in iter_term_begins(log_path):
        coef, tree_text = _split_coef(raw)
        canon = canonicalize_expr(tree_text)
        try:
            tree = parse_tree(canon)
            key = tree.canon()
        except ValueError:
            key = canon  # fall back to raw canonical string
        out[(current_iter, term_idx)] = (coef, key)
    return out


_COEF_RE = re.compile(r"^\s*(-?\d+)\s+(.+)$")


def _split_coef(text: str) -> tuple[int, str]:
    """Split leading `-1 (...)`, `2 (...)`, etc. off the tree body.

    SeQuant emits `<coef> <tree>` for each term. Coefficient may be
    omitted (implicit `1`) — handle both.
    """
    m = _COEF_RE.match(text)
    if m:
        return int(m.group(1)), m.group(2).strip()
    return 1, text.strip()


# --- Catalog side: parse + reconstruct trees --------------------------------


def parse_catalog_trees(
    equations_path: Path,
) -> tuple[dict[str, tuple[int, str]], dict[str, int]]:
    """({eq_id -> (coef, canonical_tree_key)}, {eq_id -> stage_count}).

    Reconstructs each eq's leaf-only tree by walking backward from R and
    inlining intermediates. Also records the stage_count so downstream
    per-eq tables can surface it (helps readers gauge diagram complexity)."""
    out: dict[str, tuple[int, str]] = {}
    stage_counts: dict[str, int] = {}
    current_id: str | None = None
    current_coef = 1
    stages: list[tuple[str, str, str]] = []
    with equations_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            m = re.match(r"^(eq\d+):\s*(?:#\s*coef\s*(-?\d+))?", line)
            if m:
                if current_id is not None:
                    out[current_id] = _finalize_catalog_eq(current_coef, stages)
                    stage_counts[current_id] = len(stages)
                current_id = m.group(1)
                current_coef = int(m.group(2)) if m.group(2) else 1
                stages = []
                continue
            m = re.match(r"^\s*(.+?)\s*\*\s*(.+?)\s*->\s*(.+?)\s*$", line)
            if m:
                stages.append((m.group(1), m.group(2), m.group(3)))
        if current_id is not None:
            out[current_id] = _finalize_catalog_eq(current_coef, stages)
            stage_counts[current_id] = len(stages)
    return out, stage_counts


def _finalize_catalog_eq(coef: int, stages: list[tuple[str, str, str]]) -> tuple[int, str]:
    """Walk backward from R, inline intermediates, produce canonical tree key."""
    if not stages:
        return (coef, "")
    # Build defs: intermediate_label -> (lhs, rhs) using LABEL (before parens).
    defs: dict[str, tuple[str, str]] = {}
    r_stage: tuple[str, str, str] | None = None
    for lhs, rhs, result in stages:
        label = _label_of(result)
        if label == "R":
            r_stage = (lhs, rhs, result)
        else:
            defs[_label_of(result)] = (lhs, rhs)
    if r_stage is None:
        return (coef, "")
    tree_text = _expand_operand(r_stage[0], defs) + " * " + _expand_operand(r_stage[1], defs)
    canon = canonicalize_expr(tree_text)
    try:
        tree = parse_tree(canon)
        return (coef, tree.canon())
    except ValueError:
        return (coef, canon)


def _label_of(tensor_expr: str) -> str:
    """`I0(i1,m1)` → `I0`. `C(a,i,m)` → `C`. `R(i1,a1)` → `R`."""
    m = re.match(r"^\s*([A-Za-z][A-Za-z0-9]*)", tensor_expr)
    return m.group(1) if m else tensor_expr


def _expand_operand(op: str, defs: dict[str, tuple[str, str]]) -> str:
    """Recursively expand `op` (a tensor expression string) — if it names an
    intermediate in `defs`, inline `(lhs * rhs)`; otherwise return leaf."""
    label = _label_of(op)
    if label in defs:
        l, r = defs[label]
        return "(" + _expand_operand(l, defs) + " * " + _expand_operand(r, defs) + ")"
    return op


# --- Attribution -------------------------------------------------------------


def per_term_wall(
    steps_csv: Path,
    include_disk_loads: bool = False,
    min_iter: int | None = None,
    max_iter: int | None = None,
) -> dict[tuple[int, int], int]:
    """Sum time_ns per (iter, term_idx) from the existing steps.csv.

    Excludes each leaf tensor's one-time disk-load cost by default: the
    first time a given (label, index_labels, csv_pair_indices) slice is
    touched anywhere in the file is a cold materialization; every later
    touch of that same slice is served from MPQC's in-process cache and
    is counted normally, same as every non-Tensor step (the actual
    computation). Pass include_disk_loads=True to sum every row
    unconditionally instead (the old behavior).

    min_iter/max_iter restrict which iterations are SUMMED into the
    output, but the disk-load first-touch tracker still scans every row
    in file order regardless — a slice already touched before min_iter
    must not be re-counted as a fresh disk-load once the window opens."""
    out: dict[tuple[int, int], int] = defaultdict(int)
    seen_leaf_slices: set[tuple[str, str, str]] = set()
    with steps_csv.open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            try:
                it = int(r["iter"])
                tid = int(r["term_idx"])
                ns = int(r["time_ns"] or 0)
            except (KeyError, ValueError):
                continue
            if not include_disk_loads and r.get("step_kind") == "Tensor":
                key = (r.get("label", ""), r.get("index_labels", ""), r.get("csv_pair_indices", ""))
                if key not in seen_leaf_slices:
                    seen_leaf_slices.add(key)
                    continue  # disk first-touch — excluded
            if min_iter is not None and it < min_iter:
                continue
            if max_iter is not None and it > max_iter:
                continue
            out[(it, tid)] += ns
    return out


def _pct(a: float, b: float) -> float:
    return 100.0 * a / b if b else 0.0


def emit_csv(
    out_path: Path,
    per_eq_samples: dict[str, list[int]],
    catalog_ids: list[str],
    total_ns_all: int,
    stage_counts: dict[str, int] | None = None,
    n_iters: int | None = None,
) -> None:
    cols = [
        "eq_id",
        "stage_count",
        "matches_per_iter",
        "total_samples",
        "median_ms",
        "mean_ms",
        "p90_ms",
        "min_ms",
        "max_ms",
        "total_ms_all_iters",
        "frac_of_iter_pct",
    ]
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(cols)
        # matches_per_iter denominator: the ACTUAL iteration count from the
        # trace, not `max(len(samples))` — the latter used to be picked from
        # whichever eq accumulated the most samples (~12 for eqs firing
        # multiple times/iter), which silently rescaled every other eq's
        # ratio and reported eq0=0.25 across 3 iters instead of 1.0
        # (code-review finding #1, verified 2026-07-03).
        denom = max(n_iters or 1, 1)
        for eq_id in catalog_ids:
            sc = (stage_counts or {}).get(eq_id, 0)
            samples = per_eq_samples.get(eq_id, [])
            if not samples:
                w.writerow([eq_id, sc, 0.0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
                continue
            times_ms = [ns / 1e6 for ns in samples]
            p90 = (
                statistics.quantiles(times_ms, n=10)[8]
                if len(times_ms) >= 10
                else max(times_ms)
            )
            w.writerow(
                [
                    eq_id,
                    sc,
                    round(len(samples) / denom, 3),
                    len(samples),
                    round(statistics.median(times_ms), 4),
                    round(statistics.mean(times_ms), 4),
                    round(p90, 4),
                    round(min(times_ms), 4),
                    round(max(times_ms), 4),
                    round(sum(times_ms), 4),
                    round(_pct(sum(ns for ns in samples), total_ns_all), 3),
                ]
            )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--equations", type=Path, required=True)
    ap.add_argument("--log", type=Path, required=True)
    ap.add_argument("--steps", type=Path, required=True)
    ap.add_argument("--output-csv", type=Path, required=True)
    ap.add_argument("--dump-unmatched", action="store_true")
    ap.add_argument(
        "--include-disk-loads",
        action="store_true",
        help="Sum every step's time_ns unconditionally, including each leaf "
        "tensor's one-time disk-load/materialization cost. Default excludes "
        "just the first touch of each distinct leaf slice, keeping every "
        "cache-hit re-touch and the full computation (contraction, permute, "
        "accumulate).",
    )
    ap.add_argument(
        "--min-iter",
        type=int,
        default=None,
        help="Only sum (iter, term_idx) groups with iter >= this value. "
        "Default: unrestricted (every iteration).",
    )
    ap.add_argument(
        "--max-iter",
        type=int,
        default=None,
        help="Only sum (iter, term_idx) groups with iter <= this value. "
        "Default: unrestricted (every iteration). Pass --min-iter 1 "
        "--max-iter 1 to isolate iteration 1 — the one iteration that "
        "executes every diagram's complete computation graph with nothing "
        "yet cached across CCSD iterations.",
    )
    args = ap.parse_args()

    print(f"catalog: {args.equations.name}", file=sys.stderr)
    catalog, stage_counts = parse_catalog_trees(args.equations)
    # Reverse map: canonical_tree -> [eq_ids].
    by_key: dict[tuple[int, str], list[str]] = defaultdict(list)
    for eq_id, (coef, key) in catalog.items():
        by_key[(coef, key)].append(eq_id)
    print(
        f"  {len(catalog)} eqs → {len(by_key)} unique (coef, tree) keys",
        file=sys.stderr,
    )
    dup_keys = [(k, eqs) for k, eqs in by_key.items() if len(eqs) > 1]
    if dup_keys:
        print(f"  {len(dup_keys)} keys shared by multiple eqs:", file=sys.stderr)
        for k, eqs in dup_keys[:5]:
            print(
                f"    {sorted(eqs, key=lambda x: int(x[2:]) if x[2:].isdigit() else 999)}",
                file=sys.stderr,
            )

    print(f"log: {args.log.name}", file=sys.stderr)
    term_begins = extract_term_begins(args.log)
    n_iters_log = max(it for it, _ in term_begins) if term_begins else 0
    lo = args.min_iter if args.min_iter is not None else 1
    hi = args.max_iter if args.max_iter is not None else n_iters_log
    n_iters = max(hi - lo + 1, 0) if n_iters_log else 0
    print(
        f"  {len(term_begins)} Term|Begin lines across {n_iters_log} iters"
        + (f" (restricted to iters {lo}-{hi}, {n_iters} iters)" if (args.min_iter, args.max_iter) != (None, None) else ""),
        file=sys.stderr,
    )

    print(f"steps: {args.steps.name}", file=sys.stderr)
    walls = per_term_wall(
        args.steps,
        include_disk_loads=args.include_disk_loads,
        min_iter=args.min_iter,
        max_iter=args.max_iter,
    )

    # Attribute
    per_eq_samples: dict[str, list[int]] = defaultdict(list)
    unmatched: list[tuple[int, int, int, str]] = []
    matched_ns_all = 0
    total_ns_all = sum(walls.values())
    for (it, tid), (coef, tree_key) in term_begins.items():
        if it < lo or it > hi:
            continue
        ns = walls.get((it, tid), 0)
        eqs = by_key.get((coef, tree_key), [])
        if eqs:
            eq_id = sorted(eqs, key=lambda x: int(x[2:]) if x[2:].isdigit() else 999)[0]
            per_eq_samples[eq_id].append(ns)
            matched_ns_all += ns
        else:
            unmatched.append((it, tid, ns, tree_key))

    n_matched = sum(len(v) for v in per_eq_samples.values())
    total_groups = n_matched + len(unmatched)
    print(
        f"attribution: {n_matched}/{total_groups} matched "
        f"({_pct(n_matched, total_groups):.1f}%)  "
        f"wall {matched_ns_all/1e9:.2f}s / {total_ns_all/1e9:.2f}s "
        f"({_pct(matched_ns_all, total_ns_all):.2f}%)",
        file=sys.stderr,
    )

    if args.dump_unmatched:
        print("\n=== unmatched (first 10) ===", file=sys.stderr)
        for it, tid, ns, key in unmatched[:10]:
            print(f"  iter={it} tid={tid} ms={ns/1e6:.1f}", file=sys.stderr)
            print(f"    key: {key[:200]}", file=sys.stderr)

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    catalog_ids = sorted(
        catalog.keys(), key=lambda x: int(x[2:]) if x[2:].isdigit() else 999
    )
    emit_csv(
        args.output_csv,
        per_eq_samples,
        catalog_ids,
        total_ns_all,
        stage_counts,
        n_iters=n_iters,
    )
    print(f"wrote {args.output_csv}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
