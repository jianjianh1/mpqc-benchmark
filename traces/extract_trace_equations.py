#!/usr/bin/env python3
"""
extract_trace_equations.py — extract a self-contained, real-trace-derived
computation spec from an MPQC/SeQuant eval-trace log.

Produces a text file in the same spirit as `all_equations.txt` (an
`eqN: # coef, kind` header followed by staged `LHS * RHS -> RESULT`
lines), but built directly from each real `Term | Begin` line's
fully-expanded (zero-CSE) expression, rather than reconstructed from a
different run's symbolic ground truth. See
~/.claude/plans/fluffy-seeking-blanket.md for the full design writeup.

Why not just replay `steps.csv`'s `Eval | Product` rows directly: MPQC
caches intermediates (`cache_imeds_=true`), so some real terms have a
whole cache-hit sub-branch that never gets a fresh `Eval` line at all
(confirmed on eq64's iter1/term81: `steps.csv` shows 8 real ops, but
the raw `Term | Begin` text shows two full branches — `(g*C*C*t)` and
`(g*C*C)` — multiplied together; the second branch's `g*C*C` leaves
never appear as `Eval` rows for this term because an equivalent result
was already cached from elsewhere). A "replay" built from `steps.csv`
alone would have a dangling reference for that branch. The raw
`Term | Begin` text is, by construction, the fully expanded leaf-only
product with zero CSE — exactly the "no reuse" structure we want.

Usage:
    python3 traces/extract_trace_equations.py \
        traces/checksum-run/ethane-checksum-v2.log \
        --steps traces/checksum-run/ethane-checksum-v2.steps.csv \
        --header traces/checksum-run/ethane-checksum-v2.header.json \
        --equations traces/all_equations.txt \
        --out traces/checksum-run/ethane-checksum-v2.extracted_equations.txt
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "bin"))

from term_begin_mapper import (  # noqa: E402
    _SYM_RE,
    _split_coef,
    canonicalize_expr,
    iter_term_begins,
    parse_catalog_trees,
    parse_tree,
)
from _axis_map import AXIS_MAP  # noqa: E402
from _trace_common import INDEX_LABEL_ALIASES  # noqa: E402
from parse_trace import INDEX_PIECE_RE  # noqa: E402

# Greek trace glyphs -> short Latin class letters, matching all_equations.txt's
# own convention (occ=i, hole/PAO=m, RI/DF aux=k). Order matters: the 2-codepoint
# mu-tilde (mu + combining tilde) must fold before bare mu or the combining
# mark is left orphaned (same fix as term_begin_mapper._alias_greek).
_GREEK_TO_LATIN = [("μ̃", "m"), ("μ", "m"), ("Κ", "k"), ("K", "k")]


def _alias_greek(text: str) -> str:
    for src, dst in _GREEK_TO_LATIN:
        text = text.replace(src, dst)
    return text


# ---------------------------------------------------------------------------
# Raw `Term | Begin` tokenizer + non-lossy tree parser.
#
# Grammar (confirmed against real logs — see plan's Phase 0):
#   expr    := term ('*' term)*
#   term    := '(' expr ')' | leaf
#   leaf    := label '{' group (';' group)* '}' (':' symtag)?
#   group   := atom (',' atom)*
#   atom    := ident ('<' ident (',' ident)* '>')?
#   ident   := base '_' digits          e.g. i_1, μ̃_19601, a_3
#
# Grouping is always plain '(' ')'; leaf arguments always use '{' '}' — so,
# unlike term_begin_mapper's tokenizer (which works on already-canonicalized
# `label(...)` text), there is no ambiguity between the two delimiters.
# ---------------------------------------------------------------------------

_ATOM_RE = re.compile(r"^([^\d_<>]+(?:_[^\d_<>]+)*?)_(\d+)$")


class Atom:
    """One index identity: (base_label, instance). `pairargs` is the
    (possibly empty) list of Atoms embedded in a `<...>` restriction
    bracket attached to this atom — e.g. for `a_1<i_1,i_2>`, base='a',
    instance=1, pairargs=[Atom('i',1), Atom('i',2)]."""

    __slots__ = ("base", "inst", "pairargs")

    def __init__(self, base: str, inst: int, pairargs: list["Atom"] | None = None):
        self.base = base
        self.inst = inst
        self.pairargs = pairargs or []

    @property
    def id(self) -> tuple[str, int]:
        return (self.base, self.inst)

    def __repr__(self) -> str:
        return f"{self.base}_{self.inst}" + (
            "<" + ",".join(repr(p) for p in self.pairargs) + ">" if self.pairargs else ""
        )


def _parse_atom(tok: str) -> Atom:
    tok = tok.strip()
    pairargs: list[Atom] = []
    if "<" in tok:
        base_part, rest = tok.split("<", 1)
        inner = rest.rstrip(">")
        pairargs = [_parse_atom(t) for t in inner.split(",") if t.strip()]
        tok = base_part
    m = _ATOM_RE.match(tok)
    if not m:
        raise ValueError(f"cannot parse index atom {tok!r}")
    return Atom(m.group(1), int(m.group(2)), pairargs)


class Leaf:
    __slots__ = ("label", "atoms")

    def __init__(self, label: str, atoms: list[Atom]):
        self.label = label
        self.atoms = atoms  # flat, in original (group-flattened) order

    def full_ids(self) -> set[tuple[str, int]]:
        """This leaf's full identity set: every atom's id, plus every
        pairarg's id embedded anywhere in this leaf (deduplicated)."""
        out: set[tuple[str, int]] = set()
        for a in self.atoms:
            out.add(a.id)
            for p in a.pairargs:
                out.add(p.id)
        return out

    def plain_ids(self) -> set[tuple[str, int]]:
        """This leaf's genuine tensor-axis ids only (each atom's OWN id) —
        excludes ids that appear solely as a pairarg on some other atom in
        this same leaf (pure restriction-bracket metadata, not a real axis
        of this tensor)."""
        return {a.id for a in self.atoms}


class TNode:
    """A binary-tree node over the fully-expanded term. `kind` is
    'leaf' or 'mul'. Populated bottom-up by the algebra pass with:
      - `free_ids`: this node's full surviving index identity set,
        including restriction-bracket pairargs (e.g. an occ index that
        only ever appears embedded in some OTHER leaf's `a_1<i,j>`
        bracket still needs to be "known" here so it can render/merge
        correctly upstream).
      - `plain_ids`: the subset of `free_ids` that are genuine tensor
        axes (a leaf's own bra/ket tokens) as opposed to pure
        restriction-bracket metadata. Contraction eligibility is tested
        on `plain_ids` only — see `annotate_free_ids` for why: an occ
        index embedded ONLY as another atom's pairarg is never, by
        itself, a summed axis at that node (it's PNO/CSV-restriction
        bookkeeping), so two children merely *sharing* it must not
        trigger a contraction there. Using the (lossy) full `free_ids`
        for this test was the original bug — confirmed wrong on a real
        trace: `C{a_2<i_2>;μ̃}` × `t{a_2<i_2>;i_2}` incorrectly contracted
        `i_2` immediately (since it's present in both full sets), while
        the real MPQC trace keeps `i_2` free through this exact join
        and only contracts it one node later, against a THIRD leaf
        where `i_2` is a genuine plain axis."""

    __slots__ = (
        "kind", "leaf", "left", "right",
        "free_ids", "plain_ids", "contracted_ids", "seen_counts",
    )

    def __init__(self, kind: str, leaf: Leaf | None = None, left=None, right=None):
        self.kind = kind
        self.leaf = leaf
        self.left = left
        self.right = right
        self.free_ids: set[tuple[str, int]] = set()
        self.plain_ids: set[tuple[str, int]] = set()
        self.contracted_ids: set[tuple[str, int]] = set()
        # Running per-id count of how many of that id's GLOBAL plain
        # occurrences (across the whole term) have been merged into this
        # subtree so far — see annotate_free_ids for why this, not just
        # "is it orphaned", is needed to time the auto-sweep correctly.
        self.seen_counts: dict[tuple[str, int], int] = {}


def _split_atoms(group: str) -> list[str]:
    """Split a group on ',' at angle-bracket depth 0 only — a bare split(',')
    would also split inside a pairarg bracket like `a_1<i_1,i_2>`."""
    parts, depth, buf = [], 0, []
    for ch in group:
        if ch == "<":
            depth += 1
        elif ch == ">":
            depth -= 1
        if ch == "," and depth == 0:
            parts.append("".join(buf))
            buf = []
        else:
            buf.append(ch)
    parts.append("".join(buf))
    return parts


def _parse_leaf_text(label: str, body: str) -> Leaf:
    """body is the content between a leaf's `{` and `}` — `;`-separated
    groups, each `,`-separated atoms (comma-splitting must respect any
    `<...>` pairarg bracket, hence `_split_atoms` not a bare split).
    Flattened into one ordered atom list (group structure doesn't matter
    for our flat catalog-style output)."""
    atoms: list[Atom] = []
    for group in body.split(";"):
        for tok in _split_atoms(group):
            tok = tok.strip()
            if tok:
                atoms.append(_parse_atom(tok))
    return Leaf(label, atoms)


def _tokenize_raw_term(text: str) -> list[str]:
    """Tokens: '(', ')', '*', or a leaf chunk `label{...}` (symtag already
    stripped by the caller)."""
    tokens: list[str] = []
    i, n = 0, len(text)
    while i < n:
        c = text[i]
        if c.isspace():
            i += 1
        elif c in "()* ":
            if c in "()*":
                tokens.append(c)
            i += 1
        elif c == "{":
            # Shouldn't start a token here; malformed input.
            raise ValueError(f"unexpected '{{' at {i} in {text!r}")
        else:
            # Leaf label, then its `{...}` body.
            j = i
            while j < n and text[j] not in "(){}*" and not text[j].isspace():
                j += 1
            label = text[i:j]
            if j >= n or text[j] != "{":
                raise ValueError(f"expected '{{' after label {label!r} in {text!r}")
            depth = 0
            k = j
            while k < n:
                if text[k] == "{":
                    depth += 1
                elif text[k] == "}":
                    depth -= 1
                    if depth == 0:
                        k += 1
                        break
                k += 1
            tokens.append(label + text[j:k])
            i = k
    return tokens


def parse_raw_term(text: str) -> TNode:
    """Parse a symtag-stripped raw `Term | Begin` expression into a TNode tree."""
    tokens = _tokenize_raw_term(text)
    pos = [0]

    def parse_expr() -> TNode:
        node = parse_atom_term()
        while pos[0] < len(tokens) and tokens[pos[0]] == "*":
            pos[0] += 1
            rhs = parse_atom_term()
            node = TNode("mul", left=node, right=rhs)
        return node

    def parse_atom_term() -> TNode:
        tok = tokens[pos[0]]
        if tok == "(":
            pos[0] += 1
            inner = parse_expr()
            assert tokens[pos[0]] == ")", f"unbalanced paren in {text!r}"
            pos[0] += 1
            return inner
        pos[0] += 1
        m = re.match(r"^([A-Za-z][A-Za-z0-9_]*)\{(.*)\}$", tok, re.DOTALL)
        if not m:
            raise ValueError(f"bad leaf token {tok!r}")
        leaf = _parse_leaf_text(m.group(1), m.group(2))
        return TNode("leaf", leaf=leaf)

    root = parse_expr()
    assert pos[0] == len(tokens), f"trailing tokens in {text!r}: {tokens[pos[0]:]}"
    return root


# ---------------------------------------------------------------------------
# Index algebra: contracted-vs-free, anchored on R's real free-index set so
# Hadamard/batch (ToT outer-pair) indices aren't mistaken for contractions
# just because they co-occur in both children (see plan's writeup).
# ---------------------------------------------------------------------------

_R_TOKEN_RE = re.compile(r"^(?P<label>\w+)\(([^)]*)\)$")


def parse_compact_atom(tok: str) -> tuple[Atom, list[Atom]]:
    """Parse an Eval-log *compact*-notation index token, e.g. `a_2i_1i_2`
    (base=a, inst=2, pairargs=[i_1,i_2]) or a plain `i_1`. Returns
    (own_atom, pairarg_atoms) — mirrors parse_trace.py's parse_index_token
    but keeps Atom objects instead of raw strings."""
    pieces = INDEX_PIECE_RE.findall(tok)
    if not pieces:
        raise ValueError(f"cannot parse compact atom {tok!r}")
    base, inst = pieces[0][0], int(pieces[0][1])
    pairargs = [Atom(p[0], int(p[1])) for p in pieces[1:]]
    return Atom(base, inst, pairargs), pairargs


def r_free_ids_from_expr(r_expr: str) -> set[tuple[str, int]]:
    """R's real free-index identities, from an Eval-log target string like
    `R(i_2,i_1;a_2i_1i_2,a_1i_1i_2)`."""
    m = _R_TOKEN_RE.match(r_expr.strip())
    if not m:
        raise ValueError(f"cannot parse R expr {r_expr!r}")
    ids: set[tuple[str, int]] = set()
    for tok in re.split(r"[,;]", m.group(2)):
        tok = tok.strip()
        if not tok:
            continue
        own, pairargs = parse_compact_atom(tok)
        ids.add(own.id)
        for p in pairargs:
            ids.add(p.id)
    return ids


def build_restriction_map(node: TNode, out: dict[tuple[str, int], list[tuple[str, int]]]) -> None:
    """Walk every leaf once, recording each CSV/PNO-restricted atom's pair
    dependency. An atom's restriction is a leaf-level fact — it doesn't
    change as the atom propagates up the tree, so this is computed once,
    globally per term (BEFORE the algebra pass, which needs it), rather
    than re-derived per node."""
    if node.kind == "leaf":
        for a in node.leaf.atoms:
            if a.pairargs:
                out[a.id] = [p.id for p in a.pairargs]
        return
    build_restriction_map(node.left, out)
    build_restriction_map(node.right, out)


def build_plain_occurrence_counts(node: TNode, out: dict[tuple[str, int], int]) -> None:
    """Walk every leaf once, counting how many DISTINCT leaves each id
    appears in as a genuine plain token (own axis, not just embedded as
    someone else's pairarg). Global per term, needed by `annotate_free_ids`
    to tell apart two cases that both look like "an occ id restricting a
    virtual that just resolved": one where the occ id also has a second,
    independent plain occurrence elsewhere still to be paired (must stay
    free until then), and one where it doesn't (safe to sweep away
    immediately once its restricted virtual(s) resolve)."""
    if node.kind == "leaf":
        for id_ in node.leaf.plain_ids():
            out[id_] = out.get(id_, 0) + 1
        return
    build_plain_occurrence_counts(node.left, out)
    build_plain_occurrence_counts(node.right, out)


def annotate_free_ids(
    node: TNode,
    r_free: set[tuple[str, int]],
    restriction_map: dict[tuple[str, int], list[tuple[str, int]]],
    reverse_deps: dict[tuple[str, int], set[tuple[str, int]]],
    plain_occurrence_counts: dict[tuple[str, int], int],
) -> None:
    """Bottom-up: populate `free_ids`/`plain_ids`/`contracted_ids` on every
    node.

    Contraction eligibility is tested on `plain_ids` (genuine tensor axes),
    NOT the full id set — an id that's only ever a restriction-bracket
    pairarg on one side is pure PNO/CSV metadata there, not a real summed
    axis, so merely co-occurring with the same id elsewhere must not
    trigger a contraction (confirmed against a real trace: `C{a_2<i_2>;m}`
    × `t{a_2<i_2>;i_2}` must NOT contract `i_2` here — MPQC's own trace
    keeps it free through this join, contracting it one node later against
    a leaf where it's a genuine plain axis, `g`).

    A SECOND mechanism handles the reverse situation: once ALL virtuals
    that restrict some occ id have themselves resolved (been contracted)
    AND every one of that occ id's own GLOBAL plain occurrences (per
    `plain_occurrence_counts`, computed once for the whole term) has
    already been merged into this subtree (tracked incrementally via
    `seen_counts`, since a plain occurrence sitting in a sibling subtree
    not yet joined still needs its chance at the ordinary shared-plain
    pairing below), that occ id is swept away too — it would otherwise
    never leave the free set (nothing left to pair it against). Confirmed
    against a real trace (`a3<i2,i3>`/`a4<i2,i3>`-style rank-4 ToT term):
    `i2`/`i3` are plain in TWO leaves each (a `g`, and later `t`) — the
    ordinary mechanism pairs them fine the first time (`g`'s plain `i2`
    meeting the propagated-plain `i2` from an inner `C*t`), but once
    `t`'s OWN plain `i2,i3` merge in too and `a3`/`a4` (their virtuals)
    resolve, they have nowhere left to pair against and must be swept —
    checking `seen_counts[id] == plain_occurrence_counts[id]` (all
    occurrences already inside this subtree) is what tells that state
    apart from an earlier node where a sibling subtree (not yet merged)
    still holds one of `i2`'s plain occurrences in reserve."""
    if node.kind == "leaf":
        node.free_ids = node.leaf.full_ids()
        node.plain_ids = node.leaf.plain_ids()
        node.seen_counts = {id_: 1 for id_ in node.plain_ids}
        return
    annotate_free_ids(node.left, r_free, restriction_map, reverse_deps, plain_occurrence_counts)
    annotate_free_ids(node.right, r_free, restriction_map, reverse_deps, plain_occurrence_counts)
    shared_plain = node.left.plain_ids & node.right.plain_ids
    candidate = shared_plain - r_free

    union_full = node.left.free_ids | node.right.free_ids
    union_plain = node.left.plain_ids | node.right.plain_ids
    seen_counts: dict[tuple[str, int], int] = dict(node.left.seen_counts)
    for id_, cnt in node.right.seen_counts.items():
        seen_counts[id_] = seen_counts.get(id_, 0) + cnt

    tentative_survivors = union_full - candidate
    protected = {
        pairarg_id
        for atom_id in tentative_survivors
        for pairarg_id in restriction_map.get(atom_id, ())
        if pairarg_id in candidate
    }
    contracted = candidate - protected

    remaining = union_full - contracted
    auto_swept = {
        id_
        for id_ in remaining
        if id_ not in r_free
        and id_ in reverse_deps
        and reverse_deps[id_].isdisjoint(remaining)
        and seen_counts.get(id_, 0) >= plain_occurrence_counts.get(id_, 0)
    }
    contracted = contracted | auto_swept

    node.contracted_ids = contracted
    node.free_ids = union_full - contracted
    node.plain_ids = union_plain - contracted
    node.seen_counts = seen_counts


_CLASS_PRIORITY = {"i": 0, "m": 1, "a": 2, "k": 3}


class Renumberer:
    """First-seen-order id -> short catalog-style label (i1, i2, m1, a1, ...),
    one counter per Greek-aliased base class. A pure bijective relabeling —
    unlike term_begin_mapper.canonicalize_expr's lossy class-collapsing,
    every distinct id gets its own distinct short name."""

    def __init__(self):
        self._counters: dict[str, int] = defaultdict(int)
        self._names: dict[tuple[str, int], str] = {}

    def name(self, id_: tuple[str, int]) -> str:
        if id_ not in self._names:
            base, _inst = id_
            cls = _alias_greek(base)
            self._counters[cls] += 1
            self._names[id_] = f"{cls}{self._counters[cls]}"
        return self._names[id_]

    def visit_leaf_order(self, node: TNode) -> None:
        """Pre-assign names by walking the tree left-to-right so ids read in
        natural, stable first-seen order (independent of render-time sorting)."""
        if node.kind == "leaf":
            for a in node.leaf.atoms:
                self.name(a.id)
                for p in a.pairargs:
                    self.name(p.id)
            return
        self.visit_leaf_order(node.left)
        self.visit_leaf_order(node.right)


def _sorted_ids(ids: set[tuple[str, int]], renum: Renumberer) -> list[tuple[str, int]]:
    """Canonical per-tensor axis order: class priority (i,m,a,k), then
    instance number."""

    def sort_key(id_):
        name = renum.name(id_)
        cls = re.match(r"[a-zA-Z]+", name).group(0)
        return (_CLASS_PRIORITY.get(cls, 9), id_[1])

    return sorted(ids, key=sort_key)


def render_ids(
    ids: set[tuple[str, int]],
    renum: Renumberer,
    restriction_map: dict[tuple[str, int], list[tuple[str, int]]],
) -> str:
    """Render a tensor's axes in canonical order. A restricted virtual
    (e.g. `a1` restricted to occ pair `i1,i2`) only gets the `<...>`
    suffix if at least one of its restrictors ISN'T already present as
    its own bare axis in this same `ids` set — when it is (the normal
    case: `annotate_free_ids`'s auto-sweep guarantees an occ index never
    disappears from a node while a virtual that depends on it still
    survives, so the two always co-occur), the bracket adds no
    information a reader can't already recover by noting which `i`-axes
    are present, so it's dropped in favor of the plain label."""
    sorted_ids = _sorted_ids(ids, renum)
    bare = set(sorted_ids)
    parts = []
    for id_ in sorted_ids:
        label = renum.name(id_)
        restr = restriction_map.get(id_)
        if restr and any(r not in bare for r in restr):
            label += "<" + ",".join(renum.name(r) for r in restr) + ">"
        parts.append(label)
    return ",".join(parts)


def leaf_render(node: TNode, renum: Renumberer, restriction_map) -> str:
    return f"{node.leaf.label}({render_ids(node.leaf.full_ids(), renum, restriction_map)})"


def emit_stages(
    root: TNode,
    renum: Renumberer,
    restriction_map: dict[tuple[str, int], list[tuple[str, int]]],
) -> tuple[list[str], str]:
    """Post-order walk emitting one staged binary-contraction line per
    internal node, fresh sequential I0,I1,... labels reset per term.
    Returns (stage_lines, top_ref) — `top_ref` is the whole term's own
    result reference; the caller appends the final `<top> * <coef> ->
    R(...)` line itself (needs the coef applied), or `<top> -> R(...)`
    directly if the leaf-level tree already IS a single leaf (no `mul`
    node at all, e.g. a trivial one-leaf term).

    No explicit einsum spec is emitted here — the target library's
    binary-contraction-with-batch-index API doesn't take a generic
    einsum string, and doesn't need one: given a stage's plain operand
    and result index lists alone, a shared label absent from the result
    is a plain contraction and one present in the result is a batch
    axis — the consumer implements that trivial local rule itself
    (hand-verified against real data on both the eq64 two-branch case
    and the `a3<i2,i3>`/`a4<i2,i3>` rank-4 case)."""
    stages: list[str] = []
    counter = [0]

    def walk(node: TNode) -> str:
        if node.kind == "leaf":
            return leaf_render(node, renum, restriction_map)
        left_ref = walk(node.left)
        right_ref = walk(node.right)
        label = f"I{counter[0]}"
        counter[0] += 1
        result_ref = f"{label}({render_ids(node.free_ids, renum, restriction_map)})"
        stages.append(f"    {left_ref} * {right_ref} -> {result_ref}")
        return result_ref

    top_ref = walk(root)
    return stages, top_ref


# ---------------------------------------------------------------------------
# Raw term extraction from the log (parallels term_begin_mapper's own
# extract_term_begins(), but keeps the symtag-stripped RAW text instead of
# canonicalizing it away — canonicalize_expr() is still used, separately,
# for the optional catalog cross-reference annotation).
# ---------------------------------------------------------------------------


def extract_raw_terms(log_path: Path) -> dict[tuple[int, int], tuple[int, str]]:
    """{(iter, term_idx) -> (coef, raw_symtag_stripped_text)}."""
    out: dict[tuple[int, int], tuple[int, str]] = {}
    for current_iter, term_idx, raw in iter_term_begins(log_path):
        coef, tree_text = _split_coef(raw)
        tree_text = _SYM_RE.sub("", tree_text)
        out[(current_iter, term_idx)] = (coef, tree_text)
    return out


def get_r_expr_per_term(steps_csv_path: Path) -> dict[tuple[int, int], str]:
    """{(iter, term_idx) -> raw target_expr string of the term's final
    `-> R(...)` step}, read straight from parse_trace.py's steps.csv."""
    out: dict[tuple[int, int], str] = {}
    with steps_csv_path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            tgt = row.get("intermediate_label", "")
            if tgt.startswith("R("):
                it, ti = row.get("iter"), row.get("term_idx")
                if it and ti:
                    key = (int(it), int(ti))
                    if key in out and out[key] != tgt:
                        # A term's residual contribution split across
                        # multiple steps.csv rows targeting the same R(...)
                        # for the same (iter, term_idx) — e.g. a batched
                        # partial-K contraction emitting separate
                        # accumulate-into-R steps. Silently keeping only
                        # the last one would use whichever R-expression
                        # happened to be written last for this term's
                        # r_free_ids_from_expr computation downstream.
                        print(
                            f"WARNING: get_r_expr_per_term: (iter={it}, "
                            f"term_idx={ti}) has multiple distinct R(...) "
                            f"rows; keeping the last one seen "
                            f"({out[key]!r} -> {tgt!r})",
                            file=sys.stderr,
                        )
                    out[key] = tgt
    return out


def kind_of(r_free_ids: set[tuple[str, int]], renum: Renumberer) -> str:
    n_virt = sum(1 for id_ in r_free_ids if renum.name(id_).startswith("a"))
    return {1: "singles", 2: "doubles", 3: "triples"}.get(n_virt, f"{n_virt}-virt")


# ---------------------------------------------------------------------------
# Header block: real index-space dims + leaf -> COO/.tns file mapping.
# ---------------------------------------------------------------------------


def format_header(header_json: dict, log_name: str, n_terms: int) -> str:
    lines = [
        f"# Extracted directly from a real MPQC/SeQuant trace ({log_name}):",
        "# each block below is one REAL executed Term firing's full binary-",
        "# contraction sub-DAG (leaf -> ... -> residual contribution),",
        "# taken verbatim from the fully-expanded `Term | Begin` expression --",
        "# NOT reconstructed/symbolic, and NOT deduplicated: every real",
        f"# (iter, term_idx) occurrence gets its own block ({n_terms} total),",
        "# since measuring the cost of 'no reuse' needs the full real",
        "# repetition MPQC's own CSE cache normally elides.",
        "#",
        "# REAL INDEX-SPACE DIMENSIONS (this run's basis; from the trace's own",
        "# header — CSV/PNO-restricted 'a' slots are per-pair and NOT a fixed",
        "# dense dim; each restricted virtual's occ pair is always ALSO listed",
        "# as its own bare axis in the same tensor, so no separate annotation",
        "# is needed (see 'a<i,j>' in the legend below for the rare exception):",
    ]
    skip_spin = {"↑", "↓"}
    for label, info in sorted(header_json.get("index_spaces", {}).items()):
        if any(s in label for s in skip_spin):
            continue
        lines.append(f"#   {label} = {info['dim']}")
    lines.append("#")
    lines.append("# LEAF -> COO/.tns FILE MAPPING (same AXIS_MAP convention as")
    lines.append("# ta-bench/ctf-bench's loaders; see bin/_axis_map.py):")
    for fname, axes in sorted(AXIS_MAP.items()):
        lines.append(f"#   {fname:<28s} axes=[{', '.join(axes)}]")
    lines.append("#")
    lines.append("# Index-letter legend: i=occ, m=PAO/hole, a=OSV/PNO virtual")
    lines.append("# (rank-3 C/t) or PNO virtual restricted to an occ PAIR (rank-4")
    lines.append("# C/t — its restricting occ indices are always bare axes of the")
    lines.append("# same tensor, e.g. C(i1,i2,m1,a1); only annotated inline as")
    lines.append("# a1<i1,i2> in the rare case a restrictor ISN'T already bare there),")
    lines.append("# k=RI/DF auxiliary.")
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main driver.
# ---------------------------------------------------------------------------


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("log", type=Path)
    ap.add_argument("--steps", type=Path, required=True)
    ap.add_argument("--header", type=Path, required=True)
    ap.add_argument("--equations", type=Path, default=None, help="all_equations.txt, for the optional catalog=eqXX cross-reference")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--min-iter", type=int, default=None)
    ap.add_argument("--max-iter", type=int, default=None)
    ap.add_argument(
        "--audit",
        action="store_true",
        help="after writing --out, run audit_extracted_equations.py's no-reuse/"
        "no-redundancy checks against it and exit non-zero if any fail",
    )
    args = ap.parse_args(argv)

    header_json = json.loads(args.header.read_text(encoding="utf-8"))
    raw_terms = extract_raw_terms(args.log)
    r_exprs = get_r_expr_per_term(args.steps)

    catalog_rev: dict[str, list[str]] = defaultdict(list)
    if args.equations is not None:
        catalog, _stage_counts = parse_catalog_trees(args.equations)
        for eq_id, key in catalog.items():
            catalog_rev[key[1]].append(eq_id)

    out_blocks: list[str] = []
    n_written = 0
    n_skipped = 0
    for (it, ti), (coef, raw_text) in sorted(raw_terms.items()):
        if args.min_iter is not None and it < args.min_iter:
            continue
        if args.max_iter is not None and it > args.max_iter:
            continue
        r_expr = r_exprs.get((it, ti))
        if r_expr is None:
            n_skipped += 1
            continue
        try:
            r_free = r_free_ids_from_expr(r_expr)
            root = parse_raw_term(raw_text)
            restriction_map: dict[tuple[str, int], list[tuple[str, int]]] = {}
            build_restriction_map(root, restriction_map)
            reverse_deps: dict[tuple[str, int], set[tuple[str, int]]] = defaultdict(set)
            for virt_id, pairargs in restriction_map.items():
                for occ_id in pairargs:
                    reverse_deps[occ_id].add(virt_id)
            plain_occurrence_counts: dict[tuple[str, int], int] = {}
            build_plain_occurrence_counts(root, plain_occurrence_counts)
            annotate_free_ids(root, r_free, restriction_map, reverse_deps, plain_occurrence_counts)
        except Exception as e:  # noqa: BLE001 — surface, count, keep going
            print(f"WARNING: iter{it} term{ti}: {e}", file=sys.stderr)
            n_skipped += 1
            continue

        renum = Renumberer()
        renum.visit_leaf_order(root)
        for id_ in r_free:
            renum.name(id_)  # ensure R's own ids are named even if a leaf-order quirk missed one

        stages, top_ref = emit_stages(root, renum, restriction_map)
        r_render = render_ids(r_free, renum, restriction_map)
        stages.append(f"    {top_ref} * {coef} -> R({r_render})")

        kind = kind_of(r_free, renum)
        catalog_key = canonicalize_expr(raw_text)
        try:
            catalog_key = parse_tree(catalog_key).canon()
        except ValueError:
            pass
        matches = catalog_rev.get(catalog_key, [])
        cat_note = f", catalog={'/'.join(matches)}" if matches else ""

        block_name = f"iter{it}_term{ti}"
        header_line = f"{block_name}:  # coef {coef}, {kind}{cat_note}"
        out_blocks.append(header_line + "\n" + "\n".join(stages))
        n_written += 1

    if n_written == 0:
        print(
            f"ERROR: extracted 0 blocks ({n_skipped} skipped) from {args.log} — "
            "every term was skipped (no matching R(...) row, or a parse "
            "error); refusing to write a header-only output file that "
            "looks valid but describes nothing",
            file=sys.stderr,
        )
        return 1

    header_text = format_header(header_json, str(args.log), n_written)
    # header_text already ends with exactly one "\n" (format_header's last
    # line is "", joined in) — add one more so the first block gets the
    # same blank-line separation "\n\n".join() gives every later block.
    args.out.write_text(
        header_text + "\n" + "\n\n".join(out_blocks) + "\n", encoding="utf-8"
    )

    print(f"[extract_trace_equations] wrote {n_written} blocks ({n_skipped} skipped) -> {args.out}")

    if args.audit:
        import audit_extracted_equations

        return audit_extracted_equations.main([str(args.out), "--log", str(args.log)])
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
