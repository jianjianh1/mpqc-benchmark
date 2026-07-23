#!/usr/bin/env python3
"""
Parse MPQC SeQuant trace logs into structured tables.

For each input log, emits:
    <stem>.steps.csv     — one row per `Eval | *` step
    <stem>.header.json   — header context (basis, index-space dims, summary stats, embedded input)

Layer 0 of the layered extraction plan documented in
~/.claude/plans/glowing-rolling-hollerith.md. No MPQC code is touched.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Iterable

EVAL_KINDS = ("Tensor", "Product", "Permute", "Constant", "SumInplace")

# Identify a single SeQuant index token like "i_2", "μ̃_19602", "Κ_1",
# "a_2i_2" (CSV-restricted), "a_3i_1i_2". The token is a *sequence* of
# (label, instance_number) pairs concatenated; the FIRST pair is the index
# itself, any following pairs are CSV/PNO pair restrictions.
INDEX_PIECE_RE = re.compile(r"([^_,;()\s]+?)_(\d+)")

# An Eval line — kind, ns, body (columns), trailing expression.
EVAL_LINE_RE = re.compile(
    r"^\s*Eval\s*\|\s*(?P<kind>\w+)\s*\|\s*(?P<ns>\d+)ns\s*\|\s*(?P<body>.*?)\|\s*(?P<expr>[^|]*)$"
)

# A CCSD iteration summary row, e.g. "    1    3.32552e+00    5.87722e-06   -3.325522275261         110.9".
CC_ITER_ROW_RE = re.compile(
    r"^\s*(?P<iter>\d+)\s+[+\-\deE.]+\s+[+\-\deE.]+\s+[+\-\deE.]+\s+[+\-\deE.]+\s*$"
)
CC_ITER_HEADER_RE = re.compile(r"^\s*iter\s+delta\s+residual\s+energy\s+total time/s\s*$")

# Header sections
INPUT_KEYVAL_START_RE = re.compile(r"^\s*Input KeyVal \(format=JSON\):\s*$")
BASIS_BLOCK_RE = re.compile(r"^\s*(\w+)\s*Basis\s*=\s*(.*)$")
TILES_RE = re.compile(r"tiles\s*=\s*\[\s*(\d+)\s*,\s*(\d+)\s*\)\s*,\s*elements\s*=\s*\[\s*(\d+)\s*,\s*(\d+)\s*\)")
TILE_SIZE_RE = re.compile(r"\{min,max,mean\}\s*tile size\s*=\s*\{\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\}")
INDEX_SPACE_RE = re.compile(r"^\s*\{(?P<label>.+)_\{(?P<inst>\d+)\}\}\s*:\s*(?P<dim>\d+)\s*$")

# Eval-line labels arrive as Unicode glyphs; the dim-table uses TeX forms.
# Normalize the Eval-line form to the TeX form via the shared alias table.
from _trace_common import INDEX_LABEL_ALIASES  # noqa: E402
GENERATING_RE = re.compile(r"^\s*Generating CSV-CCk equations\.\.\.")

# PNO/OSV/PAO summary stats
PAO_AVG_RE = re.compile(r"Average number of PAOs per pair:\s*([0-9.]+)")
PNO_AVG_RE = re.compile(r"Average number of PNOs per pair:\s*([0-9.]+)")
OSV_AVG_RE = re.compile(r"Average\s+OSVs\s+per\s+pair:\s*([0-9.]+)")
OSV_OFFDIAG_RE = re.compile(r"Average\s+OSVs\s+per\s+off\s+diag\s+pair:\s*([0-9.]+)")
CSV_PAIRS_RE = re.compile(r"Number of \{all,unique\} CSV pairs:\s*\{\s*(\d+)\s*,\s*(\d+)\s*\}")
MP2_PAIRS_RE = re.compile(r"Number of \{all,unique\} MP2 pairs:\s*\{\s*(\d+)\s*,\s*(\d+)\s*\}")


@dataclass
class HeaderContext:
    log_path: str
    mpqc_version: str = ""
    mpqc_revision: str = ""
    machine: str = ""
    start_time: str = ""
    n_mpi: int = 0
    n_threads: int = 0
    input_json: dict = field(default_factory=dict)
    bases: dict = field(default_factory=dict)
    index_spaces: dict = field(default_factory=dict)
    summary_stats: dict = field(default_factory=dict)


def parse_header(lines: list[str]) -> HeaderContext:
    """Scan the top of the log for setup context."""
    ctx = HeaderContext(log_path="")

    # Banner
    for i, line in enumerate(lines[:20]):
        s = line.strip()
        m = re.match(r"Version\s+([\w.\-]+)", s)
        if m:
            ctx.mpqc_version = m.group(1)
        m = re.match(r"Revision\s+([0-9a-f]+)", s)
        if m:
            ctx.mpqc_revision = m.group(1)
        m = re.match(r"Machine:\s*(.*)", s)
        if m:
            ctx.machine = m.group(1).strip()
        m = re.match(r"Start Time:\s*(.*)", s)
        if m:
            ctx.start_time = m.group(1).strip()
        m = re.match(r"Default World:\s*(\d+)\s*MPI processes", s)
        if m:
            ctx.n_mpi = int(m.group(1))
        m = re.match(r"ThreadPool:\s*(\d+)\s+worker threads", s)
        if m:
            ctx.n_threads = int(m.group(1))

    # Embedded JSON input
    json_start = None
    for i, line in enumerate(lines[:120]):
        if INPUT_KEYVAL_START_RE.match(line):
            # JSON starts at the next non-blank line
            for j in range(i + 1, min(i + 6, len(lines))):
                if lines[j].strip().startswith("{"):
                    json_start = j
                    break
            break
    if json_start is not None:
        # Find matching brace count
        depth = 0
        body = []
        for k in range(json_start, len(lines)):
            line = lines[k]
            body.append(line)
            depth += line.count("{") - line.count("}")
            if depth == 0 and len(body) > 1:
                break
        try:
            joined = "".join(body)
            # Strip leading indent then parse
            joined_stripped = "\n".join(l.lstrip() for l in joined.splitlines())
            ctx.input_json = json.loads(joined_stripped)
        except Exception as e:
            ctx.input_json = {"_parse_error": str(e)}

    # Basis section: identify lines of form "<name> Basis = <basis>" followed by tiles/elements/tile-size lines.
    current_basis = None
    for i, line in enumerate(lines):
        m = BASIS_BLOCK_RE.match(line)
        if m and ("Basis" in line) and not "Constructing" in line:
            name, val = m.group(1), m.group(2).strip()
            current_basis = name
            ctx.bases[current_basis] = {"basis": val}
            continue
        if current_basis is not None:
            m_t = TILES_RE.search(line)
            if m_t:
                ctx.bases[current_basis]["tile_lo"] = int(m_t.group(1))
                ctx.bases[current_basis]["tile_hi"] = int(m_t.group(2))
                ctx.bases[current_basis]["elem_lo"] = int(m_t.group(3))
                ctx.bases[current_basis]["elem_hi"] = int(m_t.group(4))
                ctx.bases[current_basis]["n_tiles"] = ctx.bases[current_basis]["tile_hi"] - ctx.bases[current_basis]["tile_lo"]
                ctx.bases[current_basis]["dim"] = ctx.bases[current_basis]["elem_hi"] - ctx.bases[current_basis]["elem_lo"]
            m_ts = TILE_SIZE_RE.search(line)
            if m_ts:
                ctx.bases[current_basis]["tile_size_min"] = int(m_ts.group(1))
                ctx.bases[current_basis]["tile_size_max"] = int(m_ts.group(2))
                ctx.bases[current_basis]["tile_size_mean"] = int(m_ts.group(3))
                # We have everything for this basis, but keep scanning in case there's more.
            if GENERATING_RE.match(line):
                break

    # Index-space dim table — appears just before "Generating CSV-CCk equations".
    # Look in a window around that marker.
    gen_line = None
    for i, line in enumerate(lines):
        if GENERATING_RE.match(line):
            gen_line = i
            break
    if gen_line is not None:
        for k in range(max(0, gen_line - 40), gen_line):
            m = INDEX_SPACE_RE.match(lines[k])
            if m:
                # Use the label as-is (Unicode preserved); strip trailing combining marks already in source.
                label = m.group("label").strip()
                inst = int(m.group("inst"))
                dim = int(m.group("dim"))
                ctx.index_spaces[label] = {"dim": dim, "_decl_instance": inst}

    # Summary stats (PNO/OSV/PAO/CSV pairs) — scan the whole file once, cheap.
    for line in lines:
        for rx, key in (
            (PAO_AVG_RE, "pao_per_pair_avg"),
            (PNO_AVG_RE, "pno_per_pair_avg"),
            (OSV_AVG_RE, "osv_per_pair_avg"),
            (OSV_OFFDIAG_RE, "osv_off_diag_avg"),
        ):
            m = rx.search(line)
            if m and key not in ctx.summary_stats:
                ctx.summary_stats[key] = float(m.group(1))
        m = CSV_PAIRS_RE.search(line)
        if m and "csv_pairs_all" not in ctx.summary_stats:
            ctx.summary_stats["csv_pairs_all"] = int(m.group(1))
            ctx.summary_stats["csv_pairs_unique"] = int(m.group(2))
        m = MP2_PAIRS_RE.search(line)
        if m and "mp2_pairs_all" not in ctx.summary_stats:
            ctx.summary_stats["mp2_pairs_all"] = int(m.group(1))
            ctx.summary_stats["mp2_pairs_unique"] = int(m.group(2))

    return ctx


def parse_index_token(token: str) -> tuple[str, int, list[str]]:
    """Decompose a single index token like `a_2i_2` into (base_label, instance, csv_pair_labels).

    Examples:
        "i_2"       -> ("i", 2, [])
        "μ̃_19602"  -> ("μ̃", 19602, [])
        "a_2i_2"    -> ("a", 2, ["i_2"])
        "a_3i_1i_2" -> ("a", 3, ["i_1", "i_2"])
    """
    pieces = INDEX_PIECE_RE.findall(token)
    if not pieces:
        return (token, -1, [])
    base, inst = pieces[0][0], int(pieces[0][1])
    csv = [f"{p[0]}_{p[1]}" for p in pieces[1:]]
    return (base, inst, csv)


def split_indices(blob: str) -> list[str]:
    """Split a tensor's bra/ket blob like 'i_2,μ̃_19602;a_2i_2' into a flat list of tokens.

    Semicolons separate bra/ket groups (we don't distinguish here);
    commas separate indices within a group.
    """
    if not blob:
        return []
    return [t.strip() for t in re.split(r"[,;]", blob) if t.strip()]


def parse_tensor_expr(expr: str) -> dict | None:
    """Parse a tensor expression like 'g(i_2,i_1,Κ_1)' or 't(i_2;a_2i_2)' or 'I(μ̃_19602,i_2)'.

    Returns {label, raw_indices, base_indices, csv_pair_lists, kind}.
    `kind` is 'leaf' for g/f/s/t/C, 'intermediate' for I, 'scalar' for E/Z, 'other' otherwise.
    """
    expr = expr.strip()
    m = re.match(r"^(\w+)\s*\(([^)]*)\)\s*$", expr)
    if not m:
        # Could be a bare scalar identifier like 'E' or 'Z', or a literal like '-1'.
        if expr in ("E", "Z"):
            return {"label": expr, "raw_indices": [], "base_indices": [], "csv_pair_lists": [], "kind": "scalar"}
        if re.match(r"^-?\d+(\.\d+)?$", expr):
            return {"label": "_const", "raw_indices": [], "base_indices": [], "csv_pair_lists": [], "kind": "constant"}
        return None
    label = m.group(1)
    tokens = split_indices(m.group(2))
    decomposed = [parse_index_token(t) for t in tokens]
    base_indices = [d[0] for d in decomposed]
    csv_pair_lists = [d[2] for d in decomposed]
    if label in ("g", "f", "s", "t", "C"):
        kind = "leaf"
    elif label == "I":
        kind = "intermediate"
    elif label in ("E", "Z", "R"):
        kind = "scalar" if label in ("E", "Z") else "residual"
    else:
        kind = "other"
    return {
        "label": label,
        "raw_indices": tokens,
        "base_indices": base_indices,
        "csv_pair_lists": csv_pair_lists,
        "kind": kind,
    }


CHECKSUM_TOK_RE = re.compile(
    r"^checksum=(?P<nnz>-?\d+),(?P<sum>[^,]+),(?P<sumsq>[^,]+),(?P<max_abs>[^,]+)$"
)


def parse_eval_body(kind: str, body: str) -> dict:
    """Parse the `body` of an Eval line (columns between `<ns>ns |` and `<expr>`).

    Returns a dict with whichever of {left_B, right_B, result_B, alloc_B, hw_B,
    rss_B, checksum_nnz, checksum_sum, checksum_sumsq, checksum_max_abs} are
    present. The checksum_* fields only exist in logs from a SeQuant build
    patched to emit a `checksum=nnz,sum,sumsq,max_abs` token (real computed
    values, not the byte-size-derived `nnz` estimate below) — absent
    (empty dict keys) for logs from an unpatched build, and also absent
    for a specific step if the value wasn't computable (patch emits a
    literal `checksum=NA` token in that case, which this regex doesn't
    match, so it's silently skipped exactly like a missing token).
    """
    out = {}
    for tok in body.split("|"):
        tok = tok.strip()
        if not tok:
            continue
        m = re.match(r"^(left|right|result|alloc|hw|rss)\s*=\s*(\d+)B$", tok)
        if m:
            out[f"{m.group(1)}_B"] = int(m.group(2))
            continue
        m = CHECKSUM_TOK_RE.match(tok)
        if m:
            out["checksum_nnz"] = int(m.group("nnz"))
            out["checksum_sum"] = float(m.group("sum"))
            out["checksum_sumsq"] = float(m.group("sumsq"))
            out["checksum_max_abs"] = float(m.group("max_abs"))
    return out


def resolve_dense_shape(parsed: dict | None, index_spaces: dict[str, dict]) -> tuple[list[int | None], list[list[str]], list[int]]:
    """Map base index labels to dims; CSV-restricted indices resolve to None.

    Returns (shape_dense, csv_pair_lists, missing_indices_per_axis).
    """
    if not parsed or not parsed["base_indices"]:
        return ([], [], [])
    dims: list[int | None] = []
    for base, csv_list in zip(parsed["base_indices"], parsed["csv_pair_lists"]):
        if csv_list:
            dims.append(None)  # CSV/PNO per-pair virtual — extent not in log
            continue
        canonical = INDEX_LABEL_ALIASES.get(base, base)
        info = index_spaces.get(canonical) or index_spaces.get(base)
        if info is None:
            dims.append(None)
        else:
            dims.append(info["dim"])
    return (dims, parsed["csv_pair_lists"], [])


def parse_log(log_path: Path) -> tuple[HeaderContext, list[dict]]:
    with open(log_path, "r", encoding="utf-8", errors="replace") as f:
        lines = f.readlines()

    ctx = parse_header(lines)
    ctx.log_path = str(log_path)

    # Locate start of trace section: first line after "Generating CSV-CCk equations".
    start = 0
    for i, line in enumerate(lines):
        if GENERATING_RE.match(line):
            start = i + 1
            break

    # Iteration tracking: an iteration's steps come BEFORE the line `   N    ...` that summarises iter N.
    # We bump current_iter from N to N+1 *after* seeing that summary row.
    current_iter = 1
    term_idx = 0
    rows: list[dict] = []
    for i in range(start, len(lines)):
        line = lines[i].rstrip("\n")
        # CCSD iteration summary row
        if CC_ITER_HEADER_RE.match(line):
            continue
        m_iter = CC_ITER_ROW_RE.match(line)
        if m_iter:
            iter_n = int(m_iter.group("iter"))
            current_iter = iter_n + 1
            term_idx = 0  # reset on next iteration
            continue
        if line.lstrip().startswith("Term | Begin"):
            term_idx += 1
            continue
        m = EVAL_LINE_RE.match(line)
        if not m:
            continue
        kind = m.group("kind")
        ns = int(m.group("ns"))
        body = m.group("body")
        expr = m.group("expr").strip()

        body_cols = parse_eval_body(kind, body)
        result_B = body_cols.get("result_B", 0)
        nnz = result_B // 8

        # For Product lines, expr is "<lhs> * <rhs> -> <intermediate-or-output>"
        op_lhs = op_rhs = intermediate_label = None
        target_expr = expr
        if "->" in expr:
            lhs_rhs, target = expr.split("->", 1)
            intermediate_label = target.strip()
            target_expr = intermediate_label
            if "*" in lhs_rhs:
                a, b = lhs_rhs.split("*", 1)
                op_lhs, op_rhs = a.strip(), b.strip()

        parsed = parse_tensor_expr(target_expr)
        if parsed is None and kind == "Constant":
            parsed = {
                "label": "_const",
                "raw_indices": [],
                "base_indices": [],
                "csv_pair_lists": [],
                "kind": "constant",
            }

        shape_dense, csv_pair_lists, _ = resolve_dense_shape(parsed, ctx.index_spaces)
        dense_bytes = 8
        for d in shape_dense:
            if d is None:
                dense_bytes = None
                break
            dense_bytes *= d if dense_bytes is not None else 0
        if not shape_dense:
            dense_bytes = 8  # scalar
        sparsity = (result_B / dense_bytes) if (dense_bytes and dense_bytes > 0) else None

        # shape_repr — human-readable
        if shape_dense:
            shape_repr = " × ".join("?" if d is None else str(d) for d in shape_dense)
        else:
            shape_repr = "scalar"

        rows.append({
            "iter": current_iter,
            "term_idx": term_idx,
            "step_kind": kind,
            "kind": parsed["kind"] if parsed else "unknown",
            "label": parsed["label"] if parsed else "",
            "time_ns": ns,
            "left_B": body_cols.get("left_B", ""),
            "right_B": body_cols.get("right_B", ""),
            "result_B": result_B,
            "alloc_B": body_cols.get("alloc_B", ""),
            "hw_B": body_cols.get("hw_B", ""),
            "rss_B": body_cols.get("rss_B", ""),
            "nnz": nnz,
            "checksum_nnz": body_cols.get("checksum_nnz", ""),
            "checksum_sum": body_cols.get("checksum_sum", ""),
            "checksum_sumsq": body_cols.get("checksum_sumsq", ""),
            "checksum_max_abs": body_cols.get("checksum_max_abs", ""),
            "expr": expr,
            "target_expr": target_expr,
            "op_lhs": op_lhs or "",
            "op_rhs": op_rhs or "",
            "intermediate_label": intermediate_label or "",
            "index_labels": ",".join(parsed["base_indices"]) if parsed else "",
            "csv_pair_indices": ";".join(",".join(c) for c in csv_pair_lists) if csv_pair_lists else "",
            "shape_dense": ",".join("" if d is None else str(d) for d in shape_dense),
            "shape_repr": shape_repr,
            "dense_bytes": "" if dense_bytes is None else dense_bytes,
            "sparsity": "" if sparsity is None else f"{sparsity:.6g}",
        })
    return ctx, rows


def write_outputs(ctx: HeaderContext, rows: list[dict], out_dir: Path, stem: str) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"{stem}.steps.csv"
    header_path = out_dir / f"{stem}.header.json"

    if rows:
        fieldnames = list(rows[0].keys())
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(rows)
    else:
        csv_path.write_text("", encoding="utf-8")

    with open(header_path, "w", encoding="utf-8") as f:
        json.dump(asdict(ctx), f, indent=2, ensure_ascii=False)

    return csv_path, header_path


def summarise(ctx: HeaderContext, rows: list[dict]) -> str:
    n_total = len(rows)
    by_kind = {}
    for r in rows:
        by_kind[r["step_kind"]] = by_kind.get(r["step_kind"], 0) + 1
    leaf_rows = [r for r in rows if r["kind"] == "leaf"]
    inter_rows = [r for r in rows if r["kind"] == "intermediate"]
    labels = {}
    for r in leaf_rows:
        labels[r["label"]] = labels.get(r["label"], 0) + 1
    n_iters = max((r["iter"] for r in rows), default=0)
    lines = [
        f"  total steps    : {n_total}",
        f"  by step_kind   : {by_kind}",
        f"  leaf steps     : {len(leaf_rows)} (labels: {labels})",
        f"  inter steps    : {len(inter_rows)}",
        f"  iterations seen: {n_iters}",
        f"  index spaces   : { {k: v['dim'] for k, v in ctx.index_spaces.items()} }",
        f"  summary stats  : {ctx.summary_stats}",
    ]
    return "\n".join(lines)


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("logs", nargs="+", help="Trace log files to parse")
    ap.add_argument("--out", default=None, help="Output directory (default: alongside each log)")
    args = ap.parse_args(argv)

    for log in args.logs:
        log_path = Path(log)
        out_dir = Path(args.out) if args.out else log_path.parent
        stem = log_path.stem
        ctx, rows = parse_log(log_path)
        csv_path, header_path = write_outputs(ctx, rows, out_dir, stem)
        print(f"[parsed] {log_path}")
        print(f"   -> {csv_path}")
        print(f"   -> {header_path}")
        print(summarise(ctx, rows))
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
