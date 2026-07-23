#!/usr/bin/env python3
"""Convert MPQC's `.coo.tns` value dumps into sptc-bench text-COO `.tns`.

For each leaf tensor that sptc-bench's loader expects
(`/users/jianjian/sptc-bench/ta-bench/src/ta_benchmark_main.cpp:load_coo`
 calls), translate the corresponding MPQC dump file.

Filenames follow data column order. The two C tensors are emitted
as `C_i_1_m_1_a_1.tns` / `C_i_1_i_2_m_1_a_1.tns` (canonical, matches
data layout); we additionally create symlinks under the SeQuant
aliases `C_m_1_a_1_i_1.tns` / `C_m_1_a_1_i_1_i_2.tns` so callers
that rely on either naming see the same file.

For the C tensors we also "append" the per-pair inner indices across
the i_1 (occupied) tile axis: each i_1 tile gets a unique contiguous
range of inner indices `[k*tile_size, (k+1)*tile_size)`. Tile size
defaults to max-per-pair across the file; override with `--tile-size`
or `--match-sptc-bench` to enforce a smaller uniform tile (and
truncate higher-index PNOs per pair to match sptc-bench's reported
inner extent).

See traces/SPTC_BENCH_MAPPING.md for the full crosswalk.
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path
from typing import Iterable, Optional

# Explicit path insertion rather than relying on Python's implicit
# sys.path[0] = script's own directory (only true when run directly by
# path) — this import would otherwise silently break under `python -m`,
# or if this script is ever copied/packaged without its _axis_map.py
# sibling.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _axis_map import AXIS_MAP as FILENAME_AXIS_MAP  # noqa: E402


# Map MPQC `# expr=` header → (canonical filename, alias filename or None, perm).
# `canonical` matches the data's actual column order; `alias` is the
# SeQuant-semantic name (with angle-bracket parameters listed last)
# that sptc-bench's reference dataset also carries. Empirically:
#   - expr0000 `g{i_1;i_2;Κ_1}` shape=7,7,282 — symmetric (occ,occ,ri); no perm.
#   - expr0001 `g{i_1;μ̃_1;Κ_1}` shape=114,7,282 — emitted as (μ̃_1, i_1, Κ_1),
#     but downstream wants (i_1, μ̃_1, Κ_1) = (occ, uocc, ri). Perm = [1,0,2].
#   - expr0002 `g{μ̃_1;μ̃_2;Κ_1}` shape=114,114,282 — symmetric (uocc,uocc,ri).
#   - expr0007 `C{μ̃_1;a_1<i_1>}` outer_shape=7,114 — already (i_1, μ̃_1) = (occ, uocc).
#   - expr0008 `C{μ̃_1;a_1<i_1,i_2>}` outer_shape=7,7,114 — already (occ, occ, uocc).
# Each value is a dict with keys:
#   canonical : output filename (column-order)
#   alias     : optional SeQuant-alias filename, emitted as a symlink
#   perm      : optional index-column permutation (applied before append)
#   kind      : "flat"        — plain rank-N COO, no inner-axis append
#               "tot_single"  — ToT with one virtual column; partition by
#                                col0 (= i_1) and append inner index
#                                across i_1 tiles. Used for C1/C2/T1.
#               "tot_t2"      — ToT with TWO virtuals packed into one
#                                inner column (a_1*n_pno_pair + a_2);
#                                requires `.tile.tns` sidecar to recover
#                                n_pno_pair per outer pair, then unpacks
#                                and appends both virtuals.
TENSOR_MAP = {
    # g (flat)
    "g{i_1;i_2;Κ_1}:N-S-S":      {"canonical": "g_i_1_i_2_Κ_1.tns",      "alias": None, "perm": None,      "kind": "flat"},
    "g{i_1;μ̃_1;Κ_1}:N-S-S":    {"canonical": "g_i_1_m_1_Κ_1.tns",      "alias": None, "perm": (1, 0, 2), "kind": "flat"},
    "g{μ̃_1;μ̃_2;Κ_1}:N-S-S":   {"canonical": "g_m_1_m_2_Κ_1.tns",      "alias": None, "perm": None,      "kind": "flat"},
    # f (flat)
    "f{i_1;i_2}:N-S-S":          {"canonical": "f_i_1_i_2.tns",          "alias": None, "perm": None,      "kind": "flat"},
    "f{μ̃_1;i_1}:N-N-S":        {"canonical": "f_i_1_m_1.tns",          "alias": None, "perm": (1, 0),    "kind": "flat"},
    "f{μ̃_1;μ̃_2}:N-S-S":       {"canonical": "f_m_1_m_2.tns",          "alias": None, "perm": None,      "kind": "flat"},
    # s (flat)
    "s{μ̃_1;μ̃_2}:N-S-S":       {"canonical": "s_m_1_m_2.tns",          "alias": None, "perm": None,      "kind": "flat"},
    # C (single-virtual ToT) — SeQuant alias names also emitted as symlinks.
    # tile_role: "osv" for rank-3 C (contracts with T1); "pno" for rank-4 C (contracts with T2).
    "C{μ̃_1;a_1<i_1>}:N-C-S":   {"canonical": "C_i_1_m_1_a_1.tns",     "alias": "C_m_1_a_1_i_1.tns",     "perm": None, "kind": "tot_single", "tile_role": "osv"},
    "C{μ̃_1;a_1<i_1,i_2>}:N-C-S": {"canonical": "C_i_1_i_2_m_1_a_1.tns", "alias": "C_m_1_a_1_i_1_i_2.tns", "perm": None, "kind": "tot_single", "tile_role": "pno"},
    # T1 (single-virtual ToT) — reuses C1's OSV tile_size for element-wise contractability with C1.
    "t{a_1<i_1>;i_1}:N-N-S":     {"canonical": "t_i_1_a_1.tns",           "alias": None, "perm": None,    "kind": "tot_single", "tile_role": "osv"},
    # T2 (packed-double-virtual ToT) — reuses C2's PNO tile_size for contractability with C2.
    "t{a_1<i_1,i_2>,a_2<i_1,i_2>;i_1,i_2}:N-N-S": {
        "canonical": "t_i_1_i_2_a_1_a_2.tns", "alias": None, "perm": None, "kind": "tot_t2", "tile_role": "pno",
    },
}

# FILENAME_AXIS_MAP (filename → semantic axis label per column position) is
# imported above from bin/_axis_map.py — used by the empty-file shape
# fallback (below) and by bin/check-sptc-shapes.py, which shares the same
# source of truth now (see code-review finding on check-sptc-shapes.py:23).


_KV_TOKEN_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_.]*=")


def parse_header(lines: Iterable[str]) -> dict:
    """Return a dict of header `# key=value` pairs (until first non-`#` line).

    The cck dumper writes two header shapes (see analyze_tns.py's
    parse_header_kv, which hit this same bug first): most lines have 2+
    `key=value` tokens with no spaces inside any value (e.g.
    `# kind=tile  array=flat`), but some have exactly one `key=value` whose
    value contains spaces (e.g. `# rank_note=outer only; inner is reported
    as a single extent column`). An unconditional `.split()` treats every
    whitespace as a token boundary and truncates the latter at the first
    space.

    Distinguishing on a raw count of `=` characters doesn't work: a
    single-key line's free-text value can itself contain a stray `=` (e.g.
    the real header `# rank_note=outer modes; inner contributes 1 extra
    mode = local PNO/OSV index` has 2 `=` signs but only 1 actual key).
    Instead, count whitespace-separated tokens that themselves look like
    `identifier=...` (_KV_TOKEN_RE) — on a genuine multi-kv line every
    token matches that shape; on a single-key spacey-value line only the
    first token does, regardless of how many bare `=` appear later in the
    prose.
    """
    hdr = {}
    for line in lines:
        if not line.startswith("#"):
            break
        body = line.lstrip("#").strip()
        kv_tokens = [tok for tok in body.split() if _KV_TOKEN_RE.match(tok)]
        if len(kv_tokens) >= 2:
            for tok in kv_tokens:
                k, _, v = tok.partition("=")
                hdr[k] = v
        elif "=" in body:
            k, _, v = body.partition("=")
            hdr[k.strip()] = v.strip()
    return hdr


def normalize_expr(expr: str) -> str:
    """Strip whitespace inside braces so the TENSOR_MAP keys match."""
    return re.sub(r"\s+", "", expr)


def read_inner_volumes(tile_tns_path: str) -> dict[tuple[int, ...], int]:
    """Parse a `.tile.tns` ToT sidecar and return {outer_elem_global: volume}.

    The format (cck.ipp:1980-1983 `dump_tile_tot`) is:
      # format=outer_tile_idx(N) outer_tile_extent(N)
                outer_elem_idx_global(N) inner_dense_volume(1)
    so for each row of 3N+1 ints, the outer-element multi-index lives
    at columns [2N..3N) and the inner volume is the last column.
    """
    volumes: dict[tuple[int, ...], int] = {}
    rank = None
    with open(tile_tns_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.startswith("#"):
                # Extract rank from `# rank=N` (the OUTER rank — `format=`
                # has `outer_tile_idx(N)` which equals rank).
                if line.startswith("# rank=") and rank is None:
                    try:
                        rank = int(line.split("=", 1)[1].strip().split()[0])
                    except (ValueError, IndexError):
                        pass
                continue
            if not line.strip():
                continue
            tokens = line.split()
            if rank is None:
                # Per the format `outer_tile_idx(N) outer_tile_extent(N)
                # outer_elem_idx_global(N) inner_dense_volume(1)`, total
                # cols = 3N+1. Infer N from the first data row.
                rank = (len(tokens) - 1) // 3
            outer_elem = tuple(int(t) for t in tokens[2 * rank : 3 * rank])
            volume = int(tokens[3 * rank])
            volumes[outer_elem] = volume
    return volumes


def read_tile_boundaries(
    tile_tns_path: str, pair_key_rank: int
) -> tuple[list[list[int]], dict[tuple[int, ...], int]]:
    """Parse a `.tile.tns` ToT sidecar's REAL outer tiling for the leading
    `pair_key_rank` pair-key dimensions, for the proof-of-concept
    "replicate MPQC's real ToT tiling" pass (see
    `ta-bench/src/ta_builder.h`'s `build_tot_array()`, which currently
    forces these dims to tile size 1 rather than using MPQC's own
    coarser, real tile boundaries).

    Returns:
      dim_boundaries: for each of the first `pair_key_rank` outer dims, a
        sorted list of tile-start element offsets, first entry 0, PLUS a
        final entry equal to that dimension's total extent — directly
        usable as a `TA::TiledRange1` boundary list (matches `make_tr1()`'s
        own convention in `ta_builder.h`).
      tile_pad_volume: {tile_idx[:pair_key_rank] -> max(inner_dense_volume)
        over every row whose outer_tile_idx starts with that prefix}. Only
        the pair-key dims matter here because a ToT leaf's PNO/OSV domain
        size depends only on the occupied pair, never on any other outer
        dim (e.g. C2's third outer dim, μ̃, doesn't affect PNO count at
        all — confirmed empirically: inner_dense_volume is constant across
        every μ̃ position for a fixed occupied pair in the real dump).

    Same file format as `read_inner_volumes()` above (cck.ipp's
    `dump_tile_tot`): each data row is `outer_tile_idx(N)
    outer_tile_extent(N) outer_elem_idx_global(N) inner_dense_volume(1)`,
    3N+1 integers, N = outer rank (read from `# rank=N`).
    """
    import bisect

    rank = None
    # dim d -> {tile_idx[d] -> (extent[d], min_elem_global[d])}, keyed by
    # the FILE's OWN tile_idx numbering (pass 1 only; not used past this).
    per_dim_tiles: list[dict[int, tuple[int, int]]] = []
    # Raw (elem_global[:pair_key_rank], volume) rows, re-aggregated in pass
    # 2 below once dim_boundaries (with any frozen-orbital prepend) is
    # final — needed because prepending a leading empty tile shifts every
    # OTHER tile's ordinal by +1, and the file's own tile_idx numbering
    # (used as-is) would then be off by one against the boundary list.
    raw_rows: list[tuple[tuple[int, ...], int]] = []

    with open(tile_tns_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.startswith("#"):
                if line.startswith("# rank=") and rank is None:
                    try:
                        rank = int(line.split("=", 1)[1].strip().split()[0])
                    except (ValueError, IndexError):
                        pass
                continue
            if not line.strip():
                continue
            tokens = line.split()
            if rank is None:
                rank = (len(tokens) - 1) // 3
            if not per_dim_tiles:
                per_dim_tiles = [dict() for _ in range(rank)]
            vals = [int(t) for t in tokens]
            tile_idx = vals[0:rank]
            tile_extent = vals[rank : 2 * rank]
            elem_global = vals[2 * rank : 3 * rank]
            volume = vals[3 * rank]

            for d in range(min(pair_key_rank, rank)):
                prior = per_dim_tiles[d].get(tile_idx[d])
                if prior is None:
                    per_dim_tiles[d][tile_idx[d]] = (tile_extent[d], elem_global[d])
                else:
                    prior_extent, prior_min = prior
                    if prior_extent != tile_extent[d]:
                        raise ValueError(
                            f"{tile_tns_path}: dim {d} tile {tile_idx[d]} has "
                            f"inconsistent outer_tile_extent ({prior_extent} "
                            f"vs {tile_extent[d]}) across rows — the "
                            "product-grid tiling assumption this extractor "
                            "relies on doesn't hold for this file"
                        )
                    per_dim_tiles[d][tile_idx[d]] = (
                        prior_extent,
                        min(prior_min, elem_global[d]),
                    )

            raw_rows.append((tuple(elem_global[:pair_key_rank]), volume))

    # Pass 1 result: boundaries per dim, MPQC's real (active-orbital-only)
    # range prepended with a [0, first_active) tile if needed — frozen-core
    # orbitals (always lowest-index, standard convention) never appear in
    # MPQC's own ToT dump, but the flat COO file this tiling applies to
    # declares the FULL occupied extent starting at 0. That leading tile
    # will be legitimately empty/zero (no C/T rows reference a frozen
    # orbital), not a boundary-arithmetic bug — confirmed: ethane C2's real
    # dump starts at element 2 of a declared-9-wide dimension, i.e. 2
    # frozen 1s orbitals, and starts[-1] + last_extent lands on 9 exactly.
    dim_boundaries: list[list[int]] = []
    for d in range(pair_key_rank):
        tiles = per_dim_tiles[d]
        starts = sorted(min_elem for _extent, min_elem in tiles.values())
        last_tidx = max(tiles, key=lambda k: tiles[k][1])
        last_extent, _last_start = tiles[last_tidx]
        total_extent = starts[-1] + last_extent if starts else 0
        if starts and starts[0] != 0:
            starts = [0] + starts
        dim_boundaries.append(starts + [total_extent])

    # Pass 2: re-derive each row's tile ordinal from the FINAL boundaries
    # (bisect into the boundary list, independent of the file's own
    # tile_idx numbering) so tile_pad_volume's keys are guaranteed
    # consistent with dim_boundaries even after the frozen-tile prepend.
    tile_pad_volume: dict[tuple[int, ...], int] = {}
    for elem_global, volume in raw_rows:
        pad_key = tuple(
            bisect.bisect_right(dim_boundaries[d], elem_global[d]) - 1
            for d in range(pair_key_rank)
        )
        tile_pad_volume[pad_key] = max(tile_pad_volume.get(pad_key, 0), volume)

    return dim_boundaries, tile_pad_volume


def write_tiling_spec(
    out_path: str, dim_boundaries: list[list[int]], tile_pad_volume: dict[tuple[int, ...], int]
) -> None:
    """Emit the plain-text sidecar `build_tot_array()`'s proof-of-concept
    real-tiling override reads (see plan: "same ad hoc whitespace format
    coo_loader.h already parses elsewhere, no new format to design").

    Format:
      # pair_key_rank=<N>
      # dim <d> boundaries: <b0> <b1> ... <bK>     (one line per dim)
      <tile_idx_0> ... <tile_idx_{N-1}> <pad_volume>   (one line per tile)
    """
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(f"# pair_key_rank={len(dim_boundaries)}\n")
        for d, boundaries in enumerate(dim_boundaries):
            f.write(f"# dim {d} boundaries: {' '.join(str(b) for b in boundaries)}\n")
        for tile_idx, pad_volume in sorted(tile_pad_volume.items()):
            f.write(" ".join(str(t) for t in tile_idx) + f" {pad_volume}\n")


def sptc_tile_size_for(ref_dir: str, canonical_name: str) -> Optional[int]:
    """Inspect a sptc-bench reference file for the same canonical tensor.

    Returns the inner tile_size = stride between consecutive i_1 tile
    starts in sptc-bench's data. sptc-bench enforces a uniform tile
    size by partitioning the inner axis by col0 (= i_1); tile k starts
    at `min(inner) over rows with col0 = lobound + k`, and tile_size =
    `tile_start[1] - tile_start[0]`.

    For sptc-bench's ethane C-tensors: both yield tile_size = 87 even
    though individual (i_1, i_2) sub-groups within a tile have smaller
    inner ranges (87 is the UNIFORM stride, not per-pair extent).

    Tries both `.tns` and `.txt` suffixes (sptc-bench's released data
    uses `.txt`; this project uses `.tns`).
    """
    candidates = [
        os.path.join(ref_dir, canonical_name),
        os.path.join(ref_dir, canonical_name.replace(".tns", ".txt")),
    ]
    ref = next((p for p in candidates if os.path.isfile(p)), None)
    if ref is None:
        return None
    # Map col0 → min(inner) seen in that tile.
    tile_starts: dict[int, int] = {}
    with open(ref, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            tokens = line.split()
            if len(tokens) < 2:
                continue
            indices = [int(t) for t in tokens[:-1]]
            col0 = indices[0]
            inner = indices[-1]
            cur = tile_starts.get(col0)
            if cur is None or inner < cur:
                tile_starts[col0] = inner
    sorted_starts = sorted(tile_starts.items())  # by col0
    if len(sorted_starts) < 2:
        # Single tile — fall back to max+1 of that tile.
        if not sorted_starts:
            return None
        return None  # caller will fall back to auto
    # tile_size = stride between consecutive (sorted) tile starts.
    (_, s0), (_, s1) = sorted_starts[0], sorted_starts[1]
    return s1 - s0


def emit_sptc_coo(
    in_path: str,
    out_path: str,
    alias_path: Optional[str],
    molecule: str,
    tile_size_override: Optional[int],
    match_sptc_bench_dir: Optional[str],
    tile_ctx: Optional[dict] = None,
    axis_ctx: Optional[dict] = None,
) -> tuple[int, list[int]]:
    """Read an MPQC `.coo.tns` and write sptc-bench-format text-COO.

    Returns (nnz, shape). Writes the canonical file at `out_path` and,
    if `alias_path` is non-None, creates a relative symlink there
    pointing at the canonical file.

    `tile_ctx` / `axis_ctx`: shared dicts across a single main()
    invocation. tile_ctx anchors OSV/PNO tile sizes across C↔T pairs
    (rank-3 C sets 'osv'; rank-4 C sets 'pno'; T1 reuses 'osv'; T2
    reuses 'pno'). axis_ctx caches per-axis extents (occ, pao, ri)
    from non-empty tensors so an empty flat file can back-fill its
    shape consistently with the rest of the molecule.
    """
    if tile_ctx is None:
        tile_ctx = {}
    if axis_ctx is None:
        axis_ctx = {}

    with open(in_path, "r", encoding="utf-8") as f:
        lines = list(f)

    hdr = parse_header(lines)
    expr = normalize_expr(hdr.get("expr", ""))
    if expr not in TENSOR_MAP:
        raise ValueError(f"{in_path}: expression {expr!r} not in sptc-bench map")
    entry = TENSOR_MAP[expr]
    canonical_name = entry["canonical"]
    perm = entry["perm"]
    kind = entry["kind"]
    tile_role = entry.get("tile_role")

    shape_str = hdr.get("outer_shape", hdr.get("shape", ""))
    outer_shape = [int(x) for x in shape_str.split(",") if x]
    if perm is not None:
        outer_shape = [outer_shape[perm[i]] for i in range(len(perm))]

    # First pass: collect rows + observe per-column extents.
    raw_rows: list[tuple[list[int], float]] = []
    n_idx_cols = None
    per_col_max: list[int] = []
    for line in lines:
        if line.startswith("#") or not line.strip():
            continue
        tokens = line.split()
        if len(tokens) < 2:
            raise ValueError(f"{in_path}: short data line: {line!r}")
        val = float(tokens[-1])
        indices = [int(t) for t in tokens[:-1]]
        if perm is not None:
            indices = [indices[i] for i in perm]
        if n_idx_cols is None:
            n_idx_cols = len(indices)
            per_col_max = [-1] * n_idx_cols
        for i, x in enumerate(indices):
            if x > per_col_max[i]:
                per_col_max[i] = x
        raw_rows.append((indices, val))

    def resolve_tile_size(label_for_log: str) -> int:
        """Pick tile_size from tile_ctx anchor / override / sptc-bench ref / auto.

        The tile_ctx anchor takes precedence when this file's tile_role
        (osv/pno) has already been set by a prior C-tensor in this
        run. That's what makes T1 reuse C1's tile_size and T2 reuse
        C2's — mandatory for element-wise contractability.
        """
        if tile_role and tile_role in tile_ctx:
            ts = tile_ctx[tile_role]
            print(f"  [tile-size reusing {tile_role} anchor] tile_size={ts}")
            return ts
        if tile_size_override is not None:
            ts = tile_size_override
        elif match_sptc_bench_dir is not None:
            ts_ref = sptc_tile_size_for(match_sptc_bench_dir, canonical_name)
            if ts_ref is None:
                print(
                    f"WARNING: --match-sptc-bench specified but {canonical_name} "
                    f"not found under {match_sptc_bench_dir}; falling back to "
                    "auto for " + label_for_log,
                    file=sys.stderr,
                )
                ts = per_col_max[-1] + 1
            else:
                ts = ts_ref
                print(f"  [tile-size from sptc-bench reference] tile_size={ts}")
        else:
            ts = per_col_max[-1] + 1  # auto: max per-pair PNO
        if tile_role:
            tile_ctx[tile_role] = ts  # anchor for subsequent T-tensors
        return ts

    if n_idx_cols is None:
        # Empty file: back-fill shape from axis_ctx (populated by earlier
        # non-empty files in this molecule). Falls through to the
        # header's outer_shape only if the axis hasn't been seen yet.
        #
        # For ToT kinds (tot_single/tot_t2), `outer_shape` is only the
        # OUTER portion (e.g. just `i_1` for T1, rank 1) — the appended
        # virtual/PNO axis/axes normally come from actually walking data
        # rows (see the tot_single/tot_t2 branches below), which an empty
        # file has none of. FILENAME_AXIS_MAP's `axes` DOES include those
        # appended axes (e.g. ["occ","osv"] for t_i_1_a_1.tns, rank 2), so
        # `len(axes) == len(outer_shape)` is deliberately false here and
        # must NOT fall through to `final_shape = outer_shape` — doing so
        # silently drops the ToT rank (bug: emitted `rank=1 shape=7` for
        # an all-zero T1 instead of `rank=2 shape=7,<osv extent>`, which
        # crashes TA::einsum when this leaf is later used at its documented
        # rank). Backfill the missing inner axis/axes from tile_ctx's
        # `<tile_role>_inner_total`, populated by this same tile_role's
        # non-empty C-tensor sibling (C before T, by file-processing order)
        # so the empty T-tensor still matches C's inner extent exactly —
        # required for element-wise contractability between them.
        axes = FILENAME_AXIS_MAP.get(canonical_name, [])
        if kind == "flat":
            if axes and len(axes) == len(outer_shape):
                final_shape = [axis_ctx.get(a, outer_shape[d]) for d, a in enumerate(axes)]
                if final_shape != outer_shape:
                    print(
                        f"  [empty file — using axis_ctx extents] shape={final_shape}"
                    )
            else:
                final_shape = outer_shape
        else:
            n_outer = len(outer_shape)
            outer_axes = axes[:n_outer]
            final_outer = (
                [axis_ctx.get(a, outer_shape[d]) for d, a in enumerate(outer_axes)]
                if outer_axes and len(outer_axes) == n_outer
                else outer_shape
            )
            n_inner = max(len(axes) - n_outer, 1 if kind == "tot_single" else 2)
            inner_total = tile_ctx.get(f"{tile_role}_inner_total", 0) if tile_role else 0
            final_shape = final_outer + [inner_total] * n_inner
            print(
                f"  [empty ToT file — using axis_ctx/tile_ctx extents] "
                f"shape={final_shape}"
            )
        data_rows: list[tuple[list[int], float]] = []

    elif kind == "flat":
        # Flat: pass through with the per-column max+1 sized shape.
        # If a MPQC dump was capped (100M-row limit hit mid-write), the
        # last few tiles never made it, so max+1 on those columns under-
        # reports the true extent. Cross-check against axis_ctx (set by
        # earlier non-capped tensors on the same axis) and take the max.
        per_col_extent = [m + 1 for m in per_col_max]
        outer_n = len(outer_shape)
        axes = FILENAME_AXIS_MAP.get(canonical_name, [])
        capped = any(l.strip().startswith("# trailer") and "capped=true" in l
                     for l in lines[-20:])
        final_shape = list(per_col_extent)
        for d in range(outer_n):
            a = axes[d] if d < len(axes) else None
            known = axis_ctx.get(a) if a else None
            if known is not None and known > per_col_extent[d]:
                # Trust the previously-established axis extent — the
                # current file is a capped or partially-populated dump.
                final_shape[d] = known
                print(
                    f"  [flat axis {d} ({a}) back-filled from axis_ctx: "
                    f"{per_col_extent[d]} → {known}"
                    + (" — capped dump" if capped else "")
                    + "]"
                )
            elif per_col_extent[d] > outer_shape[d]:
                print(
                    f"WARNING {in_path}: outer axis {d} observed max+1 "
                    f"{per_col_extent[d]} > header shape {outer_shape[d]}; "
                    f"data may be inconsistent with the dump",
                    file=sys.stderr,
                )
        data_rows = raw_rows
        # Register these extents in axis_ctx for later fallbacks. Use the
        # (possibly-back-filled) final_shape rather than the raw observed
        # per_col_extent so downstream files see a consistent extent.
        for d, a in enumerate(axes[:len(final_shape)]):
            axis_ctx.setdefault(a, final_shape[d])

    elif kind == "tot_single":
        # ToT with one virtual: partition inner axis by col0 (= i_1).
        # Each i_1 tile gets a unique contiguous inner range
        # [k*tile_size, (k+1)*tile_size). Truncate rows whose
        # inner_local_idx >= tile_size (matches sptc-bench's uniform
        # tile width).
        tile_size = resolve_tile_size("inner")
        col0_lobound = min(r[0][0] for r in raw_rows)

        data_rows = []
        n_truncated = 0
        for indices, val in raw_rows:
            inner = indices[-1]
            if inner >= tile_size:
                n_truncated += 1
                continue
            tile_k = indices[0] - col0_lobound
            new_inner = tile_k * tile_size + inner
            new_indices = list(indices[:-1]) + [new_inner]
            data_rows.append((new_indices, val))
        if n_truncated > 0:
            print(
                f"  [tile-size={tile_size}] dropped {n_truncated} rows where "
                f"inner_local_idx >= tile_size (keeping first {tile_size} "
                "PNOs per pair)"
            )

        max_new_inner = max(r[0][-1] for r in data_rows) if data_rows else -1
        outer_extents = [m + 1 for m in per_col_max[:-1]]
        final_shape = outer_extents + [max_new_inner + 1]
        # Register outer-axis extents in axis_ctx for later empty-file fallbacks.
        axes = FILENAME_AXIS_MAP.get(canonical_name, [])
        for d, a in enumerate(axes[:len(outer_extents)]):
            axis_ctx.setdefault(a, outer_extents[d])
        # Register this tile_role's actual total inner extent so a LATER
        # empty file sharing the same tile_role (e.g. T1 after C1) can
        # still emit the correct ToT rank instead of collapsing to just
        # its outer shape (see the `n_idx_cols is None` branch above).
        if tile_role:
            tile_ctx[f"{tile_role}_inner_total"] = final_shape[-1]

    elif kind == "tot_t2":
        # T2: inner column packs two virtuals as
        #   packed = a_1_local * n_pno_pair + a_2_local
        # n_pno_pair varies per outer (i_1, i_2) pair. We read it from
        # the .tile.tns sidecar (inner_dense_volume = n_pno_pair²).
        sidecar = in_path.replace(".coo.tns", ".tile.tns")
        if not os.path.isfile(sidecar):
            raise FileNotFoundError(
                f"T2 needs the .tile.tns sidecar for unpacking; "
                f"expected {sidecar}. Re-run MPQC with tns_mode='tile' (the "
                "default) so the sidecar is emitted."
            )
        volumes = read_inner_volumes(sidecar)
        # n_pno_pair[(i_1, i_2)] = int(sqrt(volume)); assert volume is square.
        n_pno: dict[tuple[int, int], int] = {}
        for outer, vol in volumes.items():
            n = int(round(vol ** 0.5))
            if n * n != vol:
                raise ValueError(
                    f"{sidecar}: outer pair {outer} has inner_dense_volume={vol} "
                    f"which is not a perfect square — T2 packed-inner unpacking "
                    "assumption (n_pno_pair²) doesn't hold."
                )
            n_pno[outer] = n  # type: ignore[index]

        # Tile size for the appended virtuals — partition by col0 (i_1).
        # T2's auto-mode CANNOT use per_col_max[-1] (= max packed inner =
        # max(n_pno_pair²)−1, which is way too big). The right "auto" is
        # the max n_pno_pair across all pairs, since each virtual axis
        # individually only goes up to n_pno_pair−1. But the tile_ctx
        # anchor (populated by rank-4 C, tile_role='pno') takes
        # precedence — that's what makes T2 element-wise contractable
        # against C2.
        if tile_role and tile_role in tile_ctx:
            tile_size = tile_ctx[tile_role]
            print(f"  [tile-size reusing {tile_role} anchor] tile_size={tile_size}")
        elif tile_size_override is not None:
            tile_size = tile_size_override
        elif match_sptc_bench_dir is not None:
            ts = sptc_tile_size_for(match_sptc_bench_dir, canonical_name)
            if ts is None:
                tile_size = max(n_pno.values()) if n_pno else 0
                print(
                    f"  [auto tile-size from unpacked T2 PNO counts] tile_size={tile_size}"
                )
            else:
                tile_size = ts
                print(f"  [tile-size from sptc-bench reference] tile_size={tile_size}")
        else:
            tile_size = max(n_pno.values()) if n_pno else 0
            print(f"  [auto tile-size from unpacked T2 PNO counts] tile_size={tile_size}")
        if tile_role:
            tile_ctx.setdefault(tile_role, tile_size)
        col0_lobound = min(r[0][0] for r in raw_rows)

        data_rows = []
        n_truncated = 0
        n_missing = 0
        for indices, val in raw_rows:
            i_1 = indices[0]
            i_2 = indices[1]
            packed = indices[-1]
            n = n_pno.get((i_1, i_2))
            if n is None:
                n_missing += 1
                continue
            a_1_local, a_2_local = divmod(packed, n)
            if a_1_local >= tile_size or a_2_local >= tile_size:
                n_truncated += 1
                continue
            tile_k = i_1 - col0_lobound
            a_1_global = tile_k * tile_size + a_1_local
            a_2_global = tile_k * tile_size + a_2_local
            data_rows.append(([i_1, i_2, a_1_global, a_2_global], val))
        if n_truncated > 0:
            print(
                f"  [tile-size={tile_size}] T2 dropped {n_truncated} rows "
                "where a_1_local or a_2_local >= tile_size"
            )
        if n_missing > 0:
            print(
                f"  WARNING: T2 dropped {n_missing} rows with no sidecar "
                "volume entry — likely a partial dump",
                file=sys.stderr,
            )

        if data_rows:
            max_i_1 = max(r[0][0] for r in data_rows)
            max_i_2 = max(r[0][1] for r in data_rows)
            max_a_1 = max(r[0][2] for r in data_rows)
            max_a_2 = max(r[0][3] for r in data_rows)
            final_shape = [max_i_1 + 1, max_i_2 + 1, max_a_1 + 1, max_a_2 + 1]
            # See the tot_single branch's identical comment: registers this
            # tile_role's inner extent so a later fully-empty ToT file can
            # still emit the correct rank instead of collapsing to outer-only.
            if tile_role:
                tile_ctx[f"{tile_role}_inner_total"] = max(max_a_1, max_a_2) + 1
        else:
            final_shape = outer_shape + [0, 0]
    else:
        raise ValueError(f"unsupported tensor kind: {kind!r}")

    nnz = len(data_rows)
    # Floating-point checksums emitted in the header:
    #   sum    — Σ value, order-invariant; catches missing/extra rows.
    #   sum2   — Σ value²  (Frobenius norm² for sparse storage). Invariant
    #            under unitary rotations of any MO axis — should match
    #            sptc-bench's reference EVEN for the MO-bearing tensors
    #            that diverge in tier-3 value comparison, because
    #            localization is just a unitary rotation among occupieds.
    #            (For purely PAO tensors this also matches.)
    val_sum = sum(v for _, v in data_rows)
    val_sum2 = sum(v * v for _, v in data_rows)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        shape_csv = ",".join(str(s) for s in final_shape)
        f.write(
            f"# shape={shape_csv} nnz={nnz} rank={len(final_shape)} "
            f"sum={val_sum!r} sum2={val_sum2!r} "
            f"framework=mpqc molecule={molecule} source=sptc-bench-mapping\n"
        )
        for indices, val in data_rows:
            f.write(" ".join(str(i) for i in indices))
            f.write(f" {val!r}\n")

    # Emit SeQuant-alias as a relative symlink (or copy on platforms
    # that forbid symlinks).
    if alias_path is not None:
        # If the alias path already exists, remove first — os.symlink
        # otherwise raises FileExistsError.
        if os.path.islink(alias_path) or os.path.exists(alias_path):
            os.remove(alias_path)
        rel_target = os.path.relpath(out_path, os.path.dirname(alias_path))
        try:
            os.symlink(rel_target, alias_path)
        except OSError as e:
            print(
                f"WARNING: symlink {alias_path} -> {rel_target} failed ({e}); "
                "falling back to copy",
                file=sys.stderr,
            )
            import shutil
            shutil.copy2(out_path, alias_path)

    return nnz, final_shape


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "in_dir",
        help="MPQC tns dir (e.g. /proj/.../jianjian-alkanes/tns/ethane_tns/)",
    )
    ap.add_argument(
        "--out",
        required=True,
        help="output dir (e.g. /proj/.../sptc_coo/C2H6/)",
    )
    ap.add_argument(
        "--molecule",
        required=True,
        help="molecule label for the output header (e.g. C2H6)",
    )
    ap.add_argument(
        "--tile-size",
        type=int,
        default=None,
        help="enforce a uniform inner tile size for ToT C tensors; rows "
             "with inner_local_idx >= tile-size are dropped. Default: "
             "auto (max per-pair PNO observed in the file).",
    )
    ap.add_argument(
        "--match-sptc-bench",
        default=None,
        help="auto-detect tile-size by reading the equivalent tensor from "
             "the given sptc-bench reference dir.",
    )
    args = ap.parse_args()

    coo_files = sorted(
        os.path.join(args.in_dir, f)
        for f in os.listdir(args.in_dir)
        if f.endswith(".coo.tns")
    )
    if not coo_files:
        print(
            f"ERROR: no .coo.tns files in {args.in_dir} — rerun MPQC with "
            "tns_include_values=true",
            file=sys.stderr,
        )
        return 1

    # Shared per-molecule contexts. `tile_ctx` anchors OSV/PNO tile
    # sizes across C↔T pairs (rank-3 C → 'osv' → T1; rank-4 C → 'pno' → T2)
    # so element-wise contractions in all_equations.txt work. `axis_ctx`
    # caches extents seen on non-empty files so an empty flat file (e.g.
    # f_i_1_m_1 for ethane) can back-fill its shape consistently.
    # NB: coo_files is already sorted, so expr0007/0008 (C) are processed
    # before expr0009/0010 (T) — the assertion below catches regressions.
    tile_ctx: dict[str, int] = {}
    axis_ctx: dict[str, int] = {}
    n_emitted = 0
    n_skipped = 0
    for in_path in coo_files:
        with open(in_path, "r", encoding="utf-8") as f:
            head = [next(f, "") for _ in range(2)]
        hdr = parse_header(head)
        expr = normalize_expr(hdr.get("expr", ""))
        if expr not in TENSOR_MAP:
            print(f"  skip {os.path.basename(in_path)}: {expr!r} not in sptc-bench map")
            n_skipped += 1
            continue
        entry = TENSOR_MAP[expr]
        canonical_name = entry["canonical"]
        alias_name = entry.get("alias")
        # T-tensors must be processed AFTER their C partner so tile_ctx is
        # anchored. Confirm the ordering assumption.
        tile_role = entry.get("tile_role")
        if tile_role and entry["kind"] in ("tot_t2",) and tile_role not in tile_ctx:
            # T2 with no anchor yet — indicates a processing-order regression.
            print(
                f"WARNING: T2 processed before its rank-4 C partner — "
                f"tile_ctx['{tile_role}'] is unset. Auto-mode will be used, "
                "breaking element-wise contractability against C2.",
                file=sys.stderr,
            )
        out_path = os.path.join(args.out, canonical_name)
        alias_path = os.path.join(args.out, alias_name) if alias_name else None
        nnz, shape = emit_sptc_coo(
            in_path, out_path, alias_path, args.molecule,
            args.tile_size, args.match_sptc_bench,
            tile_ctx=tile_ctx, axis_ctx=axis_ctx,
        )
        link_note = f" (+ alias {alias_name})" if alias_name else ""
        print(f"  wrote {canonical_name}: nnz={nnz} shape={shape}{link_note}")
        n_emitted += 1

    if tile_ctx:
        print(f"\ntile anchors: {tile_ctx}")
    if axis_ctx:
        print(f"axis extents: {axis_ctx}")

    print(f"\n{n_emitted} tensor files emitted, {n_skipped} skipped")
    # All 11 MPQC leaves are now mapped (3 g + 3 f + 1 s + 2 C + 2 T).
    if n_emitted < 11:
        print(
            f"WARNING: expected 11 tensor files, got {n_emitted}. "
            "Check that all 11 MPQC expr dumps are present.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
