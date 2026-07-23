# MPQC `.tile.tns` ↔ sptc-bench text-COO mapping

This document maps MPQC's 11 PNO-CCSD leaf-tensor dumps (under
`<tns_outdir>/expr*.tile.tns`) onto the 5 tensors that
`/users/jianjian/sptc-bench/` consumes from
`dataset/data_fusedsptc/<molecule>/*.txt`.

It is the source-of-truth for `bin/tns-to-sptc-coo.py` (the
forthcoming converter) and for `bin/compare-sptc-coo.py` (the
cross-validation).

## Both ends share

- **Chemistry**: DLPNO-CCSD on the linear n-alkane series
  (C2H6, C3H8, C4H10, C5H12, C6H14, plus C7H16 / C8H18 in our
  extension), cc-pVDZ-f12 basis, aug-cc-pVDZ-RI auxiliary, frozen
  1s on each carbon.
- **Index alphabet**: `i_n` for occupied; auxiliary indexed by
  the Greek letter **Κ** (U+039A, UTF-8 `\xCE\x9A`). Both writers
  emit it literally in headers/filenames.
- **PNO/OSV indices**: `a_n<i_m>` for OSV, `a_n<i_m,i_p>` for PNO.
  Inner dimension is the per-pair PNO/OSV size; SeQuant ToT and
  sptc-bench-loader's `virtual` dimension are the same concept.
- **0-based integer indices**, double-precision values.

## Index-symbol convention difference

| Concept | sptc-bench | MPQC SeQuant | filename slot |
|---|---|---|---|
| occupied | `i_1`, `i_2` | `i_1`, `i_2` | `i_1`, `i_2` |
| unoccupied / PAO | `m_1`, `m_2` | `μ̃_1`, `μ̃_2` | bare `1`, `2` (sanitized) |
| auxiliary (RI) | `Κ_1` | `Κ_1` | `1` in filename, `Κ_1` in `# expr=` header |
| OSV / PNO virtual | (inner of `c1`/`c2`) | `a_n<...>` | `a_n_i_1[_i_2]` |

**The MPQC dump filename sanitizes μ̃ → bare integer.** The `# expr=`
line inside the file keeps the literal `μ̃`. Always parse the
in-file header — never the filename — to recover symbol identity.

## The five overlapping leaves

| sptc-bench file | sptc-bench field | shape contract | MPQC expr ordinal | MPQC SeQuant expr | MPQC dump filename |
|---|---|---|---|---|---|
| `g_i_1_i_2_Κ_1.txt` | `g1` | (occ, occ, ri) | 0000 | `g{i_1;i_2;Κ_1}:N-S-S` | `expr0000_g_i_1_i_2_1_N-S-S.tile.tns` |
| `g_i_1_m_1_Κ_1.txt` | `g` (also called g2) | (occ, uocc, ri) | 0001 | `g{i_1;μ̃_1;Κ_1}:N-S-S` | `expr0001_g_i_1_1_1_N-S-S.tile.tns` |
| `g_m_1_m_2_Κ_1.txt` | `g0` | (uocc, uocc, ri) | 0002 | `g{μ̃_1;μ̃_2;Κ_1}:N-S-S` | `expr0002_g_1_2_1_N-S-S.tile.tns` |
| `C_m_1_a_1_i_1.txt` | `c1` | (occ, uocc, virtual) | 0007 | `C{μ̃_1;a_1<i_1>}:N-C-S` | `expr0007_C_1_a_1_i_1_N-C-S.tile.tns` |
| `C_m_1_a_1_i_1_i_2.txt` | `c2` | (occ, occ, uocc, virtual) | 0008 | `C{μ̃_1;a_1<i_1,i_2>}:N-C-S` | `expr0008_C_1_a_1_i_1_i_2_N-C-S.tile.tns` |

Sources of truth:
- sptc-bench field declarations:
  `/users/jianjian/sptc-bench/ta-bench/src/ta_tensors.h:13-26`
- sptc-bench filename literals (UTF-8 `\xCE\x9A` = `Κ`):
  `/users/jianjian/sptc-bench/ta-bench/src/ta_benchmark_main.cpp` (5
  `load_coo(dir + "/...")` calls)
- MPQC expr writers: `dump_tile_flat` (flat, N-S-S indices) and
  `dump_tile_tot` (ToT, N-C-S indices), at
  `/users/jianjian/mpqc4/src/mpqc/chemistry/qc/lcao/cc/cck.ipp:1911-2011`

## Mode-pattern suffix glossary

The trailing `:X-Y-Z` in expressions and the `_X-Y-Z` in filenames
encodes one mode-symbol per outer index of the dumped tensor.

| Symbol | Meaning | Storage shape |
|---|---|---|
| `N` | Normal — tiled occupied (`i_1`), block-sparse, ~4 elem/tile | block-sparse outer |
| `S` | Simple — orbital or DF index (`μ̃_*`, `Κ_*`), one big tile | flat block-sparse |
| `C` | Composite — CSV index with inner dense block (`a_*<...>`) | ToT, inner dense |

So `N-S-S` is a 3-leg block-sparse tensor with no inner dimension
(`dump_tile_flat`). `N-C-S` is a ToT with outer rank 2 and a per-pair
inner dimension (`dump_tile_tot`). `N-N-S` is 3-leg flat block-sparse
(used for t1, but the inner virtual is encoded as the `a_*` outer
index, hence the second `N`).

## Index ORDER differences

**MPQC's SeQuant uses the bra–ket convention.** `g{i_1;i_2;Κ_1}` is
`(i_1, i_2, Κ_1)` in dump order. sptc-bench's `g_i_1_i_2_Κ_1.txt`
asserts shape `(occ, occ, ri)` — same order. **No transpose needed.**

For `c1 = C{μ̃_1; a_1<i_1>}`, MPQC's outer order is
`(μ̃_1, a_1<i_1>)` but the ToT writer emits outer tiles followed by
the per-pair inner dimension. sptc-bench wants `(occ, uocc, virtual)`
= `(i_1, μ̃_1, a_local)`. **Transpose needed**: pull the parameter
`i_1` out of the angle-bracket to the front, keep `μ̃_1` second, and
the per-pair `a_local` becomes the third. The MPQC ToT dump's
`outer_elem_idx_global(N)` field gives `(i_1, μ̃_1)`; the inner
local index gives `a_local`.

For `c2 = C{μ̃_1; a_1<i_1,i_2>}`, MPQC's outer is
`(μ̃_1, a_1<i_1,i_2>)`. sptc-bench's `(occ, occ, uocc, virtual)` =
`(i_1, i_2, μ̃_1, a_local)`. Same transpose pattern: pull both
parameters out front. MPQC's outer_elem_idx_global gives
`(i_1, i_2, μ̃_1)`; inner local gives `a_local`.

## All 11 leaves (Phase F update)

The converter now emits **all 11 MPQC leaf tensors** in
column-order-matching `.tns` files. The 5 sptc-bench-compatible
tensors (3 g + 2 C) are joined by 6 more (3 f + 1 s + T1 + T2)
that sptc-bench's reference dataset doesn't carry but its
`equations_simplified.txt` flags as "filtered (need f/s tensors)"
for future equations.

| MPQC expr | Canonical filename | Rank | Notes |
| --- | --- | --- | --- |
| `g{i_1;i_2;Κ_1}` | `g_i_1_i_2_Κ_1.tns` | 3 | flat |
| `g{i_1;μ̃_1;Κ_1}` | `g_i_1_m_1_Κ_1.tns` | 3 | flat, perm `(1,0,2)` |
| `g{μ̃_1;μ̃_2;Κ_1}` | `g_m_1_m_2_Κ_1.tns` | 3 | flat, symmetric |
| `f{i_1;i_2}` | `f_i_1_i_2.tns` | 2 | flat (Fock, occ-occ) |
| `f{μ̃_1;i_1}` | `f_i_1_m_1.tns` | 2 | flat, perm `(1,0)` |
| `f{μ̃_1;μ̃_2}` | `f_m_1_m_2.tns` | 2 | flat (Fock, PAO-PAO) |
| `s{μ̃_1;μ̃_2}` | `s_m_1_m_2.tns` | 2 | flat (PAO overlap) |
| `C{μ̃_1;a_1<i_1>}` | `C_i_1_m_1_a_1.tns` | 3 | ToT-single + alias `C_m_1_a_1_i_1.tns` |
| `C{μ̃_1;a_1<i_1,i_2>}` | `C_i_1_i_2_m_1_a_1.tns` | 4 | ToT-single + alias `C_m_1_a_1_i_1_i_2.tns` |
| `t{a_1<i_1>;i_1}` | `t_i_1_a_1.tns` | 2 | T1: ToT-single, shares OSV append with C1 |
| `t{a_1<i_1,i_2>,a_2<i_1,i_2>;i_1,i_2}` | `t_i_1_i_2_a_1_a_2.tns` | 4 | T2: ToT-double (packed inner unpacked via sidecar) |

### T2 packed-inner unpacking

MPQC stores T2 (rank-2 outer × rank-2 inner) with the two virtuals
packed into a single inner column:
```
packed_inner = a_1_local * n_pno_pair + a_2_local
```
where `n_pno_pair` is the per-(i_1, i_2)-pair PNO count, recovered
from the `.tile.tns` sidecar's `inner_dense_volume` (= n_pno_pair²).
The converter reads the sidecar, computes `n_pno_pair = sqrt(volume)`
per pair, then unpacks: `a_1_local = packed // n_pno_pair`,
`a_2_local = packed % n_pno_pair`. Both virtuals are then appended
across i_1 tiles (same convention as C2) so T2 is element-wise
contractable against C2.

T2 symmetry under (i_1, i_2)↔(i_2, i_1) is preserved bit-for-bit
after unpacking (verified on ethane v3: Σv² for (2,3,*,*) equals
Σv² for (3,2,*,*) exactly), confirming the unpacking direction.

### Element-wise contractability

- C1 and T1 share the OSV append rule (partition by i_1). Both use
  the same tile_size (e.g. 87 in sptc-bench ethane, auto-detected
  from max OSV per i_1 in our v3).
- C2 and T2 share the PNO append rule (partition by i_1). Both use
  the same tile_size (e.g. 87 in sptc-bench ethane, or whatever
  auto-detects from max per-pair PNO).
- C1↔C2 tile sizes may differ in auto-mode (different per-pair
  PNO/OSV counts); pass `--tile-size N` or `--match-sptc-bench
  REF_DIR` to enforce a common value across both groups.

## Original: tensors NOT in sptc-bench's reference dataset

Sptc-bench's release dataset only ships the 5 g+C tensors; its
`equations_simplified.txt` filters out 13 equations that need f/s.
Our converter emits ALL 11 leaves regardless — sptc-bench-compatible
files for the 5 they have, and additional MPQC-native files for the
6 they don't (yet):

| MPQC expr ordinal | SeQuant expr | Purpose |
|---|---|---|
| 0003 | `f{i_1;i_2}:N-S-S` | Fock occ-occ |
| 0004 | `f{μ̃_1;i_1}:N-N-S` | Fock PAO-occ |
| 0005 | `f{μ̃_1;μ̃_2}:N-S-S` | Fock PAO-PAO |
| 0006 | `s{μ̃_1;μ̃_2}:N-S-S` | PAO overlap |
| 0009 | `t{a_1<i_1>;i_1}:N-N-S` | T1 amplitude |
| 0010 | `t{a_1<i_1,i_2>,a_2<i_1,i_2>;i_1,i_2}:N-N-S` | T2 amplitude |

These are still useful (we'll keep them in our MPQC output) but the
converter ignores them when projecting to sptc-bench format.

## Tile-config / shape consistency

`sptc-bench/ta-bench/src/ta_tensors.h:7-10` defines default tile sizes
`occ=4, uocc=50, ri=200`. **These are tile sizes, not total dims**;
total dimensions are taken from the COO file (max index + 1 per axis).
The actual C2H6 shapes will be smaller per axis (e.g., occ=7 for
frozen-core C2H6) and the COO files will reflect that.

Our MPQC dumps' shapes for ethane (cc-pVDZ-f12, frozen 1s):
- occupied: 7 (= 9 valence MOs - 2 frozen 1s on each C)
- PAO (μ̃): 114
- RI (Κ): 282
- per-pair OSV: variable (see expr0007 inner_dense_volume column)
- per-pair PNO: variable (see expr0008 inner_dense_volume column)

Compared with sptc-bench's defaults `occ=4, uocc=50, ri=200`, our shapes
**will not match** their defaults — but sptc-bench's actual dataset
files presumably encode the real dims from whatever generator produced
them. Tier-1 shape-comparison in Phase D will tell us how far apart
the two pipelines drift.

## Header convention for the converter

sptc-bench's dumper (`ta-bench/src/ta_dumper.h:147-230`) writes:

```
# shape=d0,d1,...,dN nnz=K rank=R framework=ta molecule=... equation=... stage=... nranks=... trial=...
idx0 idx1 ... idxN <double>
...
```

For our static, sweep-runner-agnostic dataset, the converter will emit:

```
# shape=d0,d1,...,dN nnz=K rank=R framework=mpqc molecule=<name> source=sptc-bench-mapping
idx0 idx1 ... idxN <double>
...
```

(Omit `equation/stage/nranks/trial` — they're benchmark-only.)

Loader is permissive of unknown header keys
(`coo_loader.h:23-74`: it walks lines, splits on whitespace, treats
the last token as the value; header lines starting with `#` are
detected by `getline` reading the first non-comment line).
