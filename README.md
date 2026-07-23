# mpqc-benchmark

Tooling to build real MPQC (an instrumented fork) and use it as the
ground-truth comparison target for
[`sequant-ta-repro`](https://github.com/jianjianh1/sequant-ta-repro), a
native SeQuant→TiledArray reproduction of MPQC's own closed-shell
CSV-CCSD T1/T2 residual evaluation. This repo does **not** vendor
MPQC's own (large) source tree — it builds it from a pinned commit on
[`jianjianh1/mpqc-fork`](https://github.com/jianjianh1/mpqc-fork) via
Apptainer, and carries only the trace-analysis tooling and real
ethane leaf/reference data this specific comparison needs.

## What's here

- `bin/` — build/run scripts for the instrumented MPQC SIF, plus the
  `.tns`/COO leaf-tensor conversion tools shared with `sequant-ta-repro`.
- `traces/` — trace-log parsers (`parse_trace.py`,
  `extract_trace_equations.py`, `term_begin_mapper.py`,
  `best_full_occurrence.py`, `compare_binarization.py`,
  `parse_mpqc_gap.py`) and the real ethane leaf/reference data this
  comparison was validated against (`traces/checksum-run/`).

## Build the instrumented SIF

```bash
bin/build-mpqc-sif.sh          # defaults to jianjianh1/mpqc-fork@3bf3413a10
```

That pinned commit is the first one where `cck.ipp`'s
`evaluate_csv_closedshell(R)` emits a real per-residual timing line
(`Eval | SumInplace | <ns>ns | R=<r> | n_terms=<n>`) on top of the
generic SeQuant `Eval | ...` trace lines — this is what
`traces/parse_mpqc_gap.py` parses. Requires Apptainer ≥ 1.5 and
passwordless root (`sudo apptainer build`); see `bin/mpqc.def`'s header
comment for the full flag reference. Build takes roughly 45-60 minutes
and produces a ~2.9 GB `.sif` — **not** committed here; rebuild it
locally (or point `-u`/the git ref at a different commit/fork if you've
made further changes).

## Run against the real ethane data

The exact input this investigation used is
`traces/checksum-run/ethane-checksum-instrumented.json` (closed-shell
CSV-CCSD, real ethane geometry/basis, `eval_level` trace tracing
enabled). Run single-rank (multi-rank MPQC on Pthreads hangs at
cross-rank collectives on at least this hardware — see the PaRSEC note
below):

```bash
sudo bin/run-mpqc-mpirun.sh \
  ./mpqc-latest.sif \
  traces/checksum-run/ethane-checksum-instrumented.json \
  -n 1 --log ethane-run.log
```

Then extract each residual's real per-iteration cost:

```bash
python3 traces/parse_mpqc_gap.py ethane-run.log
```

## Known-correct reference values

The real ethane leaf data in `traces/checksum-run/sptc_coo_iter1/` and
`tns_iter1/` (two oversized `.tns` files are gzip-compressed to stay
under GitHub's 100 MB file limit — `gunzip` them before use) is what
`sequant-ta-repro`'s native driver consumes to reproduce MPQC's own
whole-residual checksums:

| Residual | nnz | sum | sumsq | max_abs |
| --- | --- | --- | --- | --- |
| T1 (R=1) | 522 | -0.0888799157 | 0.0026454022 | 0.0151662826 |
| T2 (R=2) | 98598 | 0.2141888066 | 0.0997316723 | 0.0174497557 |

If `sequant-ta-repro`'s driver doesn't reproduce these against this
same leaf data, something in one of the two pipelines has drifted.

## PaRSEC vs. Pthreads (read before chasing "why is MPQC slower/faster")

MPQC's real production builds use the PaRSEC task backend
(`MADNESS_TASK_BACKEND=PaRSEC`, `bin/mpqc.def`'s default). The
MADNESS-ThreadPool-level tuning this investigation found valuable on
the `sequant-ta-repro` side (`SPTC_MAD_WAIT_POLICY=yield`,
`MAD_NUM_THREADS=10` beating the 16-thread hardware-concurrency
default) is **Pthreads-specific** — PaRSEC runs its own scheduler on
top of MADNESS and was independently confirmed (same TiledArray
commit, same workload) to be dramatically slower than Pthreads on this
hardware, so none of that tuning transfers to MPQC's actual production
configuration. Don't re-litigate this without a fresh measurement.

## Regenerating the SeQuant patch this compares against

The SeQuant fork that generates `sequant-ta-repro`'s residual code
lives at
[`jianjianh1/sequant-fork`](https://github.com/jianjianh1/sequant-fork),
branch `csv-ccsd-tiledarray-generator`.
