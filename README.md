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
bin/build-mpqc-sif.sh          # defaults to jianjianh1/mpqc-fork@93593706ed
```

That pinned commit's `cck.ipp` (`evaluate_csv_closedshell(R)`) carries
three additive, non-semantic instrumentation lines: `Eval | SumInplace |
<ns>ns | R=<r> | n_terms=<n>` (the `+=` accumulation step only —
`traces/parse_mpqc_gap.py` parses this, but it needs `sequant.trace.eval`
tracing on to also see per-term costs, which carries real checksum
overhead — see "Correctness vs. performance runs" below), `Eval |
WholeResidualWallTime | <ns>ns | R=<r> | n_terms=<n>` (a trace-INDEPENDENT
wall-clock timer around the whole real computation — use this one for
performance, `traces/checksum-run/ethane-perf.json` + no trace needed),
and `Eval | WholeResidualChecksum | R=<r> | checksum=nnz,sum,sumsq,max_abs`
(a direct whole-residual checksum, parsed by
`traces/extract_whole_residual_checksum.py` — MPQC never logs one
itself; per-term row-summing turned out unreliable for T2). Requires
Apptainer ≥ 1.5 and
passwordless root (`sudo apptainer build`); see `bin/mpqc.def`'s header
comment for the full flag reference. Build takes roughly 45-60 minutes
and produces a ~2.9 GB `.sif` — **not** committed here; rebuild it
locally (or point `-u`/the git ref at a different commit/fork if you've
made further changes).

## Run against the real ethane data

There are two input configs — use the right one for the question you're
asking, since `eval_level` tracing computes a real checksum on every
intermediate step, genuine extra work that inflates timing:

- **Correctness**: `traces/checksum-run/ethane-checksum-instrumented.json`
  (`sequant.trace.eval_level: 1` — needed to get any per-term detail, and
  the only way `Eval | WholeResidualChecksum` gets its data flowing
  through the traced code paths).
- **Performance**: `traces/checksum-run/ethane-perf.json` (same input,
  `sequant.trace` block removed entirely) — use this for any timing
  number; `Eval | WholeResidualWallTime` doesn't need tracing on at all.

Run single-rank (multi-rank MPQC on Pthreads hangs at cross-rank
collectives on at least this hardware — see the PaRSEC note below):

```bash
sudo bin/run-mpqc-mpirun.sh \
  ./mpqc-latest.sif \
  traces/checksum-run/ethane-perf.json \
  -n 1 --log ethane-run.log
```

Then extract results:

```bash
# whole-residual checksum (needs the correctness config, trace on)
python3 traces/extract_whole_residual_checksum.py ethane-run.log
# per-iteration real cost reconstructed from trace lines (also needs trace on)
python3 traces/parse_mpqc_gap.py ethane-run.log
```

Both tools default to **occurrence 2** of each `R=<r>` line — the state
entering CCSD iteration 2 (T1≡0, T2 from one deterministic pass off the
zero initial guess). This is the only iteration boundary safe to compare
across independent MPQC invocations; later iterations' PNO amplitudes
have a genuine run-to-run sign/gauge ambiguity on this molecule (ethane
has real orbital degeneracies — 2 CH₃ groups) that makes T2 (not T1)
numerically different — same nnz/sumsq magnitude, different sum/max_abs
— between separate runs even with identical code and input. Confirmed
directly this session: T2's checksum varied run-to-run across several
otherwise-identical SIF rebuilds. Don't be alarmed if your own T2 sum
doesn't match the reference table below exactly; T1 should always match
closely.

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

**CPU affinity/thread-binding does NOT transfer either — tried, made
things worse.** After finding that pinning `sequant-ta-repro`'s
Pthreads/MADNESS process to physical cores (`taskset -c 0-7` +
`MAD_NUM_THREADS=8`) gave a real ~20-24% win, the natural symmetric
question was whether the same idea helps MPQC's own PaRSEC side —
`bin/run-mpqc-mpirun.sh` sets `PARSEC_MCA_bind_threads=0` (PaRSEC's own
hwloc-based core-pinning, disabled) alongside `PARSEC_MCA_runtime_num_
cores=8` (thread-count cap only, no pinning). Tested both alternatives
(3 trials each, `ethane-perf.json`, single rank) against the current
default:
- `PARSEC_MCA_bind_threads=1` (PaRSEC's own per-thread core pinning):
  ~23% **slower** on both T1 and T2.
- `taskset -c 0-7` wrapping the whole run (external CPU-set constraint,
  PaRSEC's own dynamic balancing left on): no meaningful change (within
  run-to-run noise).

Read directly: PaRSEC already has its own topology-aware, dynamic
work-stealing scheduler, and forcing static thread-to-core pinning
prevents it from rebalancing away from naturally uneven load (this
workload's per-occupied-pair PNO domains are genuinely ragged) — the
current default (`bind_threads=0`, no external pinning) already looks
like a real local optimum for PaRSEC on this workload, not an oversight.
Correctness re-checked on both alternatives (T1 matched the reference
exactly; T2 showed the same already-documented run-to-run gauge
variability, not a new issue). Not adopting either change.

## Regenerating the SeQuant patch this compares against

The SeQuant fork that generates `sequant-ta-repro`'s residual code
lives at
[`jianjianh1/sequant-fork`](https://github.com/jianjianh1/sequant-fork),
branch `csv-ccsd-tiledarray-generator`.
