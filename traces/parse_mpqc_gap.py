import re, sys

path = sys.argv[1] if len(sys.argv) > 1 else "/tmp/mpqc_fresh_gap_run.log"

# cck.ipp:1710-1736 (this session's instrumentation): for R in {1,2},
# evaluate_csv_closedshell(R) evaluates each term (emitting normal
# "Eval | <mode> | <ns>ns | ..." trace lines per term via sequant::evaluate),
# then accumulates via result += temp, and prints EXACTLY ONE aggregate
# "Eval | SumInplace | <ns>ns | R=<r> | n_terms=<n>" line per call (summing
# all the individual += timings for that call). All the per-term Eval
# lines for R's terms occur BEFORE that one SumInplace|R=.. line, in the
# same call. So: accumulate every generic "Eval | <mode> | <ns>ns |" line
# into a running total; when a "SumInplace | ... | R=r | n_terms=n" line
# is hit, that running total (plus the SumInplace ns itself) is R's FULL
# per-call cost; reset and continue. The unrelated "SumInplace | ... | E"
# lines (energy-scalar accumulation, a different/pre-existing instrumentation)
# don't match the R=/n_terms= pattern and are excluded entirely (not counted
# as generic Eval lines either, since they're SumInplace-kind, not a real
# per-term Product/Tensor/Permute step).

generic_eval_re = re.compile(r"Eval \| (\w+) \| (\d+)ns \|")
suminplace_r_re = re.compile(r"Eval \| SumInplace \| (\d+)ns \| R=(\d+) \| n_terms=(\d+)")

pending_ns = 0
r1_costs = []  # list of full per-call ns
r2_costs = []

with open(path, encoding="utf-8", errors="replace") as f:
    for line in f:
        m = suminplace_r_re.search(line)
        if m:
            ns, r, n_terms = int(m.group(1)), m.group(2), int(m.group(3))
            full_ns = pending_ns + ns
            if r == "1":
                r1_costs.append(full_ns)
            elif r == "2":
                r2_costs.append(full_ns)
            pending_ns = 0
            continue
        m = generic_eval_re.search(line)
        if m and m.group(1) != "SumInplace":
            pending_ns += int(m.group(2))

def summarize(name, costs):
    if not costs:
        print(f"{name}: no occurrences found")
        return
    secs = [c / 1e9 for c in costs]
    print(f"{name}: {len(secs)} occurrence(s), per-iteration seconds: "
          f"{[f'{s:.4f}' for s in secs]}")
    secs_sorted = sorted(secs)
    n = len(secs_sorted)
    median = secs_sorted[n // 2] if n % 2 else (secs_sorted[n // 2 - 1] + secs_sorted[n // 2]) / 2
    print(f"  median={median:.4f}s  mean={sum(secs)/n:.4f}s  min={min(secs):.4f}s  max={max(secs):.4f}s")
    return median

r1_med = summarize("R=1 (T1) full per-iteration cost", r1_costs)
r2_med = summarize("R=2 (T2) full per-iteration cost", r2_costs)
if r1_med and r2_med:
    print(f"\ncombined (R1+R2) median: {r1_med + r2_med:.4f}s")
