# Local calibration, 2026-10-24

The five JSON files here preserve the measured toy and 30-second MIPLIB
results described in [the benchmark report](../../../docs/benchmarks.md).
They retain incumbent variable values so `bench/cmd/compare -problems` can
recheck original-model feasibility. The full SCIP runner emitted a 540 MB
white-box trace; these checked-in files retain the comparison fields without
that event timeline.

- Machine: Apple M4, 10 logical CPUs, 32 GiB RAM, macOS arm64.
- Solver source: base commit `8d60ebb`, with benchmark-runner edits in this
  update. The q-opt result's legacy `v1-skeleton` version label was replaced
  with a Git revision/dirty marker in source after measurement; the solver
  engine did not change. The module import path was aligned with the actual
  GitHub repository after measurement; that packaging change also did not
  alter solver behavior.
- Go: 1.26.0. OR-Tools: 9.15.6755. HiGHS: 1.15.1. PySCIPOpt: 6.3.0,
  bundled SCIP: 10.0.2.
- MIPLIB budget: 30 seconds per instance, three problems concurrently.
- Toy budget: 21 measured runs and two warmups per instance.

Recheck recorded incumbents from the repository root:

```sh
go run ./bench/cmd/compare \
  -baseline bench/calibration/2026-10-24/ortools-toy.json \
  -candidate bench/calibration/2026-10-24/qopt-toy.json \
  -problems bench/problems

go run ./bench/cmd/compare \
  -baseline bench/calibration/2026-10-24/scip10-miplib-30s.json \
  -candidate bench/calibration/2026-10-24/qopt-miplib-30s.json \
  -problems bench/problems/miplib
```

The MIPLIB comparison reports warnings when time-limited incumbents differ;
it must report zero failures. Reproduce the original measurements with the
runner commands in the benchmark report. This is a short-budget calibration,
not the five-minute acceptance run.
