# Benchmarks

## Reproducibility

All runners build models from the JSON files in `bench/problems/`. The toy
suite checks statuses, objectives, solution values, and feasibility against
OR-Tools. The 13-instance MIPLIB subset additionally reports incumbents,
valid dual bounds, gaps, proof status, nodes, and elapsed time. A time-limited
incumbent is checked against the original JSON model before it is accepted.

Run each MIPLIB suite with problem-level concurrency of three and the same
per-instance solve budget. Results depend on solver version, CPU, workload,
and model insertion order. SCIP's native MPS path is a diagnostic and is not
directly comparable to q-opt's JSON construction path.

Commands and result schema are in the [benchmark harness README](https://github.com/qizhanchan/qopt/blob/main/bench/README.md).

## October 24, 2026 local calibration

Platform: Apple M4 (10 logical CPUs), macOS arm64, Go 1.26.0. Solver source
revision: `8d60ebb` plus benchmark-harness changes in this update. Python
baselines were installed into an isolated virtual environment: OR-Tools
9.15.6755, HiGHS 1.15.1, and SCIP
10.0.2 via PySCIPOpt 6.3.0. The SCIP version differs from the historical
SCIP 11 baseline below, so these are separate comparison cohorts.

The toy suite used 21 measured runs and two warmups per instance. q-opt ran
with `-parallel 3`; OR-Tools ran sequentially because its toy runner has no
parallel option. The comparison reported **11 ok, 3 warn, 0 fail** at `1e-5`.
Two warnings are alternate optima with matching objectives; the third is
GLOP's infeasible status for an unbounded example where q-opt reports
unbounded. The aggregate of per-instance median times was 0.541 ms for
q-opt and 5.450 ms for OR-Tools; these tiny-model measurements are dominated
by interface and setup costs and should not be extrapolated to MIPLIB.

The MIPLIB calibration uses a 30-second per-instance budget. It is a quick
regression and coverage check, not a replacement for the five-minute
acceptance gate in the [milestones](solver-milestones.md).

```sh
go run ./bench/cmd/qopt_bench \
  -problems bench/problems/miplib -out /tmp/qopt-miplib-30s.json \
  -time-limit 30s -parallel 3 -lp-backend revised \
  -branching reliability -node-selection best-estimate \
  -best-estimate-best-bound-freq 10 \
  -rins -rins-adaptive-budget -local-branching -proximity-search

.venv/bin/python bench/runners/highs_miplib_baseline.py \
  --problems bench/problems/miplib --out /tmp/highs-miplib-30s.json \
  --time-limit-min 0.5 --parallel 3

.venv/bin/python bench/runners/scip_miplib_baseline.py \
  --problems bench/problems/miplib --out /tmp/scip10-miplib-30s.json \
  --time-limit-min 0.5 --parallel 3 --input-mode json-api \
  --expected-version 10.0.2
```

All three solvers produced incumbents on all 13 instances. Each proved one
within 30 seconds: q-opt proved `markshare_4_0`, while HiGHS and SCIP proved
`neos17`. The entries below are **objective (relative gap)**; ✓ means proven
optimal. For minimization, smaller incumbent objectives are better. Gaps use
each solver's own reported valid bound, so they measure proof progress as well
as incumbent quality.

| Instance | q-opt | HiGHS 1.15 | SCIP 10.0.2 |
|---|---:|---:|---:|
| `assign1-5-8` | 212 (9.3%) | 212 (7.5%) | 212 (7.3%) |
| `gen-ip002` | -4783.73 (0.5%) | -4775.4 (1.1%) | -4783.73 (0.7%) |
| `gen-ip054` | 6853.08 (0.9%) | 6853.96 (1.1%) | 6864.93 (1.2%) |
| `glass4` | 3.55003e+09 (77.5%) | 1.90002e+09 (55.0%) | 1.97145e+09 (59.4%) |
| `ic97_potential` | 4181 (7.5%) | 3949 (0.9%) | 4044 (3.6%) |
| `mad` | 0.1208 (12.1%) | 0.1956 (19.6%) | 0.17 (17.0%) |
| `markshare_4_0` | 1 ✓ | 1 (100.0%) | 1 (100.0%) |
| `neos-3046615-murg` | 1824 (82.6%) | 1609 (78.2%) | 1710 (73.7%) |
| `neos-3754480-nidda` | 15890.8 (3143.9%) | 14356.8 (5499.8%) | 13848.5 (3868.6%) |
| `neos17` | 0.155052 (6.9%) | 0.150003 ✓ | 0.150003 ✓ |
| `neos5` | 15 (7.5%) | 15 (6.5%) | 15 (5.8%) |
| `pk1` | 11 (28.5%) | 15 (93.3%) | 11 (66.0%) |
| `timtab1` | 788306 (59.2%) | 764774 (8.8%) | 764772 (7.9%) |

The [committed calibration data](https://github.com/qizhanchan/qopt/tree/main/bench/calibration/2026-10-24)
retains per-instance values, bounds, gaps, nodes, and solve times. The two
MIPLIB cross-solver comparisons each reported **0 failures** after checking
the recorded incumbents against the original JSON models. Their warnings
reflect different time-limited objectives and proof states. The calibration
files omit SCIP's very large event timeline; rerun the command above when
that telemetry is needed.

## Historical five-minute reference

The July 2026 measured profile produced verified incumbents on all 13
instances and proved `markshare_4_0` and `pk1`. The matched optimized SCIP
11.0.0 JSON-API reference produced 13 incumbents and proved four instances:
`markshare_4_0`, `pk1`, `neos17`, and `timtab1`. Those numbers are historical
and should be compared only with a new five-minute run under the same
conditions. See the [SCIP baseline details](scip-baseline.md).
