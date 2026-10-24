# q-opt

Pure Go CPU LP/MIP solver with a reproducible benchmark harness against
OR-Tools, HiGHS, and SCIP.

**Online documentation:** [q-opt docs](https://qizhanchan.github.io/qopt/)

The project is still a correctness-first solver, but the current engine is no
longer a dense-tableau prototype. The default LP backend is a bounded-variable
revised simplex with sparse columns, sparse LU, incremental node re-solves,
and dual-simplex warm starts. MIP search is single-threaded.

## Experimental project

q-opt is an experimental solver maintained by a Go developer. It is a place to
implement, measure, and refine LP/MIP algorithms in Go. The API, numerical
behavior, and search settings may change as the solver matures; validate
results against the original model before relying on them in an application.

Go keeps the solver self-contained and makes it practical to iterate quickly
on the LP engine, search rules, and benchmark harness. In a short development
cycle, q-opt progressed from a reference tableau to sparse revised simplex and
branch-and-bound with warm starts and primal heuristics. On the selected
13-instance MIPLIB subset at a 30-second budget, q-opt found a verified
incumbent for every instance and proved one optimum, matching the incumbent
coverage and proof count of the measured HiGHS and SCIP runs. Incumbent quality
and proof gaps still vary substantially by instance; see the
[dated comparison](docs/benchmarks.md) for the full results and conditions.

## Projects that informed this work

- [SCIP](https://github.com/scipopt/scip) is the main reference for MIP search
  and solver engineering: reliability branching, node selection, separation,
  cut-row lifecycle, and infeasibility proofs. Focused comparisons with SCIP
  source and behavior are recorded in the [solver history](docs/solver-history.md).
- [HiGHS](https://github.com/ERGO-Code/HiGHS) is a reference for sparse LP
  solver architecture and performance. The revised-simplex backend is
  implemented in Go and measured against HiGHS on shared models.
- [Google OR-Tools](https://github.com/google/or-tools) provides the toy-suite
  correctness reference through GLOP and CBC. Its results are checked against
  q-opt's statuses, objectives, and original-model feasibility.

The library itself is pure Go. The external solvers are used by optional
benchmark scripts, not by q-opt at runtime.

## Current status

- LP: continuous variables, minimization/maximization, `<=` / `>=` / `=`
  rows, scaling, presolve, structured statuses, and postsolve recovery.
- MIP: integer/binary variables, sequential branch-and-bound, valid global
  dual bounds and gaps, reliability branching, and DFS/best-bound/
  best-estimate node selection.
- Primal search: rounding, repair, mixed-integer feasibility pump,
  coefficient/pseudocost diving, and small RENS neighborhoods.
- Optional incumbent refinement: adaptive RINS, local branching, and
  proximity search, coordinated by a measured neighborhood scheduler when
  several are enabled.
- Optional bound strengthening: Gomory GMI, row MIR, root structural
  detection, big-M tightening, orbitope fixings, implications, and a small set
  of cover/implied-bound/flow-cover experiments.
- Validation: toy LP/MIP comparison, original-model incumbent feasibility
  checks, deterministic snapshots, MIPLIB runners, and per-phase LP/search
  telemetry.

The dense two-phase tableau remains available as `LPBackendTableau`, mainly
as a reference backend. The revised/sparse backend is the default.

## Example

```go
package main

import (
    "context"
    "fmt"

    qopt "github.com/qizhanchan/qopt"
)

func main() {
    m := qopt.NewModel(qopt.Maximize)
    x := m.AddVar(qopt.VarSpec{Name: "x", Type: qopt.Binary})
    y := m.AddVar(qopt.VarSpec{Name: "y", Type: qopt.Binary})

    _ = m.SetObjective(map[qopt.Var]float64{x: 6, y: 5})
    _ = m.AddConstraint(
        map[qopt.Var]float64{x: 4, y: 3},
        qopt.LessOrEqual,
        5,
    )

    res, err := m.Solve(context.Background(), qopt.Options{})
    fmt.Println(res.Status, res.Objective, err)
}
```

The zero-value `Options` uses revised simplex, sparse LU, depth-first node
selection, and most-fractional branching. The most useful opt-in search
configuration measured on the checked-in MIPLIB subset is:

```go
qopt.Options{
    TimeLimit:                    5 * time.Minute,
    LPBackend:                    qopt.LPBackendRevised,
    MIPBranching:                 qopt.MIPBranchReliability,
    MIPNodeSelection:             qopt.MIPBestEstimate,
    BestEstimateBestBoundFreq:    10,
    UseRINS:                      true,
    RINSAdaptiveBudget:           true,
    UseLocalBranching:            true,
    UseProximitySearch:           true,
}
```

This is an experimental MIPLIB-oriented profile, not a new library default.
It improves incumbent quality on several instances but can spend proof time
inside sub-MIPs. See [`model.go`](model.go) for the complete option set.

## Test

Use Go 1.26 or newer:

```sh
go test ./...
go test -race ./...
go vet ./...
git diff --check
```

`QOPT_DEBUG=1` enables additional LP, presolve, cut, and search diagnostics.

## Benchmark

Always use problem-level concurrency for suite runs. The standard local value
is `-parallel 3`; serial MIPLIB runs waste wall time without changing each
instance's configured solve budget.

Toy suite:

```sh
go run ./bench/cmd/qopt_bench \
  -problems bench/problems -out /tmp/qopt-toy.json \
  -repeat 21 -warmup 2 -parallel 3

.venv/bin/python bench/runners/ortools_bench.py \
  --problems bench/problems --out /tmp/ortools-toy.json \
  --repeat 21 --warmup 2

go run ./bench/cmd/compare \
  -baseline /tmp/ortools-toy.json \
  -candidate /tmp/qopt-toy.json \
  -problems bench/problems
```

Recommended five-minute q-opt profile:

```sh
go run ./bench/cmd/qopt_bench \
  -problems bench/problems/miplib \
  -out /tmp/qopt-miplib-5m.json \
  -time-limit 5m -parallel 3 \
  -lp-backend revised \
  -branching reliability \
  -node-selection best-estimate \
  -best-estimate-best-bound-freq 10 \
  -rins -rins-adaptive-budget \
  -local-branching -proximity-search
```

Matched SCIP 11 baseline using the same JSON model construction path:

```sh
.venv/bin/python bench/runners/scip_miplib_baseline.py \
  --problems bench/problems/miplib \
  --out /tmp/scip11-miplib-5m.json \
  --time-limit-min 5 --parallel 3 \
  --input-mode json-api
```

Use `--input-mode native-mps` only for source-level SCIP diagnostics. Model
insertion order changes the search trajectory enough that native MPS is not a
fair head-to-head baseline for q-opt's converted JSON input.

Create the Python environment with `python3 -m venv .venv` and install the
comparison packages with `.venv/bin/python -m pip install ortools highspy
pyscipopt`. The SCIP 11 reference command additionally requires a matching
SCIP 11 build and PySCIPOpt; see [`docs/scip-baseline.md`](docs/scip-baseline.md).
More commands and result schemas are documented in
[`bench/README.md`](bench/README.md).

## Performance snapshot

Historical five-minute snapshot from July 2026. A newer local 30-second
calibration is documented in [`docs/benchmarks.md`](docs/benchmarks.md); the
budgets and SCIP versions differ, so the figures are not directly comparable.

- The toy suite then reported `12 ok, 2 warn, 0 fail` against OR-Tools at
  `1e-5`. The warnings concerned an alternate optimum and GLOP's
  unbounded-status behavior, not q-opt feasibility failures.
- The recommended five-minute MIPLIB profile produces verified incumbents on
  all 13 checked-in instances and proves 2: `markshare_4_0` and `pk1`.
- The corrected optimized SCIP 11.0.0 / SoPlex 8.0.3 baseline also covers
  13/13 and proves 4: `markshare_4_0`, `pk1`, `neos17`, and `timtab1`.

Representative matched results:

| instance | q-opt, 5 min | SCIP 11, 5 min | current reading |
|----------|--------------|----------------|-----------------|
| `markshare_4_0` | optimal, obj 1 | optimal, obj 1 | q-opt proof retained |
| `pk1` | optimal, obj 11 | optimal, obj 11 | q-opt proof retained |
| `gen-ip002` | obj -4783.73, gap 0.365% | obj -4783.73, gap 0.408% | near parity |
| `gen-ip054` | obj 6840.97, gap about 0.60% | obj 6840.97, gap 0.599% | near parity |
| `assign1-5-8` | obj 212, gap 8.40% | obj 212, gap 4.63% | primal parity, weaker proof |
| `neos5` | obj 15, gap 6.67% | obj 15, gap 1.67% | primal parity, weaker proof |
| `neos-3046615-murg` | obj 1782, gap 80.13% | obj 1628 | primal and bound gap |

The largest remaining gaps are:

- proof strength on `neos17` and `timtab1`;
- primal quality and bounds on `glass4`, `ic97_potential`,
  `neos-3046615-murg`, and `neos-3754480-nidda`;
- cut-row lifecycle and cut-augmented LP cost;
- structural presolve, conflict learning, and richer separators;
- parallel branch-and-bound.

Detailed current priorities and acceptance gates are in
[`docs/solver-milestones.md`](docs/solver-milestones.md). The compact history
of accepted and rejected experiments is in
[`docs/solver-history.md`](docs/solver-history.md). SCIP build and baseline
details are in [`docs/scip-baseline.md`](docs/scip-baseline.md).

## Repository map

```text
model.go                    public model, options, result, and status API
mip.go                      branch-and-bound driver and node evaluation
branching.go                pseudocost and reliability branching
presolve.go                 bound propagation and presolve reductions
heuristic_scheduler.go      shared primal-heuristic budget and telemetry
feasibility_pump.go         mixed-integer feasibility pump
rins.go                     RINS neighborhood search
local_branching.go          incumbent Hamming-ball search
proximity.go                objective-cutoff proximity search
neighborhood_scheduler.go   adaptive neighborhood portfolio
cuts.go / revised_cuts.go   Gomory/MIR and revised-basis cut generation
structure.go                root structural detection and strengthening
canonical.go / node_cache.go
                            canonicalization, scaling, and cross-node reuse
lp/                         incremental revised-simplex engine and LU factors
simplex.go                  reference dense tableau solver
bench/                      shared problems, runners, comparison, telemetry
docs/                       current roadmap and retained experiment evidence
3rd_repo/                   ignored local upstream solver sources, when present
```

## Scope

q-opt currently targets linear LP/MIP only. It does not implement nonlinear
constraints, SOS/indicator modeling parity, a general plugin framework, or
parallel tree search. Numerical safeguards remain intentionally conservative;
an incumbent is never accepted without checking the original model.

## License

Apache-2.0. See [`LICENSE`](LICENSE).
