# q-opt Benchmark Harness

Cross-language correctness and performance benchmark for q-opt against
OR-Tools, HiGHS, and SCIP. Each solver consumes the checked-in JSON problems.
The comparison tool checks status, objective, and original-model incumbent
feasibility at a configurable tolerance (default `1e-5`).

## Layout

- `problems/*.json` — shared problem set (LP and MIP).
- `cmd/qopt_bench/` — Go runner using `github.com/qizhanchan/qopt`.
- `runners/ortools_bench.py` — Python runner using `ortools.linear_solver`.
- `cmd/compare/` — Go tool that diffs two result files.
- `runners/highs_miplib_baseline.py` — HiGHS MIPLIB runner.
- `runners/scip_miplib_baseline.py` — SCIP MIPLIB runner using JSON model construction.
- `results/` — mostly ignored runner output; a few historical experiment records are retained.

## Problem schema (`problems/*.json`)

```json
{
  "name": "lp_simple_max",
  "direction": "max",                 // "min" or "max"
  "vars": [
    {"name": "x", "type": "continuous", "lower": 0, "upper": null},
    {"name": "z", "type": "continuous", "lower_unbounded": true, "upper": null},
    {"name": "y", "type": "binary"}
  ],
  "objective": {"x": 3.0, "y": 2.0},
  "constraints": [
    {"name": "c1", "coeffs": {"x": 1, "y": 1}, "sense": "<=", "rhs": 4}
  ],
  "expect": {                          // optional — known answer for sanity
    "status": "optimal",
    "objective": 10.0,
    "values": {"x": 2, "y": 2}
  }
}
```

`type` is one of `continuous`, `integer`, `binary`. `sense` is one of `<=`,
`>=`, `=`. `upper` may be `null` for `+∞`; `lower` defaults to `0` per q-opt's
convention. Set `lower_unbounded: true` to represent a `-∞` lower bound; it is
mutually exclusive with a numeric `lower` and invalid for binary variables.
The `expect` block is only used for documentation / quick sanity checks. Toy
comparison uses OR-Tools as the reference, with original-model feasibility
checked independently.

## Result schema (`results/*.json`)

```json
{
  "solver": "qopt | ortools",
  "version": "...",
  "items": [
    {
      "name": "lp_simple_max",
      "status": "optimal | infeasible | unbounded | other",
      "objective": 10.0,
      "values": {"x": 2.0, "y": 2.0},
      "iterations": 7,
      "nodes": 1,
      "cut_rounds": 1,
      "cuts": 12,
      "cut_ms": 173.4,
      "elapsed_ms": 0.42
    }
  ]
}
```

## How to run

```sh
# Create an isolated Python environment for comparison solvers.
python3 -m venv .venv
.venv/bin/python -m pip install ortools highspy pyscipopt

# q-opt results
go run ./bench/cmd/qopt_bench \
    -problems bench/problems -out bench/results/qopt.json \
    -repeat 21 -warmup 2 -parallel 3

# OR-Tools reference results
.venv/bin/python bench/runners/ortools_bench.py \
    --problems bench/problems --out bench/results/ortools.json \
    --repeat 21 --warmup 2

# Compare (tolerance defaults to 1e-5)
go run ./bench/cmd/compare \
    -baseline bench/results/ortools.json \
    -candidate bench/results/qopt.json \
    -problems bench/problems
```

## MIPLIB 2017 sub-suite

A subset of small "easy" MIPLIB 2017 benchmark instances lives under
`bench/problems/miplib/`. The `.mps.gz` originals are in
`3rd_repo/miplib/` (git-ignored) and were converted with
`bench/runners/mps_to_json.py`:

```sh
.venv/bin/python bench/runners/mps_to_json.py \
    3rd_repo/miplib/*.mps.gz --out-dir bench/problems/miplib
```

To benchmark the MIPLIB sub-suite (heavy — set a wall budget):

```sh
go run ./bench/cmd/qopt_bench \
    -problems bench/problems/miplib -out bench/results/qopt_miplib.json \
    -time-limit 30s -repeat 1 -parallel 3

.venv/bin/python bench/runners/ortools_bench.py \
    --problems bench/problems/miplib --out bench/results/ortools_miplib.json \
    --time-limit-ms 30000 --repeat 1

go run ./bench/cmd/compare \
    -baseline bench/results/ortools_miplib.json \
    -candidate bench/results/qopt_miplib.json \
    -problems bench/problems/miplib
```

When `-cut-rounds` is enabled, q-opt's runner applies a separate root-cut
budget via `-cut-time-limit` (default `5s`) so cut generation cannot consume
the full per-problem time limit before branch-and-bound starts. Set
`-cut-time-limit 0` to restore the old unbounded root-cut behavior.
Use `-rounding` to enable the experimental LP-rounding incumbent heuristic;
it is off by default because it can improve some time-limited incumbents while
hurting others.

Status is `optimal` (proven), `feasible` (incumbent without proof — OR-Tools
when the time limit fires), `time_limit` (q-opt when budget elapsed), or
`infeasible`. The comparison tool downgrades `optimal vs time_limit` to a
"warn" — that pair is a slowness diagnosis, not a correctness fault.
