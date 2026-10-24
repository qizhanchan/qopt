#!/usr/bin/env python3
"""OR-Tools reference runner for the q-opt benchmark suite.

Reads every ``bench/problems/*.json`` file, builds the model with
``ortools.linear_solver.pywraplp``, and writes a result file that matches the
schema in ``bench/README.md``. LP problems are sent to GLOP, MIP problems to
CBC. The output is consumed by ``bench/cmd/compare``.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

from ortools.linear_solver import pywraplp


INFINITY = pywraplp.Solver.infinity()


def _bound(value: float | None, default: float) -> float:
    if value is None:
        return default
    return float(value)


def _make_var(solver: pywraplp.Solver, spec: dict[str, Any]) -> pywraplp.Variable:
    name = spec["name"]
    vtype = spec.get("type", "continuous")
    if vtype in ("binary", "bool"):
        if spec.get("lower_unbounded"):
            raise ValueError(f"var {name!r}: binary variable cannot be lower-unbounded")
        return solver.BoolVar(name)
    if spec.get("lower_unbounded") and spec.get("lower") is not None:
        raise ValueError(f"var {name!r}: lower and lower_unbounded are mutually exclusive")
    lower = -INFINITY if spec.get("lower_unbounded") else _bound(spec.get("lower"), 0.0)
    upper = _bound(spec.get("upper"), INFINITY)
    if vtype in ("integer", "int"):
        return solver.IntVar(lower, upper, name)
    if vtype in ("", "continuous"):
        return solver.NumVar(lower, upper, name)
    raise ValueError(f"var {name!r}: unknown type {vtype!r}")


def _add_constraint(solver: pywraplp.Solver, variables: dict[str, pywraplp.Variable],
                    constraint: dict[str, Any]) -> None:
    sense = constraint["sense"]
    rhs = float(constraint["rhs"])
    if sense in ("<=", "le"):
        ct = solver.Constraint(-INFINITY, rhs, constraint.get("name", ""))
    elif sense in (">=", "ge"):
        ct = solver.Constraint(rhs, INFINITY, constraint.get("name", ""))
    elif sense in ("=", "==", "eq"):
        ct = solver.Constraint(rhs, rhs, constraint.get("name", ""))
    else:
        raise ValueError(f"unknown sense {sense!r}")
    for var_name, coeff in constraint["coeffs"].items():
        if var_name not in variables:
            raise ValueError(f"constraint references unknown var {var_name!r}")
        ct.SetCoefficient(variables[var_name], float(coeff))


def _status_to_str(code: int) -> str:
    if code == pywraplp.Solver.OPTIMAL:
        return "optimal"
    if code == pywraplp.Solver.FEASIBLE:
        return "feasible"
    if code == pywraplp.Solver.INFEASIBLE:
        return "infeasible"
    if code == pywraplp.Solver.UNBOUNDED:
        return "unbounded"
    if code == pywraplp.Solver.ABNORMAL:
        return "abnormal"
    if code == pywraplp.Solver.NOT_SOLVED:
        return "not_solved"
    return f"status_{code}"


def _solver_id(problem: dict[str, Any]) -> str:
    for var in problem["vars"]:
        if var.get("type") in ("integer", "int", "binary", "bool"):
            return "CBC"
    return "GLOP"


def _build_solver(problem: dict[str, Any], time_limit_ms: int
                  ) -> tuple[pywraplp.Solver, dict[str, pywraplp.Variable]]:
    solver_id = _solver_id(problem)
    solver = pywraplp.Solver.CreateSolver(solver_id)
    if solver is None:
        raise RuntimeError(f"could not create OR-Tools solver {solver_id!r}")
    solver.SetTimeLimit(time_limit_ms)

    variables: dict[str, pywraplp.Variable] = {}
    for spec in problem["vars"]:
        variables[spec["name"]] = _make_var(solver, spec)

    objective = solver.Objective()
    for name, coeff in problem.get("objective", {}).items():
        if name not in variables:
            raise ValueError(f"objective references unknown var {name!r}")
        objective.SetCoefficient(variables[name], float(coeff))

    direction = problem.get("direction", "min").lower()
    if direction in ("min", "minimize"):
        objective.SetMinimization()
    elif direction in ("max", "maximize"):
        objective.SetMaximization()
    else:
        raise ValueError(f"unknown direction {direction!r}")

    for constraint in problem.get("constraints", []):
        _add_constraint(solver, variables, constraint)
    return solver, variables


def solve_problem(problem: dict[str, Any], time_limit_ms: int,
                  warmup: int = 0, repeat: int = 1) -> dict[str, Any]:
    """Solve `problem` `repeat` times and return the run with median wall time.

    A fresh solver instance is built each iteration. Timing covers Solve(),
    after model construction, matching the q-opt runner.
    """
    for _ in range(max(0, warmup)):
        solver, _vars = _build_solver(problem, time_limit_ms)
        solver.Solve()

    samples: list[tuple[float, int, pywraplp.Solver, dict[str, pywraplp.Variable]]] = []
    for _ in range(max(1, repeat)):
        solver, variables = _build_solver(problem, time_limit_ms)
        start = time.perf_counter()
        code = solver.Solve()
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        samples.append((elapsed_ms, code, solver, variables))

    samples.sort(key=lambda s: s[0])
    elapsed_ms, code, solver, variables = samples[len(samples) // 2]

    status = _status_to_str(code)
    item: dict[str, Any] = {
        "name": problem["name"],
        "status": status,
        "objective": 0.0,
        "iterations": int(solver.iterations()) if hasattr(solver, "iterations") else 0,
        "nodes": int(solver.nodes()) if _solver_id(problem) == "CBC" else 0,
        "elapsed_ms": elapsed_ms,
    }
    if code in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE):
        item["objective"] = _sanitize(solver.Objective().Value())
        item["values"] = {
            name: _sanitize(var.solution_value()) for name, var in variables.items()
        }
        if code == pywraplp.Solver.FEASIBLE:
            # OR-Tools can return FEASIBLE when the time limit fires.
            item["error"] = "ortools_returned_feasible_not_optimal"
    return item


def _sanitize(v: float) -> float:
    if math.isnan(v) or math.isinf(v):
        return v
    if abs(v) < 1e-12:
        return 0.0
    return v


def _load_problems(directory: Path) -> list[dict[str, Any]]:
    paths = sorted(p for p in directory.iterdir() if p.suffix == ".json")
    return [json.loads(p.read_text()) for p in paths]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--problems", default="bench/problems",
                        help="directory containing problem JSON files")
    parser.add_argument("--out", default="bench/results/ortools.json",
                        help="output file for results")
    parser.add_argument("--time-limit-ms", type=int, default=30_000,
                        help="per-problem time limit (milliseconds)")
    parser.add_argument("--repeat", type=int, default=1,
                        help="solve each problem N times; record median wall time")
    parser.add_argument("--warmup", type=int, default=0,
                        help="warmup runs per problem (not counted in timing)")
    args = parser.parse_args()

    problems_dir = Path(args.problems)
    if not problems_dir.is_dir():
        print(f"problems dir {problems_dir!s} does not exist", file=sys.stderr)
        return 2

    items: list[dict[str, Any]] = []
    for problem in _load_problems(problems_dir):
        try:
            item = solve_problem(problem, args.time_limit_ms,
                                  warmup=args.warmup, repeat=args.repeat)
        except Exception as exc:  # noqa: BLE001 — surface as result row
            item = {
                "name": problem.get("name", "<unknown>"),
                "status": "error",
                "objective": 0.0,
                "elapsed_ms": 0.0,
                "error": f"{type(exc).__name__}: {exc}",
            }
        items.append(item)
        print(
            f"{item['name']:<32} {item['status']:<12} "
            f"obj={item['objective']:<14g} "
            f"iter={item.get('iterations', 0):<6} "
            f"nodes={item.get('nodes', 0):<4} "
            f"median={item['elapsed_ms']:.3f}ms (n={args.repeat})"
        )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "solver": "ortools",
        "version": _ortools_version(),
        "items": items,
    }, indent=2) + "\n")
    print(f"wrote {out_path} ({len(items)} items)")
    return 0


def _ortools_version() -> str:
    try:
        from importlib.metadata import version
        return version("ortools")
    except Exception:  # noqa: BLE001
        return "unknown"


if __name__ == "__main__":
    raise SystemExit(main())
