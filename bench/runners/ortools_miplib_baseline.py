#!/usr/bin/env python3
"""Long-budget OR-Tools (CBC) baseline runner for the MIPLIB sub-suite.

Runs CBC with a long per-instance time budget (default 30 minutes) to
establish a theoretical-upper-bound reference for q-opt:

  - Which instances does CBC prove to optimum within budget?
  - For those, what's the optimum and the wall time?
  - For the rest, what's CBC's incumbent vs best-bound gap when time runs out?

The output JSON keeps the same per-item shape as bench/runners/ortools_bench.py
but adds best_bound, gap_pct, proven_optimal, and is_mip fields. Pair it with
the qopt MIPLIB result JSON via --qopt-results to print a head-to-head
table: instance, CBC optimum, qopt incumbent, qopt gap-to-optimum %.

Examples:
    # Live run, 30 min per instance, 3 concurrent.
    3rd_repo/.venv/bin/python bench/runners/ortools_miplib_baseline.py \\
        --problems bench/problems/miplib \\
        --out bench/results/ortools_miplib_baseline.json \\
        --time-limit-min 30 \\
        --parallel 3

    # Re-print comparison from an existing baseline (no solving):
    3rd_repo/.venv/bin/python bench/runners/ortools_miplib_baseline.py \\
        --baseline-existing bench/results/ortools_miplib_baseline.json \\
        --qopt-results bench/results/qopt_miplib.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from ortools.linear_solver import pywraplp


INFINITY = pywraplp.Solver.infinity()


def _bound(value, default):
    if value is None:
        return default
    return float(value)


def _make_var(solver, spec):
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


def _add_constraint(solver, variables, c):
    sense = c["sense"]
    rhs = float(c["rhs"])
    if sense in ("<=", "le"):
        ct = solver.Constraint(-INFINITY, rhs, c.get("name", ""))
    elif sense in (">=", "ge"):
        ct = solver.Constraint(rhs, INFINITY, c.get("name", ""))
    elif sense in ("=", "==", "eq"):
        ct = solver.Constraint(rhs, rhs, c.get("name", ""))
    else:
        raise ValueError(f"unknown sense {sense!r}")
    for var_name, coeff in c["coeffs"].items():
        if var_name not in variables:
            raise ValueError(f"constraint references unknown var {var_name!r}")
        ct.SetCoefficient(variables[var_name], float(coeff))


def _status_to_str(code):
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


def _is_mip(problem):
    for var in problem["vars"]:
        if var.get("type") in ("integer", "int", "binary", "bool"):
            return True
    return False


def _build_solver(problem, time_limit_ms):
    solver_id = "CBC" if _is_mip(problem) else "GLOP"
    solver = pywraplp.Solver.CreateSolver(solver_id)
    if solver is None:
        raise RuntimeError(f"could not create OR-Tools solver {solver_id!r}")
    solver.SetTimeLimit(time_limit_ms)

    variables = {}
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

    for c in problem.get("constraints", []):
        _add_constraint(solver, variables, c)
    return solver, variables


def _best_bound(solver):
    """Return the proven dual bound from CBC, or None if pywraplp doesn't
    expose it on this build. pywraplp APIs have moved across versions; try
    the two known method names before giving up."""
    objective = solver.Objective()
    for getter in (
        lambda: objective.BestBound(),
        lambda: solver.BestObjectiveBound(),
    ):
        try:
            value = getter()
            if value is None:
                continue
            if math.isnan(value) or math.isinf(value):
                continue
            return float(value)
        except Exception:
            continue
    return None


def _sanitize(v):
    if v is None:
        return None
    if isinstance(v, str):
        return v
    if math.isnan(v) or math.isinf(v):
        return v
    if abs(v) < 1e-12:
        return 0.0
    return v


def _gap_pct(obj, bound):
    """Relative MIP gap, percent. Matches Gurobi's MIPGap convention:
       gap = |obj - bound| / max(1, |obj|) * 100
    Returns None if either value is missing or non-finite."""
    if obj is None or bound is None:
        return None
    if math.isnan(obj) or math.isnan(bound) or math.isinf(obj) or math.isinf(bound):
        return None
    return abs(obj - bound) / max(1.0, abs(obj)) * 100.0


def solve_problem(problem, time_limit_ms):
    is_mip = _is_mip(problem)
    direction = problem.get("direction", "min").lower()
    solver, variables = _build_solver(problem, time_limit_ms)
    t0 = time.perf_counter()
    code = solver.Solve()
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    status = _status_to_str(code)

    item = {
        "name": problem["name"],
        "status": status,
        "is_mip": is_mip,
        "direction": direction,
        "objective": None,
        "best_bound": None,
        "gap_pct": None,
        "proven_optimal": False,
        "iterations": int(solver.iterations()) if hasattr(solver, "iterations") else 0,
        "nodes": int(solver.nodes()) if hasattr(solver, "nodes") else 0,
        "elapsed_ms": elapsed_ms,
    }

    if code in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE):
        obj = _sanitize(solver.Objective().Value())
        bound = _sanitize(_best_bound(solver))
        item["objective"] = obj
        item["best_bound"] = bound
        item["gap_pct"] = _gap_pct(obj, bound)
        item["proven_optimal"] = (code == pywraplp.Solver.OPTIMAL)
        if code == pywraplp.Solver.FEASIBLE:
            item["error"] = "ortools_returned_feasible_not_optimal"
    return item


def _solve_one_for_pool(args):
    problem, time_limit_ms = args
    try:
        return solve_problem(problem, time_limit_ms)
    except Exception as exc:
        return {
            "name": problem.get("name", "<unknown>"),
            "status": "error",
            "elapsed_ms": 0.0,
            "error": f"{type(exc).__name__}: {exc}",
        }


def _load_problems(directory):
    paths = sorted(p for p in directory.iterdir() if p.suffix == ".json")
    return [json.loads(p.read_text()) for p in paths]


def _ortools_version():
    try:
        from importlib.metadata import version
        return version("ortools")
    except Exception:
        return "unknown"


def _fmt_obj(v):
    if v is None:
        return "—"
    if isinstance(v, float):
        if math.isnan(v):
            return "nan"
        if math.isinf(v):
            return "inf" if v > 0 else "-inf"
        return f"{v:.6g}"
    return str(v)


def _print_row(item):
    elapsed_min = item.get("elapsed_ms", 0) / 60_000.0
    proven = "✓" if item.get("proven_optimal") else " "
    gap = item.get("gap_pct")
    gap_s = f"{gap:.2f}%" if gap is not None else "—"
    print(
        f"{item['name']:<32} {proven} {item.get('status','?'):<10} "
        f"obj={_fmt_obj(item.get('objective')):<14} "
        f"bound={_fmt_obj(item.get('best_bound')):<14} "
        f"gap={gap_s:<9} "
        f"nodes={item.get('nodes', 0):<8} "
        f"time={elapsed_min:6.2f}min",
        flush=True,
    )


def _print_summary(items):
    n = len(items)
    proven = [it for it in items if it.get("proven_optimal")]
    feasible_only = [
        it for it in items
        if it.get("status") == "feasible" and not it.get("proven_optimal")
    ]
    print()
    print(f"=== summary: {n} instances ===")
    print(f"  proven optimal in budget:   {len(proven)}")
    print(f"  feasible (time-limited):    {len(feasible_only)}")
    print(f"  other (infeasible / error): {n - len(proven) - len(feasible_only)}")
    if proven:
        times = sorted(it["elapsed_ms"] / 60_000.0 for it in proven)
        print(
            f"  proven-optimum wall time:   "
            f"min={times[0]:.2f}min  "
            f"median={times[len(times)//2]:.2f}min  "
            f"max={times[-1]:.2f}min"
        )
    if feasible_only:
        gaps = [it["gap_pct"] for it in feasible_only if it.get("gap_pct") is not None]
        if gaps:
            gaps_sorted = sorted(gaps)
            print(
                f"  CBC time-limited gap:       "
                f"min={gaps_sorted[0]:.2f}%  "
                f"median={gaps_sorted[len(gaps_sorted)//2]:.2f}%  "
                f"max={gaps_sorted[-1]:.2f}%"
            )


def _compare_with_qopt(items, qopt_path):
    qopt = json.loads(Path(qopt_path).read_text())
    by_name = {it["name"]: it for it in qopt.get("items", [])}
    print()
    print(f"=== q-opt vs CBC baseline ({qopt_path}) ===")
    header = (
        f"{'instance':<32} "
        f"{'cbc-status':<8} "
        f"{'opt(CBC)':<14} "
        f"{'cbc-bound':<14} "
        f"{'qopt-status':<14} "
        f"{'qopt-obj':<14} "
        f"{'qopt-gap-to-opt':<16} "
        f"{'qopt-gap-to-bound':<18} "
        f"{'qopt-time(s)':<12}"
    )
    print(header)
    for it in items:
        name = it["name"]
        qo = by_name.get(name)
        cbc_status = (
            "opt" if it.get("proven_optimal")
            else it.get("status", "?")[:8]
        )
        opt = it.get("objective") if it.get("proven_optimal") else None
        bound = it.get("best_bound")
        opt_s = _fmt_obj(opt)
        bound_s = _fmt_obj(bound)
        if qo is None:
            print(f"{name:<32} {cbc_status:<8} {opt_s:<14} {bound_s:<14} (no qopt row)")
            continue
        qopt_status = qo.get("status", "?")
        qopt_obj = qo.get("objective")
        qopt_obj_s = _fmt_obj(qopt_obj)
        # gap-to-opt: only when CBC proved optimum
        gap_opt_s = "—"
        if opt is not None and qopt_obj is not None and not (
            isinstance(qopt_obj, float) and (math.isnan(qopt_obj) or math.isinf(qopt_obj))
        ):
            gap_opt = abs(qopt_obj - opt) / max(1.0, abs(opt)) * 100.0
            gap_opt_s = f"{gap_opt:.2f}%"
        # gap-to-bound: against CBC's best bound (more useful when neither side
        # proved optimum but CBC has a tighter dual side)
        gap_bound_s = "—"
        if bound is not None and qopt_obj is not None and not (
            isinstance(qopt_obj, float) and (math.isnan(qopt_obj) or math.isinf(qopt_obj))
        ):
            gap_bound = abs(qopt_obj - bound) / max(1.0, abs(qopt_obj)) * 100.0
            gap_bound_s = f"{gap_bound:.2f}%"
        qopt_time_s = qo.get("elapsedMs", qo.get("elapsed_ms", 0)) / 1000.0
        print(
            f"{name:<32} {cbc_status:<8} {opt_s:<14} {bound_s:<14} "
            f"{qopt_status:<14} {qopt_obj_s:<14} "
            f"{gap_opt_s:<16} {gap_bound_s:<18} {qopt_time_s:<12.2f}"
        )


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--problems", default="bench/problems/miplib")
    parser.add_argument("--out", default="bench/results/ortools_miplib_baseline.json")
    parser.add_argument(
        "--time-limit-min", type=float, default=30.0,
        help="per-problem time limit in minutes (default 30)",
    )
    parser.add_argument(
        "--parallel", type=int, default=3,
        help="number of problems to solve concurrently (default 3)",
    )
    parser.add_argument(
        "--qopt-results", default=None,
        help="path to a q-opt MIPLIB result JSON; if given, print head-to-head comparison",
    )
    parser.add_argument(
        "--baseline-existing", default=None,
        help="skip the solve and load an existing baseline JSON for comparison only",
    )
    args = parser.parse_args()

    if args.baseline_existing:
        loaded = json.loads(Path(args.baseline_existing).read_text())
        items = loaded["items"]
        print(f"loaded {len(items)} items from {args.baseline_existing}")
        _print_summary(items)
        if args.qopt_results:
            _compare_with_qopt(items, args.qopt_results)
        return 0

    problems_dir = Path(args.problems)
    if not problems_dir.is_dir():
        print(f"problems dir {problems_dir!s} does not exist", file=sys.stderr)
        return 2

    problems = _load_problems(problems_dir)
    time_limit_ms = int(args.time_limit_min * 60_000.0)
    parallel = max(1, args.parallel)

    print(f"running OR-Tools CBC on {len(problems)} instances")
    print(f"  problems dir: {problems_dir}")
    print(f"  time limit:   {args.time_limit_min:.1f} min/instance")
    print(f"  parallel:     {parallel}")
    print(f"  ortools:      {_ortools_version()}")
    print()
    print(
        f"{'instance':<32}   {'status':<10} {'objective':<18} "
        f"{'bound':<18} {'gap':<10} {'nodes':<9} {'time':>10}"
    )
    print("-" * 120)

    items_by_name = {}
    start = time.perf_counter()
    if parallel == 1:
        for p in problems:
            item = _solve_one_for_pool((p, time_limit_ms))
            items_by_name[item["name"]] = item
            _print_row(item)
    else:
        with ProcessPoolExecutor(max_workers=parallel) as ex:
            futures = {
                ex.submit(_solve_one_for_pool, (p, time_limit_ms)): p["name"]
                for p in problems
            }
            for fut in as_completed(futures):
                item = fut.result()
                items_by_name[item["name"]] = item
                _print_row(item)

    # Restore problem ordering for the output JSON.
    items = [items_by_name[p["name"]] for p in problems]
    total_min = (time.perf_counter() - start) / 60.0
    print()
    print(f"total wall time: {total_min:.1f} minutes")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "solver": "ortools",
        "version": _ortools_version(),
        "time_limit_min": args.time_limit_min,
        "items": items,
    }
    out_path.write_text(json.dumps(payload, indent=2, default=str) + "\n")
    print(f"wrote {out_path} ({len(items)} items)")

    _print_summary(items)
    if args.qopt_results:
        _compare_with_qopt(items, args.qopt_results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
