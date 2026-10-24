#!/usr/bin/env python3
"""HiGHS baseline runner for the MIPLIB sub-suite.

Mirrors ortools_miplib_baseline.py's output schema (best_bound, gap_pct,
proven_optimal, nodes, elapsed_ms) so results are directly comparable with
the existing CBC baseline and q-opt result JSONs via the same
--qopt-results / --baseline-existing comparison flags.

Uses highspy (HiGHS's official Python bindings) — install with:
    pip install highspy

Examples:
    python bench/runners/highs_miplib_baseline.py \\
        --problems bench/problems/miplib \\
        --out bench/results/highs_miplib_baseline.json \\
        --time-limit-min 5 \\
        --parallel 3
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

import highspy


def _bound(value, default):
    if value is None:
        return default
    return float(value)


def _is_mip(problem: dict[str, Any]) -> bool:
    for var in problem["vars"]:
        if var.get("type") in ("integer", "int", "binary", "bool"):
            return True
    return False


def _build_and_solve(problem: dict[str, Any], time_limit_s: float):
    h = highspy.Highs()
    h.setOptionValue("output_flag", False)
    h.setOptionValue("time_limit", time_limit_s)

    variables: dict[str, Any] = {}
    for spec in problem["vars"]:
        name = spec["name"]
        vtype = spec.get("type", "continuous")
        obj = float(problem.get("objective", {}).get(name, 0.0))
        if vtype in ("binary", "bool"):
            if spec.get("lower_unbounded"):
                raise ValueError(f"var {name!r}: binary variable cannot be lower-unbounded")
            lb, ub = 0.0, 1.0
            kind = highspy.HighsVarType.kInteger
        else:
            if spec.get("lower_unbounded") and spec.get("lower") is not None:
                raise ValueError(f"var {name!r}: lower and lower_unbounded are mutually exclusive")
            lb = -highspy.kHighsInf if spec.get("lower_unbounded") else _bound(spec.get("lower"), 0.0)
            ub = _bound(spec.get("upper"), highspy.kHighsInf)
            kind = (
                highspy.HighsVarType.kInteger
                if vtype in ("integer", "int")
                else highspy.HighsVarType.kContinuous
            )
        variables[name] = h.addVariable(lb, ub, obj=obj, type=kind)

    for c in problem.get("constraints", []):
        expr = sum(
            (float(coeff) * variables[name] for name, coeff in c["coeffs"].items()),
            start=0,
        )
        sense = c["sense"]
        rhs = float(c["rhs"])
        if sense in ("<=", "le"):
            h.addConstr(expr <= rhs, name=c.get("name") or None)
        elif sense in (">=", "ge"):
            h.addConstr(expr >= rhs, name=c.get("name") or None)
        elif sense in ("=", "==", "eq"):
            h.addConstr(expr == rhs, name=c.get("name") or None)
        else:
            raise ValueError(f"unknown sense {sense!r}")

    direction = problem.get("direction", "min").lower()
    if direction in ("min", "minimize"):
        h.minimize()
    elif direction in ("max", "maximize"):
        h.maximize()
    else:
        raise ValueError(f"unknown direction {direction!r}")

    return h, variables


_STATUS_MAP = {
    "kOptimal": "optimal",
    "kInfeasible": "infeasible",
    "kUnbounded": "unbounded",
    "kUnboundedOrInfeasible": "unbounded_or_infeasible",
    "kTimeLimit": "time_limit",
    "kIterationLimit": "iteration_limit",
    "kSolutionLimit": "solution_limit",
    "kMemoryLimit": "memory_limit",
    "kObjectiveBound": "objective_bound",
    "kObjectiveTarget": "objective_target",
    "kInterrupt": "interrupt",
    "kModelEmpty": "model_empty",
    "kNotset": "not_set",
    "kUnknown": "unknown",
}


def _status_to_str(model_status) -> str:
    name = str(model_status).rsplit(".", 1)[-1]
    return _STATUS_MAP.get(name, name)


def _sanitize(v):
    if v is None:
        return None
    if math.isnan(v) or math.isinf(v):
        return v
    if abs(v) < 1e-12:
        return 0.0
    return v


def _gap_pct(obj, bound):
    if obj is None or bound is None:
        return None
    if math.isnan(obj) or math.isnan(bound) or math.isinf(obj) or math.isinf(bound):
        return None
    return abs(obj - bound) / max(1.0, abs(obj)) * 100.0


def solve_problem(problem: dict[str, Any], time_limit_s: float) -> dict[str, Any]:
    is_mip = _is_mip(problem)
    direction = problem.get("direction", "min").lower()

    t0 = time.perf_counter()
    h, variables = _build_and_solve(problem, time_limit_s)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    model_status = h.getModelStatus()
    status = _status_to_str(model_status)
    info = h.getInfo()

    item: dict[str, Any] = {
        "name": problem["name"],
        "status": status,
        "is_mip": is_mip,
        "direction": direction,
        "objective": None,
        "best_bound": None,
        "gap_pct": None,
        "proven_optimal": status == "optimal",
        "iterations": int(info.simplex_iteration_count),
        "nodes": int(info.mip_node_count) if is_mip else 0,
        "elapsed_ms": elapsed_ms,
    }

    sol = h.getSolution()
    if sol.value_valid:
        obj = _sanitize(info.objective_function_value)
        bound = _sanitize(info.mip_dual_bound) if is_mip else obj
        item["objective"] = obj
        item["best_bound"] = bound
        item["gap_pct"] = _gap_pct(obj, bound)
        item["values"] = {name: _sanitize(v) for name, v in zip(variables.keys(), sol.col_value)}
        if status != "optimal":
            item["error"] = f"highs_returned_{status}_not_optimal"

    return item


def _solve_one_for_pool(args):
    problem, time_limit_s = args
    try:
        return solve_problem(problem, time_limit_s)
    except Exception as exc:  # noqa: BLE001
        return {
            "name": problem.get("name", "<unknown>"),
            "status": "error",
            "elapsed_ms": 0.0,
            "error": f"{type(exc).__name__}: {exc}",
        }


def _load_problems(directory: Path) -> list[dict[str, Any]]:
    paths = sorted(p for p in directory.iterdir() if p.suffix == ".json")
    return [json.loads(p.read_text()) for p in paths]


def _highs_version() -> str:
    try:
        from importlib.metadata import version
        return version("highspy")
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
        if it.get("status") not in ("optimal", "infeasible", "unbounded", "error")
        and it.get("objective") is not None
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
                f"  time-limited gap:           "
                f"min={gaps_sorted[0]:.2f}%  "
                f"median={gaps_sorted[len(gaps_sorted)//2]:.2f}%  "
                f"max={gaps_sorted[-1]:.2f}%"
            )


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--problems", default="bench/problems/miplib")
    parser.add_argument("--out", default="bench/results/highs_miplib_baseline.json")
    parser.add_argument("--time-limit-min", type=float, default=5.0)
    parser.add_argument("--parallel", type=int, default=3)
    parser.add_argument("--qopt-results", default=None)
    parser.add_argument("--baseline-existing", default=None)
    args = parser.parse_args()

    if args.baseline_existing:
        loaded = json.loads(Path(args.baseline_existing).read_text())
        items = loaded["items"]
        print(f"loaded {len(items)} items from {args.baseline_existing}")
        _print_summary(items)
        return 0

    problems_dir = Path(args.problems)
    if not problems_dir.is_dir():
        print(f"problems dir {problems_dir!s} does not exist", file=sys.stderr)
        return 2

    problems = _load_problems(problems_dir)
    time_limit_s = args.time_limit_min * 60.0
    parallel = max(1, args.parallel)

    print(f"running HiGHS on {len(problems)} instances")
    print(f"  problems dir: {problems_dir}")
    print(f"  time limit:   {args.time_limit_min:.1f} min/instance")
    print(f"  parallel:     {parallel}")
    print(f"  highspy:      {_highs_version()}")
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
            item = _solve_one_for_pool((p, time_limit_s))
            items_by_name[item["name"]] = item
            _print_row(item)
    else:
        with ProcessPoolExecutor(max_workers=parallel) as ex:
            futures = {
                ex.submit(_solve_one_for_pool, (p, time_limit_s)): p["name"]
                for p in problems
            }
            for fut in as_completed(futures):
                item = fut.result()
                items_by_name[item["name"]] = item
                _print_row(item)

    items = [items_by_name[p["name"]] for p in problems]
    total_min = (time.perf_counter() - start) / 60.0
    print()
    print(f"total wall time: {total_min:.1f} minutes")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "solver": "highs",
        "version": _highs_version(),
        "time_limit_min": args.time_limit_min,
        "items": items,
    }
    out_path.write_text(json.dumps(payload, indent=2, default=str) + "\n")
    print(f"wrote {out_path} ({len(items)} items)")

    _print_summary(items)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
