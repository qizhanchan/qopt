#!/usr/bin/env python3
"""Run the MIPLIB sub-suite against SCIP.

The default ``json-api`` mode rebuilds each model from the same JSON consumed
by q-opt. This preserves variable and constraint insertion order, which can
materially change SCIP's presolve and search trajectory even when an exported
MPS is mathematically equivalent. ``native-mps`` remains available for
white-box diagnostics against the original MIPLIB files.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from scip_diagnostics import (
    ROOT,
    _build_driver,
    _normalize_status,
    _rank_plugins,
    _sum_fields,
)


DEFAULT_SCIP_BUILD = Path("/tmp/qopt-scip11-vanilla-build")
DEFAULT_DRIVER_BUILD = Path("/tmp/qopt-scip11-vanilla-driver-build")


def _sanitize(value: Any) -> Any:
    if value is None:
        return None
    value = float(value)
    if math.isfinite(value) and abs(value) < 1e-12:
        return 0.0
    return value


def _gap_pct(objective: float | None, bound: float | None) -> float | None:
    if objective is None or bound is None:
        return None
    if not math.isfinite(objective) or not math.isfinite(bound):
        return None
    return abs(objective - bound) / max(1.0, abs(objective)) * 100.0


def _is_mip(problem: dict[str, Any]) -> bool:
    return any(
        variable.get("type") in ("integer", "int", "binary", "bool")
        for variable in problem["vars"]
    )


def _bound(value: Any, default: float) -> float:
    return default if value is None else float(value)


def _lower_bound(spec: dict[str, Any], infinity: float) -> float:
    if spec.get("lower_unbounded"):
        if spec.get("lower") is not None:
            raise ValueError(
                f"var {spec.get('name')!r}: lower and lower_unbounded are mutually exclusive"
            )
        return -infinity
    return _bound(spec.get("lower"), 0.0)


def _apply_parameters(model: Any, parameters: list[str]) -> None:
    for assignment in parameters:
        if "=" not in assignment:
            raise ValueError(f"expected NAME=VALUE parameter assignment: {assignment}")
        name, value = assignment.split("=", 1)
        model.setParam(name, value)


def _build_json_model(
    problem: dict[str, Any],
    time_limit_s: float,
    parameters: list[str],
    dynamic_constraints: bool = False,
):
    from pyscipopt import Model

    model = Model()
    model.hideOutput()
    model.setParam("limits/time", time_limit_s)
    _apply_parameters(model, parameters)

    variables = {}
    for spec in problem["vars"]:
        name = spec["name"]
        var_type = spec.get("type", "continuous")
        if var_type in ("binary", "bool"):
            if spec.get("lower_unbounded"):
                raise ValueError(f"var {name!r}: binary variable cannot be lower-unbounded")
            variable = model.addVar(name, vtype="B", lb=0, ub=1)
        elif var_type in ("integer", "int"):
            variable = model.addVar(
                name,
                vtype="I",
                lb=_lower_bound(spec, model.infinity()),
                ub=_bound(spec.get("upper"), model.infinity()),
            )
        else:
            variable = model.addVar(
                name,
                vtype="C",
                lb=_lower_bound(spec, model.infinity()),
                ub=_bound(spec.get("upper"), model.infinity()),
            )
        variables[name] = variable

    direction = problem.get("direction", "min").lower()
    objective = sum(
        (
            float(coefficient) * variables[name]
            for name, coefficient in problem.get("objective", {}).items()
        ),
        start=0,
    )
    model.setObjective(
        objective,
        sense="minimize" if direction in ("min", "minimize") else "maximize",
    )

    for constraint in problem.get("constraints", []):
        expression = sum(
            float(coefficient) * variables[name]
            for name, coefficient in constraint["coeffs"].items()
        )
        sense = constraint["sense"]
        rhs = float(constraint["rhs"])
        name = constraint.get("name") or ""
        if sense in ("<=", "le"):
            model.addCons(
                expression <= rhs, name=name, dynamic=dynamic_constraints
            )
        elif sense in (">=", "ge"):
            model.addCons(
                expression >= rhs, name=name, dynamic=dynamic_constraints
            )
        elif sense in ("=", "==", "eq"):
            model.addCons(
                expression == rhs, name=name, dynamic=dynamic_constraints
            )
        else:
            raise ValueError(f"unknown constraint sense {sense!r}")
    return model, variables


def _reorder_json_variables_from_native(
    problem: dict[str, Any], mps_path: Path
) -> dict[str, Any]:
    from pyscipopt import Model

    native = Model()
    native.hideOutput()
    native.readProblem(str(mps_path))
    order = {variable.name: index for index, variable in enumerate(native.getVars())}
    native.freeProb()
    reordered = dict(problem)
    reordered["vars"] = sorted(
        problem["vars"], key=lambda spec: order.get(spec["name"], len(order))
    )
    return reordered


def _finite_or_none(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) and abs(result) < 1e19 else None


def _lp_snapshot(model: Any) -> dict[str, Any]:
    snapshot: dict[str, Any] = {
        "objective": _finite_or_none(model.getLPObjVal()),
        "lp_iterations": int(model.getNLPIterations()),
        "lp_solves": int(model.getNLPs()),
        "separation_rounds": int(model.getNSepaRounds()),
        "cuts_applied": int(model.getNCutsApplied()),
    }
    try:
        _, values, fractions, candidates, priority_candidates, implicit = (
            model.getLPBranchCands()
        )
        active_fractions = [abs(float(value)) for value in fractions[:candidates]]
        snapshot["fractionality"] = {
            "candidates": int(candidates),
            "priority_candidates": int(priority_candidates),
            "implicit_candidates": int(implicit),
            "sum": sum(active_fractions),
            "max": max(active_fractions, default=0.0),
            "mean": (
                sum(active_fractions) / len(active_fractions)
                if active_fractions
                else 0.0
            ),
            "values_sample": [float(value) for value in values[:8]],
        }
    except Exception:  # noqa: BLE001
        snapshot["fractionality"] = None

    try:
        columns = model.getLPColsData()
        rows = model.getLPRowsData()
        basic_columns = [
            column for column in columns if column.getBasisStatus() == "basic"
        ]
        basic_at_bound = sum(
            min(
                abs(float(column.getPrimsol()) - float(column.getLb())),
                abs(float(column.getPrimsol()) - float(column.getUb())),
            )
            <= 1e-7
            for column in basic_columns
        )
        snapshot["lp_size"] = {
            "columns": len(columns),
            "rows": len(rows),
            "nonzeros": sum(int(row.getNLPNonz()) for row in rows),
        }
        snapshot["degeneracy"] = {
            "basic_columns": len(basic_columns),
            "basic_columns_at_bound": basic_at_bound,
            "basic_at_bound_ratio": (
                basic_at_bound / len(basic_columns) if basic_columns else 0.0
            ),
        }
        cut_rows = [row for row in rows if row.isRemovable()]
        densities = [
            int(row.getNLPNonz()) / max(1, len(columns)) for row in cut_rows
        ]
        efficacies = []
        for row in cut_rows:
            try:
                efficacies.append(float(model.getCutEfficacy(row)))
            except Exception:  # noqa: BLE001
                continue
        ages = [int(row.getAge()) for row in cut_rows]
        snapshot["active_cuts"] = {
            "rows": len(cut_rows),
            "density_mean": sum(densities) / len(densities) if densities else 0.0,
            "density_max": max(densities, default=0.0),
            "efficacy_mean": (
                sum(efficacies) / len(efficacies) if efficacies else None
            ),
            "efficacy_max": max(efficacies, default=None),
            "age_mean": sum(ages) / len(ages) if ages else 0.0,
            "age_max": max(ages, default=0),
        }
    except Exception:  # noqa: BLE001
        snapshot["lp_size"] = None
        snapshot["degeneracy"] = None
        snapshot["active_cuts"] = None
    return snapshot


def _attach_trace(model: Any, sample_interval_s: float) -> None:
    from pyscipopt import Eventhdlr, SCIP_EVENTTYPE, SCIP_STAGE

    event_names = {
        SCIP_EVENTTYPE.PRESOLVEROUND: "presolve_round",
        SCIP_EVENTTYPE.NODEFOCUSED: "node_focused",
        SCIP_EVENTTYPE.DUALBOUNDIMPROVED: "dual_bound",
        SCIP_EVENTTYPE.FIRSTLPSOLVED: "first_lp",
        SCIP_EVENTTYPE.BESTSOLFOUND: "incumbent",
        SCIP_EVENTTYPE.NODEBRANCHED: "node_branched",
    }

    class WhiteBoxTrace(Eventhdlr):
        def eventinit(self):
            for event_type in event_names:
                self.model.catchEvent(event_type, self)

        def eventexec(self, event):
            event_type = event.getType()
            event_name = event_names.get(event_type, f"event_{int(event_type)}")
            now = float(self.model.getSolvingTime())
            if event_name in ("dual_bound", "node_focused", "node_branched"):
                last = self.model.data["trace_last_sample"].get(event_name, -math.inf)
                if now - last < sample_interval_s:
                    return
                self.model.data["trace_last_sample"][event_name] = now

            solving = self.model.getStage() == SCIP_STAGE.SOLVING
            current = self.model.getCurrentNode() if solving else None
            record: dict[str, Any] = {
                "event": event_name,
                "time_s": now,
                "nodes": int(self.model.getNNodes()),
                "depth": int(current.getDepth()) if current is not None else None,
                "primal_bound": _finite_or_none(self.model.getPrimalbound()),
                "dual_bound": _finite_or_none(self.model.getDualbound()),
                "gap": _finite_or_none(self.model.getGap()),
                "solutions": int(self.model.getNSols()),
                "lp_solves": int(self.model.getNLPs()),
                "lp_iterations": int(self.model.getNLPIterations()) if solving else None,
                "strong_branch_lp_iterations": (
                    int(self.model.getNStrongbranchLPIterations()) if solving else None
                ),
                "separation_rounds": int(self.model.getNSepaRounds()) if solving else None,
                "cuts_applied": int(self.model.getNCutsApplied()) if solving else None,
            }
            if event_name == "presolve_round":
                record["presolved_size"] = {
                    "variables": int(self.model.getNVars()),
                    "constraints": int(self.model.getNConss()),
                }
            if (
                event_name == "first_lp"
                and record["depth"] == 0
                and self.model.data["root_lp"] is None
            ):
                record["lp"] = _lp_snapshot(self.model)
                self.model.data["root_lp"] = record["lp"]
            if (
                event_name == "node_branched"
                and record["depth"] == 0
                and self.model.data["root_final_lp"] is None
            ):
                record["lp"] = _lp_snapshot(self.model)
                self.model.data["root_final_lp"] = record["lp"]
            if event_name == "incumbent":
                solution = self.model.getBestSol()
                record["objective"] = _finite_or_none(self.model.getPrimalbound())
                record["solution_origin"] = (
                    str(solution.getOrigin()) if solution is not None else None
                )
            self.model.data["trace"].append(record)

    model.data = {
        "trace": [],
        "trace_last_sample": {},
        "root_lp": None,
        "root_final_lp": None,
    }
    model.includeEventhdlr(
        WhiteBoxTrace(), "qopt_whitebox", "q-opt SCIP parity trace"
    )


def _compact_statistics(
    problem: dict[str, Any],
    stats: dict[str, Any],
    root_lp: dict[str, Any] | None,
    root_final_lp: dict[str, Any] | None,
) -> dict[str, Any]:
    original = stats.get("origprob", {})
    presolved = stats.get("presolvedprob", {})
    root = stats.get("root", {})
    lp = stats.get("lp", {})
    solution = stats.get("solution", {})
    tree = stats.get("tree", {})
    return {
        "original_size": {
            "variables": original.get("num_variables", len(problem["vars"])),
            "constraints": original.get(
                "num_initial_constraints", len(problem.get("constraints", []))
            ),
            "nonzeros": sum(
                len(constraint.get("coeffs", {}))
                for constraint in problem.get("constraints", [])
            ),
        },
        "presolved_size": {
            "variables": presolved.get("num_variables"),
            "constraints": presolved.get("num_initial_constraints"),
            "max_constraints": presolved.get("num_maximal_constraints"),
            "nonzeros": presolved.get("constraint_nonzeros"),
        },
        "root": {
            "first_lp_value": root.get("first_lp_value"),
            "first_lp_iterations": root.get("first_lp_iterations"),
            "first_lp_time_s": root.get("first_lp_time"),
            "final_dual_bound": root.get("final_dual_bound"),
            "final_root_iterations": root.get("final_root_iterations"),
            "first_lp_snapshot": root_lp,
            "final_lp_snapshot": root_final_lp,
        },
        "lp_breakdown": {
            "primal_iterations": lp.get("primal_lp", {}).get("iterations"),
            "dual_iterations": lp.get("dual_lp", {}).get("iterations"),
            "strong_branch_calls": _sum_fields(
                lp.get("strongbranch_lp", {}), ("calls",)
            )
            + _sum_fields(lp.get("strongbranch_root_lp", {}), ("calls",)),
            "strong_branch_iterations": _sum_fields(
                lp.get("strongbranch_lp", {}), ("iterations",)
            )
            + _sum_fields(lp.get("strongbranch_root_lp", {}), ("iterations",)),
        },
        "first_solution": {
            "objective": solution.get("first_primal_bound"),
            "time_s": solution.get("first_solution_time"),
            "nodes": solution.get("first_solution_nodes"),
            "depth": solution.get("first_solution_depth"),
        },
        "tree": tree,
        "presolvers": _rank_plugins(
            stats,
            "presolver",
            (
                "fixed_vars",
                "aggregated_vars",
                "changed_bounds",
                "deleted_constraints",
                "changed_coefficients",
            ),
            limit=64,
        ),
        "propagators": _rank_plugins(
            stats, "propagator", ("cutoffs", "domreds_found"), limit=64
        ),
        "separators": _rank_plugins(
            stats,
            "separator",
            ("cuts_applied", "cuts_added_via_pool", "cuts_found"),
            limit=64,
        ),
        "branchrules": _rank_plugins(
            stats,
            "branchrules",
            ("nchildren", "ndomreds", "nconssfound", "nlpcalls"),
            limit=64,
        ),
        "heuristics": _rank_plugins(
            stats, "heuristics", ("best_solutions_found", "solutions_found"), limit=64
        ),
    }


def _solve_json_problem(
    problem: dict[str, Any],
    time_limit_s: float,
    artifacts_dir: Path,
    parameters: list[str],
    sample_interval_s: float,
    expected_version: str,
    dynamic_constraints: bool,
    variable_order: str,
    mps_dir: Path,
) -> dict[str, Any]:
    from pyscipopt import __version__ as pyscipopt_version

    instance_dir = artifacts_dir / problem["name"]
    instance_dir.mkdir(parents=True, exist_ok=True)
    statistics_path = instance_dir / "statistics.json"
    solution_path = instance_dir / "best.sol"
    trace_path = instance_dir / "trace.jsonl"
    log_path = instance_dir / "scip.log"

    started = time.perf_counter()
    build_problem = problem
    if variable_order == "native-mps":
        build_problem = _reorder_json_variables_from_native(
            problem, mps_dir / f"{problem['name']}.mps.gz"
        )
    model, variables = _build_json_model(
        build_problem, time_limit_s, parameters, dynamic_constraints
    )
    scip_version = (
        f"{model.getMajorVersion()}.{model.getMinorVersion()}."
        f"{model.getTechVersion()}"
    )
    if scip_version != expected_version:
        model.freeProb()
        raise RuntimeError(f"expected SCIP {expected_version}, got {scip_version}")
    model.setLogfile(str(log_path))
    _attach_trace(model, sample_interval_s)
    model.optimize()
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    model.writeStatisticsJson(str(statistics_path))
    stats = json.loads(statistics_path.read_text())
    with trace_path.open("w") as trace_file:
        for record in model.data["trace"]:
            trace_file.write(json.dumps(record, sort_keys=True) + "\n")
    status = _normalize_status(str(model.getStatus()))
    objective = _sanitize(model.getObjVal()) if model.getNSols() > 0 else None
    bound = _sanitize(model.getDualbound())
    trace = model.data["trace"]
    item: dict[str, Any] = {
        "name": problem["name"],
        "status": status,
        "is_mip": _is_mip(problem),
        "direction": problem.get("direction", "min").lower(),
        "objective": objective,
        "best_bound": bound,
        "gap_pct": _gap_pct(objective, bound),
        "proven_optimal": status == "optimal",
        "iterations": int(model.getNLPIterations()),
        "nodes": int(model.getNNodes()),
        "elapsed_ms": elapsed_ms,
        "scip_solving_time_s": float(model.getSolvingTime()),
        "scip_version": scip_version,
        "pyscipopt_version": pyscipopt_version,
        "parameters": parameters,
        "json_dynamic_constraints": dynamic_constraints,
        "json_variable_order": variable_order,
        "whitebox": _compact_statistics(
            problem,
            stats,
            model.data.get("root_lp"),
            model.data.get("root_final_lp"),
        ),
        "incumbent_timeline": [
            record
            for record in trace
            if record["event"] == "incumbent" and record.get("objective") is not None
        ],
        "bound_timeline": [
            record
            for record in trace
            if record["event"] in ("dual_bound", "node_focused", "first_lp")
        ],
    }
    if objective is not None:
        model.writeBestSol(str(solution_path))
        item["values"] = {
            name: _sanitize(model.getVal(variable))
            for name, variable in variables.items()
        }
    item["files"] = {
        "statistics": str(statistics_path),
        "solution": str(solution_path) if objective is not None else None,
        "trace": str(trace_path),
        "log": str(log_path),
    }
    if status not in ("optimal", "infeasible", "unbounded"):
        item["error"] = f"scip_returned_{status}_not_optimal"
    model.freeProb()
    return item


def _parse_solution(path: Path, problem: dict[str, Any]) -> dict[str, float]:
    values = {variable["name"]: 0.0 for variable in problem["vars"]}
    if not path.is_file():
        return values
    with path.open(errors="replace") as solution_file:
        for line in solution_file:
            stripped = line.strip()
            if not stripped or stripped.startswith("objective value:"):
                continue
            fields = stripped.split()
            if len(fields) < 2 or fields[0] not in values:
                continue
            try:
                values[fields[0]] = _sanitize(fields[1])
            except ValueError:
                continue
    return values


def _solve_native_problem(
    driver: Path,
    problem: dict[str, Any],
    mps_dir: Path,
    artifacts_dir: Path,
    time_limit_s: float,
    parameters: list[str],
) -> dict[str, Any]:
    name = problem["name"]
    instance = mps_dir / f"{name}.mps.gz"
    instance_dir = artifacts_dir / name
    instance_dir.mkdir(parents=True, exist_ok=True)
    stats_path = instance_dir / "statistics.json"
    trace_path = instance_dir / "trace.jsonl"
    solution_path = instance_dir / "best.sol"
    result_path = instance_dir / "result.json"
    log_path = instance_dir / "scip.log"

    command = [
        str(driver),
        "--input",
        str(instance),
        "--stats",
        str(stats_path),
        "--trace",
        str(trace_path),
        "--solution",
        str(solution_path),
        "--summary",
        str(result_path),
        "--log",
        str(log_path),
        "--time-limit",
        str(time_limit_s),
        "--sample-interval",
        "1.0",
    ]
    for parameter in parameters:
        command.extend(("--set", parameter))

    started = time.perf_counter()
    subprocess.run(command, check=True, cwd=ROOT)
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    result = json.loads(result_path.read_text())
    status = _normalize_status(str(result.get("status", "unknown")))
    objective = _sanitize(result.get("objective"))
    bound = _sanitize(result.get("dual_bound"))
    item: dict[str, Any] = {
        "name": name,
        "status": status,
        "status_detail": result.get("status"),
        "is_mip": _is_mip(problem),
        "direction": problem.get("direction", "min").lower(),
        "objective": objective,
        "best_bound": bound,
        "gap_pct": _gap_pct(objective, bound),
        "proven_optimal": status == "optimal",
        "iterations": int(result.get("lp_iterations", 0)),
        "nodes": int(result.get("nodes", 0)),
        "elapsed_ms": elapsed_ms,
        "scip_solving_time_s": result.get("solving_time_s"),
        "scip_version": result.get("scip_version"),
        "scip_git_hash": result.get("scip_git_hash"),
    }
    if objective is not None:
        item["values"] = _parse_solution(solution_path, problem)
    if status not in ("optimal", "infeasible", "unbounded"):
        item["error"] = f"scip_returned_{status}_not_optimal"
    return item


def _solve_native_one(args: tuple[Any, ...]) -> dict[str, Any]:
    problem = args[1]
    try:
        return _solve_native_problem(*args)
    except Exception as exc:  # noqa: BLE001
        return {
            "name": problem.get("name", "<unknown>"),
            "status": "error",
            "elapsed_ms": 0.0,
            "error": f"{type(exc).__name__}: {exc}",
        }


def _solve_json_one(args: tuple[Any, ...]) -> dict[str, Any]:
    problem = args[0]
    try:
        return _solve_json_problem(*args)
    except Exception as exc:  # noqa: BLE001
        return {
            "name": problem.get("name", "<unknown>"),
            "status": "error",
            "elapsed_ms": 0.0,
            "error": f"{type(exc).__name__}: {exc}",
        }


def _load_problems(directory: Path) -> list[dict[str, Any]]:
    return [json.loads(path.read_text()) for path in sorted(directory.glob("*.json"))]


def _fmt_obj(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        if math.isnan(value):
            return "nan"
        if math.isinf(value):
            return "inf" if value > 0 else "-inf"
        return f"{value:.6g}"
    return str(value)


def _print_row(item: dict[str, Any]) -> None:
    elapsed_min = item.get("elapsed_ms", 0) / 60_000.0
    proven = "yes" if item.get("proven_optimal") else "no"
    gap = item.get("gap_pct")
    gap_s = f"{gap:.2f}%" if gap is not None else "n/a"
    print(
        f"{item['name']:<32} {proven:<3} {item.get('status', '?'):<14} "
        f"obj={_fmt_obj(item.get('objective')):<14} "
        f"bound={_fmt_obj(item.get('best_bound')):<14} "
        f"gap={gap_s:<9} nodes={item.get('nodes', 0):<8} "
        f"time={elapsed_min:6.2f}min",
        flush=True,
    )


def _print_summary(items: list[dict[str, Any]]) -> None:
    proven = [item for item in items if item.get("proven_optimal")]
    feasible_only = [
        item
        for item in items
        if item.get("status") not in ("optimal", "infeasible", "unbounded", "error")
        and item.get("objective") is not None
    ]
    print(f"\n=== summary: {len(items)} instances ===")
    print(f"  proven optimal in budget:   {len(proven)}")
    print(f"  feasible (time-limited):    {len(feasible_only)}")
    print(f"  other (infeasible / error): {len(items) - len(proven) - len(feasible_only)}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--problems", default="bench/problems/miplib")
    parser.add_argument("--only", default="", help="comma-separated problem names")
    parser.add_argument("--mps-dir", default="3rd_repo/miplib")
    parser.add_argument("--out", default="bench/results/scip_miplib_baseline.json")
    parser.add_argument(
        "--input-mode",
        choices=("json-api", "native-mps"),
        default="json-api",
        help="json-api is the matched q-opt baseline; native-mps is diagnostic",
    )
    parser.add_argument(
        "--artifacts-dir", default="bench/results/scip11-baseline-artifacts"
    )
    parser.add_argument("--time-limit-min", type=float, default=5.0)
    parser.add_argument("--sample-interval-s", type=float, default=0.25)
    parser.add_argument("--parallel", type=int, default=3)
    parser.add_argument("--expected-version", default="11.0.0")
    parser.add_argument("--scip-build", default=str(DEFAULT_SCIP_BUILD))
    parser.add_argument("--driver-build", default=str(DEFAULT_DRIVER_BUILD))
    parser.add_argument("--driver", default="")
    parser.add_argument("--baseline-existing", default=None)
    parser.add_argument("--set", action="append", default=[], metavar="NAME=VALUE")
    parser.add_argument(
        "--json-dynamic-conss",
        action="store_true",
        help=(
            "create JSON-API constraints with dynamic=true, matching SCIP's "
            "default MPS reader metadata; diagnostic only"
        ),
    )
    parser.add_argument(
        "--json-variable-order",
        choices=("problem", "native-mps"),
        default="problem",
        help=(
            "variable insertion order for JSON API; native-mps borrows only "
            "the native reader's order and is diagnostic"
        ),
    )
    args = parser.parse_args()

    if args.baseline_existing:
        items = json.loads(Path(args.baseline_existing).read_text())["items"]
        _print_summary(items)
        return 0

    problems_dir = Path(args.problems)
    if not problems_dir.is_dir():
        print("problem JSON directory does not exist", file=sys.stderr)
        return 2
    problems = _load_problems(problems_dir)
    selected = {name for name in args.only.split(",") if name}
    if selected:
        problems = [problem for problem in problems if problem["name"] in selected]
        missing = selected - {problem["name"] for problem in problems}
        if missing:
            print(f"unknown problem names: {', '.join(sorted(missing))}", file=sys.stderr)
            return 2
    time_limit_s = args.time_limit_min * 60.0
    parallel = max(1, args.parallel)
    artifacts_dir = Path(args.artifacts_dir).resolve()
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    mps_dir = Path(args.mps_dir)
    if args.input_mode == "native-mps" or args.json_variable_order == "native-mps":
        if not mps_dir.is_dir():
            print("MPS directory does not exist", file=sys.stderr)
            return 2
    print(f"running SCIP on {len(problems)} instances")
    print(f"  input mode:   {args.input_mode}")
    print(f"  time limit:   {args.time_limit_min:.1f} min/instance")
    print(f"  parallel:     {parallel}\n")

    items_by_name: dict[str, dict[str, Any]] = {}
    started = time.perf_counter()
    if args.input_mode == "json-api":
        work = [
            (
                problem,
                time_limit_s,
                artifacts_dir,
                list(args.set),
                args.sample_interval_s,
                args.expected_version,
                args.json_dynamic_conss,
                args.json_variable_order,
                mps_dir,
            )
            for problem in problems
        ]
        pool_type = ProcessPoolExecutor
        solve_one = _solve_json_one
    else:
        missing = [
            problem["name"]
            for problem in problems
            if not (mps_dir / f"{problem['name']}.mps.gz").is_file()
        ]
        if missing:
            print(f"missing MPS files: {', '.join(missing)}", file=sys.stderr)
            return 2
        driver = (
            Path(args.driver).resolve()
            if args.driver
            else _build_driver(Path(args.scip_build), Path(args.driver_build))
        )
        print(f"  driver:       {driver}\n")
        work = [
            (driver, problem, mps_dir, artifacts_dir, time_limit_s, args.set)
            for problem in problems
        ]
        pool_type = ThreadPoolExecutor
        solve_one = _solve_native_one

    if parallel == 1:
        results = (solve_one(item) for item in work)
        for result in results:
            items_by_name[result["name"]] = result
            _print_row(result)
    else:
        with pool_type(max_workers=parallel) as pool:
            futures = {
                pool.submit(solve_one, item): problem["name"]
                for item, problem in zip(work, problems, strict=True)
            }
            for future in as_completed(futures):
                result = future.result()
                items_by_name[result["name"]] = result
                _print_row(result)
    items = [items_by_name[problem["name"]] for problem in problems]

    versions = {item.get("scip_version") for item in items if item.get("scip_version")}
    git_hashes = {item.get("scip_git_hash") for item in items if item.get("scip_git_hash")}
    binding_versions = {
        item.get("pyscipopt_version")
        for item in items
        if item.get("pyscipopt_version")
    }
    payload = {
        "solver": "scip",
        "version": next(iter(versions)) if len(versions) == 1 else sorted(versions),
        "git_hash": (
            next(iter(git_hashes))
            if len(git_hashes) == 1
            else sorted(git_hashes) if git_hashes else None
        ),
        "pyscipopt_version": (
            next(iter(binding_versions))
            if len(binding_versions) == 1
            else sorted(binding_versions) if binding_versions else None
        ),
        "input_mode": args.input_mode,
        "driver": "pyscipopt_json_api" if args.input_mode == "json-api" else "native_c_api",
        "time_limit_min": args.time_limit_min,
        "parallel": parallel,
        "parameters": args.set,
        "json_dynamic_constraints": (
            args.json_dynamic_conss if args.input_mode == "json-api" else None
        ),
        "json_variable_order": (
            args.json_variable_order if args.input_mode == "json-api" else None
        ),
        "total_wall_time_s": time.perf_counter() - started,
        "items": items,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"\nwrote {out_path} ({len(items)} items)")
    _print_summary(items)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
