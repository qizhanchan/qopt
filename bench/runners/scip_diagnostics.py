#!/usr/bin/env python3
"""Collect white-box diagnostics from the locally built SCIP 11.

The native driver links directly against ``3rd_repo/scip``'s build and writes:

* ``<name>.trace.jsonl``: sampled primal/dual, LP, node, cut, and strong-branch events
* ``<name>.stats.json``: SCIP 11's complete plugin statistics
* ``<name>.sol``: best solution in SCIP's text format
* ``<name>.log``: native detailed solve log
* ``summary.json``: compact plugin rankings and per-instance metrics

Example:

    python bench/runners/scip_diagnostics.py \
        --names neos17,timtab1 --time-limit-s 30 --parallel 2 \
        --out-dir /tmp/scip11-diagnostics
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SCIP_BUILD = Path("/tmp/qopt-scip11-build")
DEFAULT_DRIVER_BUILD = Path("/tmp/qopt-scip-diagnostic-build")


def _sum_fields(values: dict[str, Any], fields: tuple[str, ...]) -> float:
    return sum(float(values.get(field, 0) or 0) for field in fields)


def _rank_plugins(
    stats: dict[str, Any],
    section: str,
    score_fields: tuple[str, ...],
    limit: int = 8,
) -> list[dict[str, Any]]:
    plugins = stats.get(section, {}).get("plugins", {})
    ranked = []
    for name, values in plugins.items():
        score = _sum_fields(values, score_fields)
        if score <= 0:
            continue
        useful = {
            key: value
            for key, value in values.items()
            if key != "description" and isinstance(value, (int, float)) and value != 0
        }
        ranked.append({"name": name, "score": score, **useful})
    ranked.sort(key=lambda item: (-item["score"], item["name"]))
    return ranked[:limit]


def _trace_counts(trace_path: Path) -> dict[str, int]:
    counts: Counter[str] = Counter()
    with trace_path.open() as trace_file:
        for line in trace_file:
            if line.strip():
                counts[json.loads(line)["event"]] += 1
    return dict(sorted(counts.items()))


def _parse_scalar(value: str) -> int | float | str:
    try:
        return int(value)
    except ValueError:
        try:
            return float(value)
        except ValueError:
            return value


def _aggregation_diagnostics(log_path: Path, output_path: Path) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    with log_path.open(errors="replace") as log_file:
        for line in log_file:
            marker = "QOPT_AGG "
            marker_pos = line.find(marker)
            if marker_pos < 0:
                continue
            record = {}
            for field in line[marker_pos + len(marker) :].split():
                if "=" not in field:
                    continue
                key, value = field.split("=", 1)
                record[key] = _parse_scalar(value)
            records.append(record)

    with output_path.open("w") as output_file:
        for record in records:
            output_file.write(json.dumps(record, sort_keys=True) + "\n")

    totals = {
        field: sum(float(record.get(field, 0) or 0) for record in records)
        for field in (
            "rows",
            "candidates",
            "ignored",
            "zero_frac",
            "eligible_lhs",
            "eligible_rhs",
            "reject_infinite",
            "reject_slack",
            "reject_density",
            "tries",
            "fails",
            "cuts",
            "elapsed",
        )
    }
    tries = totals["tries"]
    return {
        "rounds": len(records),
        "root_rounds": sum(record.get("depth") == 0 for record in records),
        "nonroot_rounds": sum(record.get("depth") != 0 for record in records),
        "totals": totals,
        "cuts_per_try": totals["cuts"] / tries if tries else None,
        "max_round_time_s": max(
            (float(record.get("elapsed", 0) or 0) for record in records),
            default=0.0,
        ),
    }


def _normalize_status(status: str) -> str:
    status_lower = status.lower()
    if "optimal" in status_lower:
        return "optimal"
    if "time limit" in status_lower or status_lower == "timelimit":
        return "time_limit"
    if "node limit" in status_lower or status_lower == "nodelimit":
        return "node_limit"
    if "gap limit" in status_lower or status_lower == "gaplimit":
        return "gap_limit"
    if "infeasible" in status_lower and "unbounded" in status_lower:
        return "infeasible_or_unbounded"
    if "infeasible" in status_lower:
        return "infeasible"
    if "unbounded" in status_lower:
        return "unbounded"
    return status_lower.replace(" ", "_")


def _compact_summary(
    name: str,
    driver_result: dict[str, Any],
    stats: dict[str, Any],
    trace_path: Path,
    aggregation_summary: dict[str, Any],
) -> dict[str, Any]:
    orig = stats.get("origprob", {})
    presolved = stats.get("presolvedprob", {})
    root = stats.get("root", {})
    lp = stats.get("lp", {})

    status_detail = str(driver_result.get("status", "unknown"))
    normalized_result = {
        **driver_result,
        "status": _normalize_status(status_detail),
        "status_detail": status_detail,
    }
    return {
        "name": name,
        **normalized_result,
        "original_size": {
            "variables": orig.get("num_variables"),
            "constraints": orig.get("num_initial_constraints"),
        },
        "presolved_size": {
            "variables": presolved.get("num_variables"),
            "constraints": presolved.get("num_initial_constraints"),
            "nonzeros": presolved.get("constraint_nonzeros"),
        },
        "root": {
            "first_lp_value": root.get("first_lp_value"),
            "first_lp_iterations": root.get("first_lp_iterations"),
            "first_lp_time_s": root.get("first_lp_time"),
            "final_dual_bound": root.get("final_dual_bound"),
            "final_root_iterations": root.get("final_root_iterations"),
        },
        "lp_breakdown": {
            "primal_iterations": lp.get("primal_lp", {}).get("iterations"),
            "dual_iterations": lp.get("dual_lp", {}).get("iterations"),
            "strong_branch_iterations": _sum_fields(
                lp.get("strongbranch_lp", {}), ("iterations",)
            )
            + _sum_fields(lp.get("strongbranch_root_lp", {}), ("iterations",)),
        },
        "event_counts": _trace_counts(trace_path),
        "aggregation_diagnostics": aggregation_summary,
        "top_presolvers": _rank_plugins(
            stats,
            "presolver",
            (
                "fixed_vars",
                "aggregated_vars",
                "changed_var_types",
                "changed_bounds",
                "added_holes",
                "deleted_constraints",
                "added_constraints",
                "changed_sides",
                "changed_coefficients",
            ),
        ),
        "top_propagators": _rank_plugins(
            stats, "propagator", ("cutoffs", "domreds_found")
        ),
        "top_separators": _rank_plugins(
            stats,
            "separator",
            ("cuts_applied", "cuts_applied_via_pool", "cuts_found"),
        ),
        "top_heuristics": _rank_plugins(
            stats, "heuristics", ("best_solutions_found", "solutions_found")
        ),
        "top_branchrules": _rank_plugins(
            stats,
            "branchrules",
            ("nchildren", "ndomreds", "nconssfound", "nlpcalls"),
        ),
    }


def _build_driver(scip_build: Path, driver_build: Path) -> Path:
    source_dir = Path(__file__).resolve().parent / "scip_native"
    driver = driver_build / "qopt_scip_diagnostic"
    subprocess.run(
        [
            "cmake",
            "-S",
            str(source_dir),
            "-B",
            str(driver_build),
            "-G",
            "Ninja",
            f"-DSCIP_DIR={scip_build}",
            "-DCMAKE_BUILD_TYPE=RelWithDebInfo",
        ],
        check=True,
    )
    subprocess.run(
        ["cmake", "--build", str(driver_build), "-j", "3"],
        check=True,
    )
    if not driver.is_file():
        raise FileNotFoundError(f"native SCIP diagnostic driver not found: {driver}")
    return driver


def _instance_name(path: Path) -> str:
    suffix = ".mps.gz"
    if not path.name.endswith(suffix):
        raise ValueError(f"expected {suffix} file: {path}")
    return path.name[: -len(suffix)]


def _load_selected(mps_dir: Path, names: set[str]) -> list[Path]:
    instances = sorted(mps_dir.glob("*.mps.gz"))
    selected = [path for path in instances if not names or _instance_name(path) in names]
    missing = names - {_instance_name(path) for path in selected}
    if missing:
        raise ValueError(f"unknown problem names: {', '.join(sorted(missing))}")
    return selected


def _solve_one(
    driver: Path,
    instance: Path,
    out_dir: Path,
    time_limit_s: float,
    sample_interval_s: float,
    parameters: list[str],
    expected_version: str,
) -> dict[str, Any]:
    name = _instance_name(instance)
    stats_path = out_dir / f"{name}.stats.json"
    trace_path = out_dir / f"{name}.trace.jsonl"
    solution_path = out_dir / f"{name}.sol"
    result_path = out_dir / f"{name}.result.json"
    log_path = out_dir / f"{name}.log"
    aggregation_path = out_dir / f"{name}.aggregation.jsonl"
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
        str(sample_interval_s),
    ]
    for parameter in parameters:
        command.extend(("--set", parameter))

    started = time.perf_counter()
    subprocess.run(command, check=True, cwd=ROOT)
    wall_time_s = time.perf_counter() - started
    driver_result = json.loads(result_path.read_text())
    if driver_result.get("scip_version") != expected_version:
        raise RuntimeError(
            f"{name}: expected SCIP {expected_version}, "
            f"got {driver_result.get('scip_version')}"
        )
    stats = json.loads(stats_path.read_text())
    aggregation_summary = _aggregation_diagnostics(log_path, aggregation_path)
    summary = _compact_summary(
        name, driver_result, stats, trace_path, aggregation_summary
    )
    summary["wall_time_s"] = wall_time_s
    summary["parameters"] = parameters
    summary["files"] = {
        "statistics": str(stats_path),
        "trace": str(trace_path),
        "solution": str(solution_path),
        "log": str(log_path),
        "aggregation_trace": str(aggregation_path),
    }
    return summary


def _format_number(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float) and not math.isfinite(value):
        return "n/a"
    return f"{value:.8g}" if isinstance(value, float) else str(value)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mps-dir", default=str(ROOT / "3rd_repo/miplib"))
    parser.add_argument("--names", default="")
    parser.add_argument("--out-dir", default="bench/results/scip11-diagnostics")
    parser.add_argument("--time-limit-s", type=float, default=30.0)
    parser.add_argument("--sample-interval-s", type=float, default=0.25)
    parser.add_argument("--parallel", type=int, default=3)
    parser.add_argument("--scip-build", default=str(DEFAULT_SCIP_BUILD))
    parser.add_argument("--driver-build", default=str(DEFAULT_DRIVER_BUILD))
    parser.add_argument("--expected-version", default="11.0.0")
    parser.add_argument(
        "--driver",
        default="",
        help="reuse an existing native driver instead of configuring its CMake build",
    )
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="NAME=VALUE",
        help="override a SCIP parameter; may be repeated",
    )
    args = parser.parse_args()

    names = {name.strip() for name in args.names.split(",") if name.strip()}
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    instances = _load_selected(Path(args.mps_dir), names)
    if args.driver:
        driver = Path(args.driver).resolve()
        if not driver.is_file():
            raise FileNotFoundError(f"native SCIP diagnostic driver not found: {driver}")
    else:
        driver = _build_driver(Path(args.scip_build), Path(args.driver_build))

    summaries = []
    with ThreadPoolExecutor(max_workers=max(1, args.parallel)) as pool:
        futures = {
            pool.submit(
                _solve_one,
                driver,
                instance,
                out_dir,
                args.time_limit_s,
                args.sample_interval_s,
                args.set,
                args.expected_version,
            ): _instance_name(instance)
            for instance in instances
        }
        for future in as_completed(futures):
            summary = future.result()
            summaries.append(summary)
            print(
                f"{summary['name']:<28} status={summary['status']:<24} "
                f"obj={_format_number(summary['objective']):<12} "
                f"bound={_format_number(summary['dual_bound']):<12} "
                f"gap={_format_number(summary['gap']):<10} "
                f"nodes={summary['nodes']}",
                flush=True,
            )

    summaries.sort(key=lambda item: item["name"])
    summary_path = out_dir / "summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "scip_build": str(Path(args.scip_build).resolve()),
                "driver": str(driver.resolve()),
                "expected_version": args.expected_version,
                "items": summaries,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    print(f"wrote {summary_path}")


if __name__ == "__main__":
    main()
