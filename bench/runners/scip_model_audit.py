#!/usr/bin/env python3
"""Audit JSON-API SCIP models against SCIP's native MPS reader."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

from scip_miplib_baseline import _build_json_model


def _number(value: float) -> str:
    if math.isinf(value):
        return "+inf" if value > 0 else "-inf"
    if abs(value) < 1e-12:
        value = 0.0
    return format(value, ".15g")


def _variables(model: Any) -> tuple[list[str], dict[str, dict[str, Any]]]:
    order: list[str] = []
    result: dict[str, dict[str, Any]] = {}
    for variable in model.getVars():
        order.append(variable.name)
        result[variable.name] = {
            "type": variable.vtype(),
            "lower": float(variable.getLbOriginal()),
            "upper": float(variable.getUbOriginal()),
            "objective": float(variable.getObj()),
        }
    return order, result


def _side_signature(
    sense: str, rhs: float, coefficients: dict[str, float]
) -> tuple[str, str, tuple[tuple[str, str], ...]]:
    entries = tuple(
        sorted(
            (name, _number(float(value)))
            for name, value in coefficients.items()
            if float(value) != 0.0
        )
    )
    return sense, _number(rhs), entries


def _row_sides(model: Any) -> tuple[list[str], Counter[Any]]:
    order: list[str] = []
    sides: Counter[Any] = Counter()
    infinity = float(model.infinity())
    for constraint in model.getConss():
        order.append(constraint.name)
        if constraint.getConshdlrName() != "linear":
            sides[("handler", constraint.getConshdlrName(), constraint.name)] += 1
            continue
        coefficients = model.getValsLinear(constraint)
        lhs = float(model.getLhs(constraint))
        rhs = float(model.getRhs(constraint))
        lhs_finite = lhs > -0.5 * infinity
        rhs_finite = rhs < 0.5 * infinity
        if lhs_finite and rhs_finite and _number(lhs) == _number(rhs):
            sides[_side_signature("=", lhs, coefficients)] += 1
        else:
            if lhs_finite:
                sides[_side_signature(">=", lhs, coefficients)] += 1
            if rhs_finite:
                sides[_side_signature("<=", rhs, coefficients)] += 1
    return order, sides


def _constraint_flags(model: Any) -> Counter[Any]:
    attributes = (
        "isInitial",
        "isSeparated",
        "isEnforced",
        "isChecked",
        "isPropagated",
        "isLocal",
        "isModifiable",
        "isDynamic",
        "isRemovable",
        "isStickingAtNode",
    )
    return Counter(
        tuple(getattr(constraint, attribute)() for attribute in attributes)
        for constraint in model.getConss()
    )


def _domain_equivalent(left: dict[str, Any], right: dict[str, Any]) -> bool:
    if _number(left["lower"]) != _number(right["lower"]):
        return False
    if _number(left["upper"]) != _number(right["upper"]):
        return False
    if left["type"] == right["type"]:
        return True
    return {left["type"], right["type"]} == {"BINARY", "INTEGER"}


def _sample_counter(counter: Counter[Any], limit: int = 8) -> list[dict[str, Any]]:
    result = []
    for signature, count in counter.items():
        result.append({"count": count, "signature": signature})
        if len(result) >= limit:
            break
    return result


def audit(problem_path: Path, mps_path: Path) -> dict[str, Any]:
    from pyscipopt import Model

    problem = json.loads(problem_path.read_text())
    json_model, _ = _build_json_model(problem, 1.0, [])
    native_model = Model()
    native_model.hideOutput()
    native_model.readProblem(str(mps_path))

    json_var_order, json_vars = _variables(json_model)
    native_var_order, native_vars = _variables(native_model)
    json_row_order, json_sides = _row_sides(json_model)
    native_row_order, native_sides = _row_sides(native_model)
    json_flags = _constraint_flags(json_model)
    native_flags = _constraint_flags(native_model)

    missing_in_json = sorted(set(native_vars) - set(json_vars))
    missing_in_native = sorted(set(json_vars) - set(native_vars))
    domain_differences = []
    objective_differences = []
    type_only_differences = []
    for name in sorted(set(json_vars) & set(native_vars)):
        json_var = json_vars[name]
        native_var = native_vars[name]
        if not _domain_equivalent(json_var, native_var):
            domain_differences.append(
                {"name": name, "json": json_var, "native": native_var}
            )
        elif json_var["type"] != native_var["type"]:
            type_only_differences.append(
                {"name": name, "json": json_var, "native": native_var}
            )
        if _number(json_var["objective"]) != _number(native_var["objective"]):
            objective_differences.append(
                {"name": name, "json": json_var, "native": native_var}
            )

    only_json_sides = json_sides - native_sides
    only_native_sides = native_sides - json_sides
    result = {
        "name": problem["name"],
        "mathematically_equivalent": not (
            missing_in_json
            or missing_in_native
            or domain_differences
            or objective_differences
            or only_json_sides
            or only_native_sides
            or _number(float(json_model.getObjoffset()))
            != _number(float(native_model.getObjoffset()))
            or json_model.getObjectiveSense() != native_model.getObjectiveSense()
        ),
        "json": {
            "variables": len(json_vars),
            "constraints": len(json_row_order),
            "row_sides": sum(json_sides.values()),
            "objective_offset": float(json_model.getObjoffset()),
            "objective_sense": json_model.getObjectiveSense(),
        },
        "native": {
            "variables": len(native_vars),
            "constraints": len(native_row_order),
            "row_sides": sum(native_sides.values()),
            "objective_offset": float(native_model.getObjoffset()),
            "objective_sense": native_model.getObjectiveSense(),
        },
        "missing_variables_in_json": missing_in_json,
        "missing_variables_in_native": missing_in_native,
        "domain_differences": domain_differences,
        "objective_differences": objective_differences,
        "type_only_differences": type_only_differences,
        "row_sides_only_in_json": {
            "count": sum(only_json_sides.values()),
            "sample": _sample_counter(only_json_sides),
        },
        "row_sides_only_in_native": {
            "count": sum(only_native_sides.values()),
            "sample": _sample_counter(only_native_sides),
        },
        "order": {
            "same_variable_order": json_var_order == native_var_order,
            "same_constraint_order": json_row_order == native_row_order,
            "json_variables_first": json_var_order[:12],
            "native_variables_first": native_var_order[:12],
            "json_constraints_first": json_row_order[:12],
            "native_constraints_first": native_row_order[:12],
        },
        "constraint_flags": {
            "same": json_flags == native_flags,
            "json": [
                {"count": count, "flags": flags}
                for flags, count in json_flags.items()
            ],
            "native": [
                {"count": count, "flags": flags}
                for flags, count in native_flags.items()
            ],
        },
    }
    json_model.freeProb()
    native_model.freeProb()
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--problems", default="bench/problems/miplib")
    parser.add_argument("--mps-dir", default="3rd_repo/miplib")
    parser.add_argument("--only", default="")
    parser.add_argument("--out")
    args = parser.parse_args()

    selected = {name.strip() for name in args.only.split(",") if name.strip()}
    problem_dir = Path(args.problems)
    mps_dir = Path(args.mps_dir)
    results = []
    for problem_path in sorted(problem_dir.glob("*.json")):
        problem = json.loads(problem_path.read_text())
        if selected and problem["name"] not in selected:
            continue
        results.append(
            audit(problem_path, mps_dir / f"{problem['name']}.mps.gz")
        )

    payload = {"items": results}
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.out:
        Path(args.out).write_text(text)
    else:
        print(text, end="")
    return 1 if any(not item["mathematically_equivalent"] for item in results) else 0


if __name__ == "__main__":
    raise SystemExit(main())
