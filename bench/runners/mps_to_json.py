#!/usr/bin/env python3
"""Convert one or more MPS files into the bench/problems JSON format.

Uses OR-Tools' ``model_builder`` to parse the MPS (it handles ``.mps`` and
``.mps.gz`` and is well-tested against MIPLIB), then writes a JSON file per
input. Designed for the MIPLIB instances downloaded under ``3rd_repo/miplib``.

Each constraint is normalized to a single sense:

* ``lb == ub``                    →  "="  (rhs = lb)
* finite lb, no finite ub         →  ">=" (rhs = lb)
* finite ub, no finite lb         →  "<=" (rhs = ub)
* finite lb AND finite ub (range) →  emitted as TWO rows: a ">=" and a "<=",
  named ``<orig>__lo`` / ``<orig>__hi``.

Free constraints (lb=-inf, ub=+inf) are dropped — they are vacuous.

Variables: finite lower/upper bounds are copied verbatim. ``+inf`` upper bounds
remain ``null`` because that is the shared schema default. A ``-inf`` lower
bound is represented by ``lower_unbounded: true``; plain ``lower: null`` keeps
the q-opt-compatible default lower bound of zero. ``binary`` vs ``integer`` is
decided by integrality flag plus [0,1] bounds.

Output file name is ``<problem_name>.json``; problem name is taken from the
MPS header (NAME line) when present, otherwise from the file basename.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

from ortools.linear_solver.python import model_builder as mb


def _classify_var_type(helper: Any, idx: int) -> str:
    if not helper.var_is_integral(idx):
        return "continuous"
    lb = helper.var_lower_bound(idx)
    ub = helper.var_upper_bound(idx)
    if lb == 0.0 and ub == 1.0:
        return "binary"
    return "integer"


def _finite(x: float) -> bool:
    return not (math.isinf(x) or math.isnan(x))


def _bound_or_none(x: float) -> float | None:
    return None if not _finite(x) else x


def convert(path: Path, sanitize_name: bool = True) -> dict[str, Any]:
    model = mb.Model()
    if path.suffix == ".gz":
        # model_builder can read .mps via import_from_mps_file but the bindings
        # don't transparently gunzip. Decompress in memory.
        import gzip
        text = gzip.decompress(path.read_bytes()).decode("utf-8", errors="replace")
        model.import_from_mps_string(text)
    else:
        model.import_from_mps_file(str(path))

    helper = model.helper
    var_count = helper.num_variables()
    con_count = helper.num_constraints()

    name = helper.name() or path.stem.replace(".mps", "")
    if sanitize_name:
        name = name.strip()
        if not name:
            name = path.stem.replace(".mps", "")

    vars_out: list[dict[str, Any]] = []
    var_names: list[str] = []
    for i in range(var_count):
        vn = helper.var_name(i) or f"x{i}"
        var_names.append(vn)
        lower = helper.var_lower_bound(i)
        variable = {
            "name": vn,
            "type": _classify_var_type(helper, i),
            "lower": _bound_or_none(lower),
            "upper": _bound_or_none(helper.var_upper_bound(i)),
        }
        if math.isinf(lower) and lower < 0:
            variable["lower_unbounded"] = True
        vars_out.append(variable)

    objective: dict[str, float] = {}
    for i in range(var_count):
        c = helper.var_objective_coefficient(i)
        if c != 0.0:
            objective[var_names[i]] = c

    direction = "max" if helper.maximize() else "min"
    obj_offset = helper.objective_offset()

    constraints_out: list[dict[str, Any]] = []
    for j in range(con_count):
        lb = helper.constraint_lower_bound(j)
        ub = helper.constraint_upper_bound(j)
        idx = helper.constraint_var_indices(j)
        coefs = helper.constraint_coefficients(j)
        coeffs = {var_names[k]: coefs[t] for t, k in enumerate(idx)}
        cname = helper.constraint_name(j) or f"c{j}"

        lb_finite = _finite(lb)
        ub_finite = _finite(ub)
        if not lb_finite and not ub_finite:
            continue  # free row — vacuous
        if lb_finite and ub_finite and lb == ub:
            constraints_out.append({"name": cname, "coeffs": coeffs, "sense": "=", "rhs": lb})
        elif lb_finite and ub_finite:
            # Range row → emit both halves.
            constraints_out.append({"name": f"{cname}__lo", "coeffs": coeffs, "sense": ">=", "rhs": lb})
            constraints_out.append({"name": f"{cname}__hi", "coeffs": coeffs, "sense": "<=", "rhs": ub})
        elif ub_finite:
            constraints_out.append({"name": cname, "coeffs": coeffs, "sense": "<=", "rhs": ub})
        else:
            constraints_out.append({"name": cname, "coeffs": coeffs, "sense": ">=", "rhs": lb})

    out: dict[str, Any] = {
        "name": name,
        "direction": direction,
        "vars": vars_out,
        "objective": objective,
        "constraints": constraints_out,
    }
    if obj_offset != 0.0:
        # We deliberately do NOT propagate offset into the bench JSON: both
        # runners read from the JSON without offset, so they'll agree, but the
        # reported objective will differ from the original MPS by exactly this
        # amount. Surface it so a user comparing against an external reference
        # (e.g. MIPLIB's published optimum) isn't confused.
        print(f"[warn] {name}: dropping non-zero objective_offset={obj_offset}",
              file=sys.stderr)
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("inputs", nargs="+", help="MPS or MPS.gz files to convert")
    p.add_argument("--out-dir", required=True, help="directory to write JSON into")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for in_path in args.inputs:
        path = Path(in_path)
        try:
            problem = convert(path)
        except Exception as exc:  # noqa: BLE001
            print(f"[skip] {path.name}: {type(exc).__name__}: {exc}", file=sys.stderr)
            continue
        out_path = out_dir / f"{problem['name']}.json"
        out_path.write_text(json.dumps(problem, indent=2) + "\n")
        print(f"{path.name:<40} -> {out_path.name}  "
              f"(vars={len(problem['vars'])}, cons={len(problem['constraints'])}, "
              f"obj_nz={len(problem['objective'])})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
