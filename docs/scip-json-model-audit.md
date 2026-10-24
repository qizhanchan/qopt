# SCIP JSON/API versus native MPS audit

This document records the July 18, 2026 audit of the shared benchmark JSON
construction path against SCIP 11's native MPS reader. The machine-readable
auditor is [`scip_model_audit.py`](https://github.com/qizhanchan/qopt/blob/main/bench/runners/scip_model_audit.py).

## Conclusion

For `neos17` and `timtab1`, the JSON/API and native-MPS paths build the same
mathematical model within `1e-12`:

- the same variables, original bounds, and effective variable types;
- the same objective coefficients, sense, and zero objective offset;
- the same row sides and sparse coefficients;
- the same constraint order.

The different SCIP trajectories are caused by construction metadata and
internal order, not by lost coefficients or bounds:

1. SCIP's MPS reader creates model constraints with `dynamic=true`; the
   PySCIPOpt JSON builder used the API default `dynamic=false`.
2. The MPS reader produces a different internal variable order even though
   the variable names and domains are identical.

Therefore, JSON/API input is not intrinsically more stable or faster than
native MPS. It is the fair benchmark path because q-opt consumes that same
converted model, and it gives a deterministic construction independent of the
SCIP reader. Native MPS remains useful for source and plugin diagnostics.

## Exact target comparison

| instance | path | variables | row sides | objective offset | variable order | constraint order |
|----------|------|----------:|----------:|-----------------:|----------------|------------------|
| `neos17` | JSON/API | 535 | 486 | 0 | JSON order | same |
| `neos17` | native MPS | 535 | 486 | 0 | reader order | same |
| `timtab1` | JSON/API | 397 | 171 | 0 | JSON order | same |
| `timtab1` | native MPS | 397 | 171 | 0 | reader order | same |

`timtab1` contains 13 variables fixed at zero that the source JSON classifies
as integer because the raw MPS integrality flag is set. SCIP normalizes those
fixed `[0,0]` variables to binary in both actual constructed models, so this is
not a domain or model difference.

## Causal 30-second A/B

All runs used SCIP 11.0.0, JSON input unless noted, and `--parallel 3`.

| construction | `neos17` | `timtab1` dual bound / gap | `timtab1` nodes |
|--------------|----------|------------------------------|-----------------:|
| JSON default | optimal, 4116 nodes | 695941.49 / 9.00% | 20189 |
| JSON with dynamic constraints | optimal, 4116 nodes | 711080.31 / 7.02% | 25941 |
| native MPS | optimal, 1614 nodes | 715910.97 / 6.39% | 23181 |
| JSON with dynamic constraints and native variable order | optimal, 1614 nodes | 715838.50 / 6.40% | 23138 |

The final JSON A/B reproduces native MPS almost exactly: identical `neos17`
node and LP-iteration counts, and a 43-node / 0.01 percentage-point difference
on `timtab1`. This accounts for the observed trajectory difference without a
mathematical-model discrepancy.

The runner exposes both controls for diagnostics only:

```sh
PYTHONPATH=/tmp/qopt-pyscipopt-scip11-package \
conda run -n base python bench/runners/scip_miplib_baseline.py \
  --problems bench/problems/miplib \
  --only neos17,timtab1 \
  --out /tmp/scip11-json-native-order-dynamic.json \
  --artifacts-dir /tmp/scip11-json-native-order-dynamic-artifacts \
  --time-limit-min 0.5 \
  --parallel 3 \
  --input-mode json-api \
  --json-dynamic-conss \
  --json-variable-order native-mps
```

Do not use these switches for the fair q-opt score. The fair default remains
the checked-in JSON variable order with API-default model-constraint flags.

## SCIP source basis

The behavior is explicit in the local SCIP 11 source:

- `3rd_repo/scip/src/scip/set.c` defines
  `SCIP_DEFAULT_READ_DYNAMICCONSS TRUE` and describes it as enabling constraint
  aging.
- `3rd_repo/scip/src/scip/reader_mps.c` reads `reading/dynamicconss` and passes
  it to `SCIPcreateConsLinear` for MPS rows.
- `3rd_repo/scip/src/scip/cons.c` uses dynamic constraint age to mark
  constraints obsolete and reprioritize useful constraints.

This flag is a SCIP reader policy. It is not information stored in the MPS
mathematical model and should not be interpreted as JSON conversion loss.

## Real conversion loss found and fixed

The full 13-instance audit found one separate schema defect in
`neos-3046615-murg`: `C0017` and `C0018` are MPS `FR` continuous variables.
The old converter wrote `lower: null`, which all benchmark builders interpreted
as the project default lower bound zero rather than `-infinity`.

The shared schema now uses `lower_unbounded: true` for `-infinity`, while
retaining `lower: null` as the default-zero representation. The converter and
q-opt, SCIP, OR-Tools, and HiGHS runners all use the same interpretation. The
checked-in murg JSON was regenerated accordingly. q-opt's propagation derives
finite lower bounds for these columns (`C0017 >= 0`, `C0018 >= -203`) before
canonicalization.

After the fix, the auditor reports mathematical equivalence for all 13
checked-in instances. All 13 have zero objective offset and no ranged-row
discrepancy. This is a statement about the current suite, not proof that the
JSON schema preserves every possible MPS extension such as SOS metadata.

## Reproduction

```sh
PYTHONPATH=/tmp/qopt-pyscipopt-scip11-package \
conda run -n base python bench/runners/scip_model_audit.py \
  --problems bench/problems/miplib \
  --mps-dir 3rd_repo/miplib \
  --out /tmp/scip-model-audit.json
```

The command exits nonzero if any instance differs in variable presence,
original domain, objective coefficient/sense/offset, or canonical row side.
