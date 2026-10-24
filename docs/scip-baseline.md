# SCIP reference baseline

This file defines the SCIP build and input path used for q-opt performance
comparison. It is intentionally separate from the active solver roadmap.

## Reference build

Current reference as of July 17, 2026:

- SCIP 11.0.0
- SoPlex 8.0.3
- optimized build (`-O2 -g -DNDEBUG`)
- PySCIPOpt 6.2.1 with the local SCIP 11 compatibility patch
- conda environment: `base`

The patch is stored at
[`pyscipopt_scip11.patch`](https://github.com/qizhanchan/qopt/blob/main/bench/runners/pyscipopt_scip11.patch).
The local SCIP source is under `../3rd_repo/scip`; use it instead of web search
for implementation details and white-box instrumentation.

## Fair input mode

The matched baseline must use `--input-mode json-api`, which reconstructs the
same converted JSON model consumed by q-opt. `--input-mode native-mps` reads
the original MIPLIB MPS and is reserved for source diagnostics.

The distinction matters. Variable and row insertion order changes SCIP's
presolve and search trajectory; exporting the API-built model to MPS and
reading it back can materially change node throughput, incumbent timing, and
proof time even though the mathematical model is equivalent.

The direct model-equivalence audit and the causal constraint-metadata/order
A/B are recorded in
[`scip-json-model-audit.md`](scip-json-model-audit.md). For `neos17` and
`timtab1`, the objective, domains, and complete row matrix are equivalent; the
observed trajectory difference is explained by SCIP's MPS-reader
`dynamic=true` constraint policy plus its internal variable order.

## Reproduction

```sh
conda run -n base python bench/runners/scip_miplib_baseline.py \
  --problems bench/problems/miplib \
  --out /tmp/scip11-miplib-5m.json \
  --time-limit-min 5 \
  --parallel 3 \
  --input-mode json-api
```

The runner records linked SCIP and binding versions separately and writes
per-instance SCIP statistics JSON for white-box comparison.

## Five-minute result

The corrected three-concurrent run produces incumbents on all 13 instances
and proves four.

| instance | objective | gap/status | result |
|----------|----------:|------------|--------|
| `markshare_4_0` | 1 | optimal in 139.1s | proven |
| `pk1` | 11 | optimal in 112.4s | proven |
| `neos17` | 0.150003 | optimal in 13.8s | proven |
| `timtab1` | 764772 | optimal in 48.9s | proven |
| `gen-ip002` | -4783.73 | 0.408% | optimum incumbent found |
| `gen-ip054` | 6840.97 | 0.599% | optimum incumbent found |
| `assign1-5-8` | 212 | 4.63% | incumbent |
| `neos5` | 15 | 1.67% | optimum incumbent found |
| `ic97_potential` | 3949 | 0.93% | incumbent |
| `mad` | 0.072 | time limit | incumbent |
| `neos-3046615-murg` | 1628 | time limit | incumbent |
| `neos-3754480-nidda` | 13513.9 | time limit | incumbent |
| `glass4` | 1.70e9 | 50.79% | incumbent |

Use the machine-readable result file, not rounded values in this table, for
future scoring.

## SCIP 10/11 white-box finding

A matched 30-second comparison showed that SCIP 11 is not generally slower.
On `pk1`, `markshare_4_0`, and `timtab1`, SCIP 11 processes roughly 19-34%
more nodes and LP iterations and applies more aggregation/Gomory cuts,
producing stronger bounds.

`neos17` is the notable strategy change: SCIP 10 presolves 535 variables to
300 and proves in 437 nodes, while SCIP 11 keeps 511 variables and uses 4116
nodes. Both still prove in under 15 seconds. Future q-opt work should compare
phase metrics and plugin ablations rather than treating version-level totals
as one algorithmic mechanism.

## White-box workflow

For a target instance:

1. Run SCIP with JSON API input and detailed statistics.
2. Record presolve reductions, root LP, separator families, branch decisions,
   heuristic incumbents, bound timeline, nodes, and LP iterations.
3. Disable one plugin family at a time to establish causality.
4. Read the corresponding implementation under `3rd_repo/scip/src/scip`.
5. Add matching q-opt telemetry before implementing a reduced mechanism.
6. Accept the change only after a matched q-opt A/B and original-model
   feasibility verification.

The first targets are `neos17` and `timtab1`, because SCIP proves both quickly
while q-opt already has feasible incumbents but weak proof progress.

## timtab1 Gomory ablation

The first single-family causal result is now established. On the fair JSON API
path at 30 seconds, setting `separating/gomory/freq=-1` keeps the incumbent at
764772 but weakens the dual bound from 705256.58 to 640616.91 and worsens the
gap from 7.78% to 16.23%. Presolve and the first LP are identical, while the
final root bound moves by only about 0.2%; the material effect is sustained
tree separation. See
[`scip-timtab1-gomory-ab.md`](scip-timtab1-gomory-ab.md) for cut-family,
branching, root, incumbent, and q-opt acceptance telemetry.

## neos17 conflict ablation

The `neos17` single-family result is infeasible-LP conflict analysis. Turning
off only `conflict/useinflp` raises proof time from 8.65 to 11.99 seconds and
nodes from 4116 to 9248. Turning off SCIP conflict analysis entirely is worse
at 14.41 seconds and 7949 nodes. In contrast, setting
`branching/relpscost/conflictweight=0` improves the proof to 7.17 seconds and
1033 nodes, so conflict-derived branch scoring is not causal.

The corresponding minimal q-opt certificate no-goods are correct and useful
as instrumentation, but miss the acceptance gate. SCIP upgrades conflict rows
to logicor/setppc constraints and repropagates resolved reasons; directly
adding dynamic conflict rows to q-opt's LP is currently far too expensive.
See [`scip-neos17-conflict-ab.md`](scip-neos17-conflict-ab.md).
