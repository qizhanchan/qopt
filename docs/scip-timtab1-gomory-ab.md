# timtab1 SCIP 11 Gomory causal A/B

This note records the July 18, 2026 white-box comparison that isolated
Gomory separation as a causal mechanism on `timtab1`, followed by the minimal
q-opt implementation and matched acceptance runs.

## Contract

- SCIP 11.0.0 / PySCIPOpt 6.2.1 from the dedicated local SCIP 11 package.
- SCIP input mode: JSON API. Native MPS was not used for scoring.
- Focused A/B budget: 30 seconds, `--parallel 3`.
- q-opt acceptance budget: 5 minutes, `-parallel 3`.
- All incumbents were checked against `bench/problems/miplib/timtab1.json`
  with `bench/cmd/compare` at tolerance `1e-5`.
- SCIP implementation references came only from `3rd_repo/scip`.

## SCIP causal result

The only changed parameter was:

```text
separating/gomory/freq = -1
```

| 30-second metric | SCIP default | Gomory disabled | effect of disabling |
|------------------|-------------:|----------------:|--------------------:|
| incumbent | 764772 | 764772 | unchanged |
| dual bound | 705256.58 | 640616.91 | 9.2% lower |
| gap | 7.782% | 16.234% | 2.09x worse |
| nodes | 23717 | 34107 | 43.8% more |
| LP iterations | 895699 | 1094209 | 22.2% more |

The mathematical model and presolve result were identical in both runs:

| size | original | presolved |
|------|---------:|----------:|
| columns | 397 | 201 |
| rows | 171 | 167 |
| nonzeros | 829 | 629 |

The first LP was also identical: objective `157896.036606`, 225 simplex
iterations, 101 fractional candidates with mean fractionality `0.4413`, and a
basic-at-bound degeneracy ratio of `0.0451`.

Root-only movement was modest. Default Gomory reached a final root bound of
`480226.10` in 2024 root iterations; disabling Gomory reached `479240.29` in
1834 iterations. The causal gain therefore comes mainly from sustained tree
separation, not from one exceptional root cut batch.

Default Gomory statistics were 2311 calls, 10 root calls, 30014 cuts found,
23674 cuts applied, and 2.007 seconds of separator time. With Gomory disabled,
aggregation compensated aggressively: applied aggregation cuts rose from
41935 to 75000, but did not recover the bound. Relative-pseudocost branching
also worked harder without Gomory: branch-rule LP calls rose from 13867 to
20275 and total strong-branch calls from 1810 to 2528.

At the end of root separation, the default run retained 176 active cut rows
with mean density `0.0358` and mean age `6.97`; Gomory-off retained 214 rows
with mean density `0.0280` and mean age `7.38`. This is further evidence that
the cut family, not simply the number of active rows, matters.

## Minimal q-opt mechanism

q-opt already generated valid root GMI rows, but `MaxCutRounds > 0` forced a
second solve of the original no-cut LP at every B&B node and used that clean
vertex for branching. This discarded the tree-steering effect identified in
SCIP and substantially reduced throughput.

The retained implementation treats global root cuts as the fixed relaxation:

- pruning, reliability probes, branching, and child warm starts use the
  cut-augmented LP;
- every accepted incumbent still passes `feasibleIncumbent` on the original
  model;
- symmetry rows and node-local cuts retain their existing clean-LP behavior.

## q-opt focused A/B

All runs used revised LP, reliability branching, best-estimate selection, and
a true best-bound selection every tenth choice.

| 30-second configuration | incumbent | dual bound | gap | nodes | cuts |
|-------------------------|----------:|-----------:|----:|------:|-----:|
| no cuts | 851310 | 274036.67 | 67.81% | 18788 | 0 |
| 1 round, 20 cuts, direct cut LP | 1048131 | 326850.65 | 68.82% | 14795 | 20 |
| 2 rounds, 40 cuts, direct cut LP | 1080250 | 359770.67 | 66.70% | 15577 | 40 |
| 3 rounds, 60 cuts, direct cut LP | 1115148 | 355534.95 | 68.12% | 9766 | 60 |

The root-node proof signal is much larger than the short-run final gap:

| root-only configuration | dual bound | elapsed | cuts |
|-------------------------|-----------:|--------:|-----:|
| no cuts | 28694 | 14.7 ms | 0 |
| 2 rounds | 221552 | 437.0 ms | 40 |

The root relaxation bound improved by 7.72x for about 0.42 seconds of root-cut
work.

## Five-minute acceptance

| metric | no cuts | 2 GMI rounds | change |
|--------|--------:|-------------:|-------:|
| incumbent | 811339 | 783290 | 3.46% better |
| dual bound | 349632.00 | 418646.94 | 19.74% stronger |
| gap | 56.91% | 46.55% | 18.19% reduction |
| nodes | 160999 | 178010 | 10.57% more |
| LP iterations | 11280855 | 8555651 | 24.16% fewer |

Both reported incumbents passed the independent original-JSON verifier. The
candidate does not yet approach SCIP's proof, but it meets the milestone gate
through a clearly stronger root relaxation at modest cost and improves both
the incumbent and proof bound at five minutes.

## Negative findings

- Four root cuts were too weak; their 30-second bound fell below the no-cut
  baseline.
- Three 20-cut rounds crossed the cost/strength optimum and reduced throughput.
- Solving a clean original LP at the root and every 128 nodes did not improve
  the incumbent and cut node throughput by more than half.
- Existing per-node GMI with row aging remains too expensive because changing
  row sets cause warm-start failures and repeated Phase I work. It is not part
  of the accepted mechanism.

Machine-readable summary: `bench/results/timtab1-gomory-parity-20260718.json`.
