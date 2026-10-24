# neos17 SCIP 11 conflict causal A/B

This note records the July 18, 2026 white-box comparison that isolated
infeasible-LP conflict analysis on `neos17`, followed by the minimal q-opt
implementation and matched rejection runs.

## Contract

- SCIP 11.0.0 / PySCIPOpt 6.2.1 from the dedicated local SCIP 11 package.
- SCIP scoring input: JSON API. Native MPS was used only for diagnostics.
- Focused A/B budget: 30 seconds, `--parallel 3`.
- q-opt focused budget: 30 seconds or a fixed 300 nodes, `-parallel 3`.
- All accepted incumbents passed the benchmark original-model verifier.
- SCIP implementation references came only from `3rd_repo/scip`.

The JSON/API model is not missing mathematical information. The independent
audit proves identical objective, variable domains, row sides, coefficients,
and row order against native MPS. SCIP trajectory differences come from the
MPS reader's dynamic-constraint policy and internal variable order; see
[`scip-json-model-audit.md`](scip-json-model-audit.md).

## SCIP white-box baseline

| phase metric | value |
|--------------|------:|
| original columns / rows / nonzeros | 535 / 486 / 4931 |
| presolved columns / rows / nonzeros | 511 / 488 / 2506 |
| first LP objective | 0.0006814985 |
| first LP iterations | 1200 SCIP statistics, 1464 event snapshot |
| fractional candidates / sum / maximum | 171 / 0.43756 / 0.007992 |
| first-LP basic-at-bound degeneracy | 120 / 435 = 27.59% |
| final root objective | 0.11774217 |
| root separation rounds / cuts applied | 89 / 477 |
| final root rows / nonzeros | 681 / 3377 |
| active cut rows / mean age / maximum age | 194 / 1.876 / 50 |
| active cut mean / maximum density | 0.008806 / 0.04110 |
| final root fractional candidates / mean | 148 / 0.42625 |

The default run proves objective `0.1500025774` in 8.65 seconds, 4116 nodes,
and 44113 non-strong-branch LP iterations. Relative pseudocost branching makes
2287 LP calls and creates 433 constraints plus 639 domain reductions. Strong
branching accounts for 1213 calls and 55474 additional iterations.

The first solution is found at the root after 0.034 seconds. The incumbent
timeline is dominated by simple/zero rounding and strong-branch solutions,
with diving and neighborhood heuristics contributing later improvements.

## SCIP plugin A/B

Each row changes only the named plugin family or parameter.

| configuration | seconds | nodes | LP iterations | final root objective | reading |
|---------------|--------:|------:|--------------:|---------------------:|---------|
| default | 8.65 | 4116 | 44113 | 0.117742 | reference |
| Gomory off | 5.10 | 1719 | 22493 | 0.121376 | Gomory hurts alone |
| aggregation off | 8.89 | 1164 | 108942 | 0.040981 | fewer nodes, expensive LPs |
| aggregation and Gomory off | 12.21 | 10787 | 146282 | 0.040981 | strong family interaction |
| all separation off | 9.07 | 7700 | 80898 | 0.000681 | cuts collectively matter |
| relative pseudocost off | 8.61 | 3318 | 124046 | 0.117742 | branch rule is not singular |
| presolve off | 6.99 | 3280 | 46862 | 0.099741 | default presolve not causal |
| propagation off | 9.45 | 2688 | 76969 | 0.101333 | more LP work, no proof loss |
| all conflict analysis off | 14.41 | 7949 | 177145 | 0.097803 | conflict analysis is causal |
| conflict strong-branch input off | 6.89 | 2853 | 38502 | 0.117742 | this input hurts trajectory |
| infeasible-LP conflict off | 11.99 | 9248 | 115137 | 0.098688 | strongest clean causal result |
| bound-exceeding LP conflict off | 7.45 | 4023 | 66667 | 0.117742 | not the important source |
| relpscost conflict weight zero | 7.17 | 1033 | 38866 | 0.117742 | conflict branch score hurts |

Disabling only infeasible-LP analysis materially worsens proof time and tree
size. The zero conflict-weight A/B rules out copying SCIP's VSIDS contribution
to relative-pseudocost scoring as the minimal mechanism.

Default conflict statistics include 174 infeasible-LP calls with 166
successful analyses, 503 bound-exceeding calls with 278 successes, and 54
propagation analyses with 27 successes. The generated constraints are active
in specialized handlers: logicor reaches 382 active constraints, 45 cutoffs,
and 1110 domain reductions; setppc reaches 168 active constraints, 7 cutoffs,
and 921 domain reductions; bounddisjunction contributes 49 domain reductions.
Repropagation performs 481 calls, 9 cutoffs, and 1147 domain reductions.

## Local SCIP source reading

`3rd_repo/scip/src/scip/conflict_resolution.c` first creates a linear conflict
row and then calls `SCIPupgradeConsLinear`. Binary conflicts can therefore
become logicor or setppc constraints instead of permanent LP rows. Conflict
analysis also records resolution reasons and schedules repropagation.

`3rd_repo/scip/src/scip/branch_relpscost.c` combines pseudocost, inference,
cutoff, and conflict terms. Its default conflict weight is `0.01`, but the A/B
above shows that term is harmful on this instance even though conflict
constraints themselves are useful.

## Minimal q-opt mechanism

The retained experimental implementation adds:

- a Phase-I infeasibility certificate with duals, reduced costs, auxiliary
  contribution, orientation, and tolerance;
- certificate re-evaluation under relaxed branch bounds;
- integral bound literals for binary and general-integer decisions;
- no-good deduplication, subsumption, pre-LP pruning, and unit propagation;
- certificate-guided deletion filtering of irrelevant decisions;
- explicit telemetry for learned literals, prunes, propagations, tests, and
  certificate proofs.

It is opt-in through `UseLPConflictLearning` / `-lp-conflicts`. The default
maximum conflict size is 12 decisions. Incumbents continue to be checked on
the original model.

## q-opt focused A/B

| configuration | budget | incumbent | dual bound | gap | nodes | LP iterations |
|---------------|--------|----------:|-----------:|----:|------:|--------------:|
| control | 300 nodes | 0.2219191 | 0.0336076 | 18.831% | 300 | 103149 |
| certificate no-goods | 300 nodes | 0.2008509 | 0.0318180 | 16.903% | 300 | 106968 |
| control | 30 seconds | 0.2094093 | 0.0462717 | 16.314% | 596 | 197250 |
| certificate no-goods | 30 seconds | 0.2008509 | 0.0391043 | 16.175% | 494 | 204471 |

At 30 seconds the candidate learns 19 conflicts containing 53 literals,
prunes 8 nodes, performs 52 unit propagations, and proves 92 of 135 attempted
certificate reductions. Its final gap improves by only 0.14 percentage point,
while its dual bound and node throughput are worse. This does not meet the
acceptance gate.

The same option was rejected on the five-minute `timtab1` cross-check: the
control gap is 57.41%, while maximum-size-20 conflicts produce 57.95% with a
worse incumbent, weaker dual bound, and 24% fewer LP iterations completed.

## Negative findings

- Maximum conflict sizes 20 and 32 generate many more propagations but weaken
  the fixed-300-node `neos17` proof bound and final gap.
- Adding all learned conflicts as dynamic LP rows raises the fixed-300-node
  work from 103149 to 536238 iterations and elapsed time from 16.1 to 92.1
  seconds.
- Adding only conflicts with at most three literals still needs 400252 LP
  iterations and 44.6 seconds. The row-set change and repeated Phase I work,
  not row density, dominate the cost.
- Expanding tree decisions with the current inferred local bounds reproduces
  the same no-goods and search trajectory, so it is not reason resolution.

## Decision

The white-box target is explained, but q-opt has not reproduced the accepted
mechanism. Keep the certificate/no-good code and telemetry default-off. Do not
add conflict VSIDS scoring, deep path no-goods, or dynamic LP conflict rows.

The next implementation priority is item 2 of the roadmap: cheap incremental
row activation/deactivation and row aging. In parallel, structural learning
needs explicit propagation reasons so binary conflicts can become compact
logicor/set constraints with repropagation rather than raw branch-path
no-goods.

Machine-readable summary:
`bench/results/neos17-conflict-parity-20260718.json`.
