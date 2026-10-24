# q-opt timtab1 proof scheduling and cut lifecycle

This note records the July 18, 2026 focused q-opt experiments that followed
the SCIP 11 Gomory ablation. All runs use the JSON API model and every reported
incumbent passed the original-model verifier. The machine-readable summary is
[`timtab1-proof-switch-20260718.json`](https://github.com/qizhanchan/qopt/blob/main/bench/results/timtab1-proof-switch-20260718.json).

## Mechanisms implemented

- Exact continuous equality aggregation now composes with global root cuts.
  Canonical slack/surplus terms are expanded through the aggregated original
  row before a GMI cut is mapped back to model variables.
- Aged cut rows become empty redundant row slots rather than RHS-relaxed rows.
  Row order and auxiliary-column positions remain stable, so a parent basis can
  be retargeted and refactored when a cut is deactivated or reactivated.
- Best-estimate search can switch once to pure best-bound when a newly improved
  incumbent brings the exact open-frontier gap below a configured threshold.
- The benchmark output records incumbent and global-bound timelines, including
  elapsed time, processed nodes, LP iterations, and the authoritative bound.
- Warm dual-infeasibility certificates report why a candidate proof was
  rejected instead of silently falling back to cold Phase I.

## timtab1 focused results

| configuration | budget | incumbent | dual bound | gap | nodes | LP iterations |
|---------------|-------:|----------:|-----------:|----:|------:|--------------:|
| equality aggregation | 5 min | 793315 | 415654 | 47.61% | 869664 | 5047976 |
| aggregation + adaptive proof switch | 5 min | 817894 | 503516 | 38.44% | 627008 | 5002528 |
| two root GMI rounds + adaptive proof switch | 5 min | 793168 | 516737 | 34.85% | 235162 | 8250430 |

The last run generated 40 root GMI cuts in two rounds. Its first incumbent was
`1246822` at 25.6 seconds and node 6215. Rounding found `793168` at 107.7
seconds and node 68583; the queue switched to pure best-bound at node 68584
with a 47.29% exact frontier gap. The dual bound then progressed from roughly
`419794` at 108 seconds to `452582` at 120 seconds, `490470` at 180 seconds,
and `516737` at the five-minute limit.

Relative to the previously retained two-round root-GMI result at 46.55%, the
34.85% gap is a 25.1% reduction. It is a real proof-scheduling gain, but it is
still far from SCIP's 48.9-second proof and does not raise the suite score
above 2/13 optimal.

## Negative results

- Exact equality aggregation alone improves LP shape and raw tree throughput,
  but does not reproduce SCIP's root relaxation or proof trajectory.
- Aggregation plus 4 or 20 mapped root GMI cuts preserved all 171 eliminations
  but produced substantially worse 30-second incumbents.
- Empty-slot cut aging removes stale matrix work and increases node throughput,
  but the tested sustained per-node GMI schedules steered `timtab1` toward
  worse incumbents. This is not yet an accepted tree-separation policy.
- A pure best-bound run from the start raised the bound quickly but found no
  incumbent in 30 seconds. The one-way switch is useful because it preserves a
  primal-search phase before committing the remaining budget to proof.

## neos17 certificate diagnosis

In the 30-second focused run, warm dual reoptimization reached 372 no-entering
states. Only one bounded tableau row produced a certificate; 371 candidates
were rejected because a helpful slack/surplus direction had infinite width.
Lowering the pivot threshold after no-entering found no additional candidates.
The current single-row certificate therefore cannot replace SCIP's
reason-resolved infeasible-LP conflict analysis.

## Next experiment

Keep the accepted root GMI and one-way proof switch as the `timtab1` reference.
The next cut experiment must improve bound progress without delaying the second
incumbent: separate a very small number of high-efficacy local GMI rows only
after the proof switch, use empty-slot aging, and compare bound gain per LP
iteration against the root-only run. Stop immediately if the 30/60-second
incumbent or bound trajectory regresses.
