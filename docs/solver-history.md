# Solver experiment history

This document retains conclusions that are useful for avoiding repeated work.
It intentionally omits the chronological command log that previously lived in
`README.md` and `solver-milestones.md`.

## Benchmark discipline

- Use fixed per-instance budgets and problem-level concurrency.
- Separate an isolated target run from the full concurrent suite when a first
  incumbent appears near the time limit.
- Treat objective, bound, gap, nodes, LP iterations, first-solution source,
  and incumbent timeline as a single result; final objective alone is not
  enough to explain causality.
- Verify incumbents against the original model after every search or numerical
  change.
- Prefer local SCIP source and machine-readable logs over README claims.
- Keep an experiment only when a matched A/B demonstrates the intended effect.
- The SCIP JSON/native-MPS trajectory difference on `neos17` and `timtab1` is
  caused by constraint aging metadata and variable order, not mathematical
  model loss; see [`scip-json-model-audit.md`](scip-json-model-audit.md).
- SCIP 11 infeasible-LP conflict analysis is causal on `neos17`, but its
  conflict contribution to relative-pseudocost scoring is not; setting the
  branch-rule conflict weight to zero improves the proof trajectory. See
  [`scip-neos17-conflict-ab.md`](scip-neos17-conflict-ab.md).

## Primal search conclusions

Accepted:

- Mixed-integer feasibility pump and deterministic repair provide broad root
  coverage.
- Coefficient diving is the decisive first-solution mechanism on `glass4`.
- Reliability branching is essential on `timtab1`, `ic97_potential`, `pk1`,
  and `markshare_4_0`.
- Best-estimate ordering materially improves short-budget incumbent coverage.
- Local branching reliably reaches `assign1-5-8=212` with radius 18.
- Proximity search gives repeatable gains on `murg`; adaptive RINS can also
  improve it with a different neighborhood.
- A reward-per-time neighborhood scheduler avoids repeatedly paying for weak
  sub-MIP calls and preserves `neos5=15` in the current portfolio.

Rejected or deliberately not default:

- Global coordinate repair consumed seconds without solving the hard targets.
- Complete round-and-fix assignments were commonly presolve-infeasible.
- Small local-branching neighborhoods excluded known feasible solutions; the
  measured `assign1-5-8` improvement has exact Hamming distance 18.
- Broad RENS variants did not improve the targeted mixed-integer instances.
- Static neighborhood fixing-rate thresholds could not separate successful
  `murg` calls from unsuccessful `nidda` calls.
- Repeating proximity/RINS calls can improve the sub-MIP objective while
  worsening the final main-tree incumbent by consuming search time.

## LP engine conclusions

Accepted:

- Sparse revised simplex is the default; the tableau is a reference path.
- Cross-node canonicalization and scaling reuse removes substantial per-node
  rebuild cost.
- Sparse LU produces the major throughput gain on large sparse instances.
- Warm basis remapping across row-sign flips prevents unnecessary cold Phase I
  solves, especially on `ic97_potential`.
- Exponential backoff for futile dense rescues avoids repeated expensive work
  on near-singular `nidda` bases.
- Refactor when eta-chain work reaches factor work; a fixed long eta chain was
  expensive on `glass4`.
- Sparse-to-dense fallback, cold verification, Phase I retry, and non-root
  numerical-failure containment are still required correctness safeguards.
- The certified warm-reopt shortcut is bounded (July 21, 2026): dual
  feasibility is re-verified every 8th consecutive certificate-trusted
  accept, and a failed check returns the vertex with a Neumaier–Shcherbina
  bound correction instead of a primal resume (resume only past a 1e-4
  relative cap, or on infinite-range violations). Unlimited chains accepted
  dual-infeasible vertices on 11 of 13 instances and falsely "proved" pk1
  at 12; the cadence keeps 13/13 incumbents and both proofs at roughly
  10–25% node-throughput cost. See scip-pipeline-parity.md, Stage 4c
  follow-up.
- Infeasible warm-dual exits are certified by default (July 21, 2026) with
  a SCIP-shape aggregate Farkas validation (conclusion-checked, no
  basis-residual precondition, candidacy-tolerance coefficients treated as
  zero) instead of a cold Phase I re-proof. The old derivation-gated
  certificate hit 2 of 376 candidates on neos17; the rewrite hits 100%
  everywhere it fires, halves neos17's Phase I time, and delivers 4-5x
  nodes on neos17/timtab1/murg with no regression, node-identical retained
  proofs, and an unchanged toy suite. Disable switch:
  RevisedNoCertifyInfeasible. See scip-pipeline-parity.md, Stage 4d.
- Reliability probes now use a native strong-branch LP session (July 21,
  2026): every side starts from one immutable dual-feasible parent basis,
  iteration-limit estimates are separated from valid child bounds, certified
  infeasibility is the only native prune, and probe bases are discarded.
  SCIP's dynamic 10..500 per-side cap is retained, but q-opt's measured
  1600-iteration offset and eight-candidate width remain because the raw
  SCIP 100k/100-candidate constants over-invested and regressed primal search.
  The 30-second suite records 35,221 native sides, 10,531 valid capped bounds,
  and zero uncertified prunes; `markshare_4_0` and `pk1` still prove, now in
  39.6s and 72.4s in the five-minute profile. See scip-pipeline-parity.md,
  Stage 4e.
- Inherited global root cuts can use the tree RowSet lifecycle (July 21,
  2026): root rows keep stable slots, path-local ages can deactivate them,
  pooled rows are retested and reactivated, active rows participate in bound
  propagation, and the dynamic cache reuses the complete cut model's Ruiz
  scales. Keep this infrastructure opt-in through `CutRowAgeLimit`. After
  scaling/propagation parity, the age-10 policy regressed `assign1-5-8` and
  `neos17` and produced only a small `timtab1` gap improvement; hard active
  caps and 100/500-node warmups were also rejected.
  See scip-pipeline-parity.md, Stage 4f.
- Dual-proof conflict pool (July 22, 2026, opt-in `UseDualProofs`): certified
  Farkas exits export their proof row as a bound-independent model-space
  inequality; a 100-entry pool prunes later nodes/probe sides whose boxes
  provably violate a pooled proof, before any LP work, and proofs never
  enter an LP (SCIP's default dual-ray posture, conflict_dualproofanalysis.c;
  the July 18 rejection was branch-path no-goods, a different mechanism).
  Sound (0 of 8027 shadow-verified prunes contradicted; toy suite and both
  five-minute proofs node-identical), but the matched suite is mixed: neos17
  bound 0.0992→0.1000 and murg bound 345→383 (2.6x nodes) improve while
  timtab1's incumbent and bound regress at both budgets — the promotion gate
  is not met and the flag stays off. Next slice: size-1 proofs as global
  bound changes with repropagation, long-proof one-shot propagation, and
  probe-side pooling. See scip-pipeline-parity.md, Stage 4g.

Operational caution (July 22, 2026): harness background tasks run
QoS-throttled on macOS (~28% of one core for a `-parallel 3` suite) and
silently lose proofs; promotion-grade suites must run in the foreground.

Measured non-levers:

- Factorization persistence reduces refactor counts but lowers throughput due
  to copied eta chains and memory cost.
- Forrest-Tomlin/Bartels-Golub was not justified by profiles: factorization is
  not the dominant cost on the current suite.
- Full Dantzig pricing is a small CPU share; Devex/partial pricing is not a
  current priority.
- Allocation removal in pivot loops is useful infrastructure but was neutral
  in wall-clock A/B tests.
- Harris two-pass ratio tests changed degenerate vertices and caused incumbent
  loss on `glass4`/`ic97_potential`; keep experimental only.

## Cut and structural conclusions

Correct infrastructure now exists for root and per-node GMI/MIR cuts on the
revised backend, including native revised-basis GMI generation. Removing the
tableau re-solve did not make cuts competitive: the enlarged active LP, not
cut generation itself, is the measured bottleneck.

Accepted findings:

- Root cover cuts reduce `markshare_4_0` proof nodes.
- Verified partitioning-orbitope fixings improve the `assign1-5-8` bound.
- Strongly gated big-M replacement improves `glass4` throughput.
- Weak but valid big-M changes can alter incumbent trajectories; require a
  strength gate.
- `nidda` exposes many variable-bound implications, but current implied-bound
  and minimal flow-cover cuts do not move its root relaxation.
- Glass4 can have violated implied-bound cuts that move only between
  degenerate vertices; require root-bound movement before attaching them to
  the tree.
- SCIP 11 Gomory separation is causal on `timtab1`: disabling only that
  separator doubles the 30-second gap while leaving the incumbent unchanged.
  The gain is tree-wide rather than root-only.
- Global q-opt root GMI rows should drive branching directly. The previous
  clean-LP solve at every node discarded the strengthened vertex and reduced
  throughput. Two 20-cut rounds improve the `timtab1` root bound 7.72x and the
  matched five-minute gap from 56.91% to 46.55%.
- Phase-I infeasibility certificates can safely minimize tree-decision
  no-goods and support pre-LP prune/unit propagation. Keep this infrastructure
  experimental: the current no-good form does not reproduce SCIP's
  reason-resolved logicor/setppc propagation on `neos17`.
- Exact continuous equality aggregation reduces `timtab1` from
  `397/171/829` to `226/168/651` columns/rows/nonzeros. Root GMI generation
  can now map aggregated canonical slack terms back to original variables, but
  the measured aggregation-plus-GMI schedules worsen the primal trajectory.
- Position-stable empty cut rows preserve revised-simplex warm-start shape and
  remove stale coefficient work. This materially improves mechanical node
  throughput, but the tested tree-wide schedules have not improved the matched
  `timtab1` proof trajectory.

Negative results to avoid repeating:

- Root/per-node cuts without active-row control can reduce B&B throughput by
  orders of magnitude.
- More cut rounds, larger per-node caps, and cheaper generation do not solve
  the active-LP cost problem.
- Single-row MIR does not explain the `markshare_4_0` proof; the earlier claim
  was a misattribution.
- Simple single-column symmetry verification finds no useful orbits on the
  suite. `assign1-5-8` symmetry is block-level and needs a real
  individualization-refinement search.
- Reduced-cost fixing, pseudo-objective propagation, and fixed-weight hybrid
  estimate ordering regressed incumbent quality or throughput in matched runs.
- Periodic clean-LP refresh for root-cut primal search did not recover a better
  `timtab1` incumbent and more than halved 30-second node throughput. A third
  20-cut root round also crossed the measured cost/strength optimum.
- Injecting learned conflicts as dynamic LP rows is prohibitive with the
  current revised-LP row-set path: even two short rows more than doubled LP
  iterations on the fixed 300-node `neos17` test. Deeper path no-goods also
  propagate more while producing weaker proof bounds.
- On `neos17`, 371 of 372 warm dual no-entering states cannot be certified by
  the current single tableau-row proof because a helpful slack/surplus
  direction has infinite width. Lowering the post-failure pivot threshold did
  not expose additional entering candidates; a reason-resolved conflict or a
  fuller Farkas proof is required.

## Search conclusions

- Reliability branching plus sparse LP can prove `markshare_4_0` and `pk1`
  by enumeration; cuts are not required for the first two proofs.
- Pure best-estimate improves primal search but can neglect the proof frontier.
- Selecting the true best LP-bound node every tenth selection is the current
  useful balance. The accepted five-minute A/B improved nine gaps, left three
  unchanged, and regressed one (`ic97_potential`).
- A one-way switch from best-estimate to pure best-bound after an incumbent
  improvement is useful on `timtab1`. Combined with two root-GMI rounds it
  reduces the focused five-minute gap from 46.55% to 34.85%, while pure
  best-bound from the root fails to find a 30-second incumbent. See
  [`qopt-timtab1-proof-switch.md`](qopt-timtab1-proof-switch.md).
- DFS bound reporting is conservative and can be loose; proof comparisons
  should prefer selectors with an exact open-frontier bound.
- `gen-ip002` and `gen-ip054` now have optimum incumbents and sub-1% gaps. The
  remaining tail is proof strength, not primal coverage.

## Historical milestone outcomes

Milestone 1 closed primal coverage: all 13 instances can produce feasible
incumbents, with concurrency sensitivity resolved by later LP work.

Milestone 2 met its original throughput target on `ic97_potential` and
`nidda`; `glass4` improved substantially in LP-iteration throughput but still
has deeper expensive relaxations.

Milestone 3 built useful structural infrastructure but did not meet its bound
acceptance target on `glass4` and `nidda`. It remains open.

The post-Milestone-3 search work reached 10/10 short-budget incumbent coverage
on the initially selected targets, then 13/13 at five minutes. The adaptive
neighborhood profile currently preserves two proofs and improves
`assign1-5-8`, `murg`, and `mad`, but remains opt-in.

## Reference locations

- Active roadmap: [`solver-milestones.md`](solver-milestones.md)
- Current SCIP baseline: [`scip-baseline.md`](scip-baseline.md)
- Detailed solver telemetry and runner flags: [benchmark harness README](https://github.com/qizhanchan/qopt/blob/main/bench/README.md)
- Local SCIP source: `../3rd_repo/scip`
