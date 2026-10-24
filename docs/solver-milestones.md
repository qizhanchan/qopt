# q-opt solver milestones

This file is the active engineering roadmap. It contains current conclusions,
next work, and acceptance gates. Historical A/B details are kept in
[`solver-history.md`](solver-history.md); the current SCIP reference is in
[`scip-baseline.md`](scip-baseline.md).

## Target

Build a strong pure-Go linear LP/MIP solver whose five-minute behavior on the
checked-in 13-instance MIPLIB subset approaches the optimized SCIP 11
baseline. The immediate target is single-threaded algorithmic parity; parallel
B&B comes after the search and LP work is measurable and stable.

The benchmark contract is fixed:

- use the same JSON model construction path for q-opt and SCIP;
- run suites with `-parallel 3`;
- use a five-minute per-instance budget for acceptance;
- retain 30-second focused runs for iteration speed only;
- verify every incumbent against the original JSON model;
- compare objective, valid dual bound, relative gap, proof status, node/LP
  throughput, and the first-solution mechanism;
- keep defaults conservative; experimental wins remain opt-in until the full
  suite shows no material regression.

## Current scorecard

Current recommended q-opt profile:

```sh
/opt/homebrew/bin/go run ./bench/cmd/qopt_bench \
  -problems bench/problems/miplib \
  -out /tmp/qopt-miplib-5m.json \
  -time-limit 5m -parallel 3 \
  -lp-backend revised \
  -branching reliability \
  -node-selection best-estimate \
  -best-estimate-best-bound-freq 10 \
  -rins -rins-adaptive-budget \
  -local-branching -proximity-search
```

At five minutes:

- q-opt: 13/13 verified incumbents, 2/13 proven optimal;
- SCIP 11: 13/13 incumbents, 4/13 proven optimal;
- q-opt is near SCIP on `gen-ip002` and `gen-ip054`;
- q-opt reaches the same incumbent as SCIP on `assign1-5-8` and `neos5`, but
  has weaker proof bounds;
- the largest gaps remain `glass4`, `ic97_potential`, `murg`, `nidda`,
  `neos17`, and `timtab1`.

| target group | q-opt position | interpretation |
|--------------|----------------|----------------|
| `markshare_4_0`, `pk1` | both proven | search kernel is capable of complete proofs |
| `gen-ip002`, `gen-ip054` | optimum incumbents, about 0.4-0.6% gap | close; improve proof tail without losing throughput |
| `assign1-5-8`, `neos5` | SCIP-quality incumbents, weaker bounds | cut/presolve/proof problem |
| `neos17`, `timtab1` | feasible incumbents, SCIP proves quickly | highest-value white-box targets |
| `murg`, `nidda` | weak incumbents and very weak bounds | primal and relaxation both need work |
| `glass4`, `ic97_potential` | numerically/structurally expensive | LP, presolve, and heuristic efficiency problem |

## Completed milestones

### Milestone 1: primal coverage

Status: complete.

The solver has event-driven propagation, mixed-integer feasibility pump,
repair, coefficient and pseudocost diving, RENS, and a bounded heuristic
scheduler. Reliability branching accounts for infeasible sides and reuses
probe information. All 13 MIPLIB instances can produce verified incumbents.

The accepted mechanisms were coefficient diving for `glass4`, reliability
search for `timtab1`/`ic97_potential`, and mixed-integer pump/repair for the
remaining coverage gaps. Broad coordinate repair and speculative small
neighborhoods were rejected when matched runs showed cost without solutions.

### Milestone 2: LP throughput and stability

Status: complete for the original acceptance target.

The main accepted changes were:

- incremental revised-simplex solves with cross-node canonicalization reuse;
- sparse LU by default with dense fallback;
- warm-start remapping across row-sign/standard-form flips;
- backoff for futile dense rescues on intrinsically near-singular bases;
- cost-based eta refactor cadence;
- allocation-free FTRAN/BTRAN scratch paths and detailed phase telemetry.

These changes closed the concurrent 13/13 incumbent-coverage gap and produced
large throughput gains on `ic97_potential` and `nidda`. Harris ratio tests,
factor persistence, FT/BG work, and partial pricing were not promoted because
profiles or matched runs did not justify them.

### Milestone 3: structural bound strengthening

Status: partially complete; acceptance not met.

Implemented infrastructure includes binary conflicts, verified partitioning
orbitopes, gated big-M tightening, variable-bound implications, root probing,
knapsack covers, implied-bound cuts, a minimal flow-cover separator, and
inference-aware branching telemetry.

Useful results:

- cover cuts shorten the `markshare_4_0` proof;
- orbitope fixings improve the `assign1-5-8` bound;
- strong big-M tightening improves `glass4` throughput;
- implication extraction fires heavily on `nidda`.

The milestone remains open because these mechanisms do not materially close
the five-minute bound on `glass4` or `nidda`, and some valid formulation
changes alter incumbent trajectories. Do not expand the existing cut families
without a measured root-bound or matched-time gain.

### Search and neighborhood parity

Status: useful opt-in profile, not a completed parity milestone.

Best-estimate node selection plus a true best-bound step every tenth selection
is the current proof-oriented search profile. It improves nine of 13 gaps in
the matched five-minute comparison and preserves the two proofs.

Adaptive RINS, local branching, and proximity search are coordinated by a
small reward-per-time scheduler when enabled together. Current five-minute
gates include:

| instance | incumbent | dual bound / gap | mechanism |
|----------|----------:|-----------------:|-----------|
| `assign1-5-8` | 212 | 194.191 / 8.40% | local branching |
| `neos-3046615-murg` | 1782 | 354 / 80.13% | proximity search |
| `neos5` | 15 | 14 / 6.67% | scheduler avoids repeated sub-MIP cost |

The portfolio remains off by default because it optimizes primal quality and
can displace proof work.

## Next work

Work in this order. Each item has a stop condition so experiments do not turn
into permanent speculative complexity.

The implementation order below is governed by the staged pipeline plan in
[`scip-pipeline-parity.md`](scip-pipeline-parity.md). A partial SCIP mechanism
is not rejected solely because it lacks end-to-end gap gains before its
numerical cleanup, iterative separation, cut-pool, or propagation dependencies
exist. It remains experimental until the full pipeline passes promotion.

### 1. White-box SCIP parity on `neos17` and `timtab1`

Goal: explain why SCIP 11 proves both quickly while q-opt does not.

Measure the same phase counters in both solvers:

- original and presolved rows/columns/nonzeros;
- root LP objective, iterations, degeneracy, and fractionality;
- separator rounds, accepted cuts by family, efficacy, density, and aging;
- branch rule, strong-branch work, pseudocost reliability, and child order;
- incumbent timeline and heuristic source;
- node count, LP iterations, global bound timeline, and proof tail.

Use the local source under `3rd_repo/scip`, especially aggregation/Gomory
separators, relative pseudocost branching, node estimate selection, presolve,
and propagation. Add temporary SCIP plugin-disable A/B runs to identify which
families are causally responsible; do not infer causality from aggregate logs.

Acceptance gate: identify one SCIP mechanism whose removal materially worsens
one target, reproduce its relevant telemetry in q-opt, and obtain either a
twofold proof-time/gap improvement or a clearly stronger root relaxation at a
matched cost. If no single mechanism dominates, retain the phase breakdown and
move to item 2 rather than copying the full SCIP portfolio.

Current result (July 18, 2026): the `timtab1` causal half of this gate is
complete, and the q-opt proof path has a second accepted improvement.
Disabling SCIP Gomory separation worsens the 30-second gap from 7.78% to
16.23% with the same incumbent; the root-only difference is small, so the
causal mechanism is sustained tree separation. q-opt now lets global root GMI
cuts drive branching instead of re-solving and branching from a clean LP at
every node. Two rounds improve the root bound from 28694 to 221552 and the
five-minute gap from 56.91% to 46.55%. Switching best-estimate search to pure
best-bound after a newly improved incumbent then reduces the same focused
five-minute gap to 34.85% (`793168 / 516737`) without losing the incumbent.
All incumbents were verified against the original JSON model. Detailed
evidence is in [`scip-timtab1-gomory-ab.md`](scip-timtab1-gomory-ab.md) and
[`qopt-timtab1-proof-switch.md`](qopt-timtab1-proof-switch.md).

The `neos17` causal mechanism is also identified, but the q-opt acceptance
half remains open. Disabling only SCIP infeasible-LP conflict analysis raises
proof time from 8.65 to 11.99 seconds and nodes from 4116 to 9248. Disabling
the conflict score in relative-pseudocost branching instead improves the run
to 7.17 seconds and 1033 nodes, so VSIDS-style branch scoring is not the
mechanism to copy. A minimal q-opt Phase-I-certificate no-good implementation
learns and propagates valid conflicts, but improves the 30-second gap only
from 16.31% to 16.17% while weakening the dual bound. Adding those conflicts
as LP rows or allowing deeper no-goods is much slower. Detailed evidence is
in [`scip-neos17-conflict-ab.md`](scip-neos17-conflict-ab.md).

Stop condition reached: retain the white-box breakdown and opt-in conflict
telemetry, but do not expand the SCIP portfolio. Move to item 2. The next
missing pieces are cheap row activation/deactivation and reason-certified
logic/set conflict propagation; both are required before revisiting tree-wide
separation or conflict rows.

### 2. Numerical cut pipeline and cut-augmented reoptimization

Goal: make useful cuts survive without collapsing node throughput.

Current evidence says cut validity and revised-basis generation are not the
main blocker. The active LP becomes too expensive after rows accumulate.

Implement the smallest measurable slices in dependency order:

1. Numerical status contract: factorization and basis failures remain
   unresolved numerical limits and are never classified as infeasibility.
2. SCIP-style row postprocessing with bound-safe coefficient cleanup and RHS
   relaxation before any cut reaches selection or validation.
3. Iterative root separation over the current active LP, so later rounds can
   consume earlier cut rows.
4. Per-cut metadata plus distinct candidate-store, pool, and active-row
   states; activate only a ranked subset and remove stale nonbinding rows.
5. Incremental batched row activation/deactivation in the revised LP body.
6. Aggregation-family parity and internal SCIP ablations only after the
   lifecycle can support the resulting row volume.

Current implementation (July 18, 2026): stale cut rows can now be replaced by
empty, position-stable slots and reactivated without forcing a cold Phase I;
same-shape matrix changes retain basis identities and refactor against the new
matrix. This removes the old stale-row coefficient work and materially raises
`timtab1` node throughput, but the tested tree-wide GMI schedules worsened the
incumbent trajectory and have not passed the acceptance gate. Exact continuous
equality aggregation reduces `timtab1` from `397/171/829` to `226/168/651`
columns/rows/nonzeros and now composes with root GMI mapping, but aggregation
plus root GMI also changes the primal trajectory adversely. Keep both
mechanisms opt-in while the separator schedule is refined.

Current implementation (July 20, 2026): LU/refactor/basis-update failures in
the revised LP are classified as `IterationLimit`, not `Infeasible`. This is
the first Stage-1 correctness guard; repeated cold-solve consistency on actual
`neos17` cut batches and explicit numerical telemetry remain open.

The root cut loop is now an explicit solve/generate/select/activate/validate
state machine. Each attempted round records its input row count, generated and
selected candidates by family, selection rejection reasons, coefficient
metrics, and validation status. Matched `neos17` one-node runs are behaviorally
unchanged: the 20-cut batch applies 20 Gomory cuts, while the 30-cut batch
selects 27 Gomory and 3 cMIR cuts and is rolled back after revised validation
returns `IterationLimit`. This establishes the control-plane baseline for
rowprep and iterative active-LP separation.

SCIP-style conservative rowprep is now active for root cut candidates. It
removes fixed-variable terms exactly, improves excessive coefficient ranges by
aggregating global variable bounds, performs bound-compensated near-integer
rounding, power-of-two scaling, and a `1e-4` root efficacy check. On `neos17`,
2560 cMIR candidates lose 7544 fixed-variable terms and 40 Gomory candidates
receive 44 compensated coefficient roundings. The previously failing 30-cut
batch now validates consistently as Optimal and applies 22 cMIR plus 8 Gomory
cuts; three independent cold runs produced identical objective, bound,
iterations, family counts, and cut counts. This is pipeline progress, not yet
an acceptance result for final search quality.

In a matched 30-second reliability/best-estimate smoke, 30 active cuts process
457 nodes versus 432 with 20 cuts and improve the dual bound from `0.02655` to
`0.02963`; the final gaps are 19.315% and 19.394%, respectively. The 30-cut
incumbent is slightly worse, so this is evidence that rowprep removed the
immediate LP-instability barrier, not that the larger cut set is ready for the
recommended profile. Active-row selection and lifecycle remain next.

The root separator now also has SCIP-style logical separation between the
per-round candidate store, a global cut pool, and active LP rows. On a
two-round `neos17` root-only run, round 1 starts from the 30 rows applied in
round 0, retests 781 retained pool rows, finds 602 still violated, and chooses
27 of 30 applied cuts from the retained pool. The generator emits 2192 cMIR
rows in round 1, but all are scalar duplicates already present in the pool;
the observed second-round gain therefore comes from iterative closure, not a
new first-layer cMIR formula. The one-node dual bound improves from
`0.00350959` to `0.00713467`, while cut time rises from about `0.94s` to
`3.98s` and active rows grow from 30 to 60.

This completes the first cut-pool control-flow slice. It does not promote the
cMIR experiment: root rows are still permanent once activated, and each
round cold-solves the rebuilt model. Next implement a total active-row budget,
binding-based active aging/removal, pool reactivation, and persistent dual
reoptimization before expanding aggregation or separator families.

Current implementation (July 20, 2026, persistent loop): the SCIP
`priceAndCutLoop` port is in. Root separation now runs on one persistent
revised LP (RowSet slots + warm dual re-solves), with SCIP's stall-round
control (`RootCutStallRounds`), row aging into retestable pool state
(`RootCutRowAgeLimit`), slot reuse, cold-retry escalation on numerical
failure, cut-row equilibration into the base Ruiz scaling, the 1e4 dynamism
acceptance gate, root nonbinding-row cleanup, and a zero-relative-gain gate
that drops inert cut sets. Bench flags: `-cut-stall-rounds`,
`-root-cut-row-age-limit` on top of the existing cut flags.

Matched five-minute suite (same machine, gated): `neos17` gap
`9.09% -> 3.37%` (bound `0.0641 -> 0.1214`, incumbent `0.15510` vs optimum
`0.15000`), `timtab1` gap `57.6% -> 38.2%` (bound `488664`),
`pk1`/`markshare_4_0` proofs retained and slightly faster (the zero-gain
gate drops their inert cut sets), 13/13 incumbents intact, `assign1-5-8`
regresses `8.99% -> 9.79%` because 67 permanent rows cost about 3x node
throughput. This meets the item-2 acceptance gate (two target gaps
improved more than 25% relative with proofs and coverage intact) with one
known regression to fix via in-tree lifecycle of inherited root rows.
Detailed stage status is in
[`scip-pipeline-parity.md`](scip-pipeline-parity.md).

Current implementation (July 21, 2026, tree lifecycle): validated root cuts
can now occupy inherited position-stable tree slots, use the complete cut
model's Ruiz scales, participate in node bound propagation, age out by path,
and reactivate from the pool. The infrastructure is retained behind
`CutRowAgeLimit`, but the policy did not promote: after scaling/propagation
parity, age 10 regresses `assign1-5-8` and `neos17` and gives only a small
`timtab1` gap improvement; hard active caps and delayed activation were worse
and were removed. Do not repeat global age/cap sweeps. A future policy needs an
importance signal stable across degenerate reliability bases, or this item
should yield to reason-certified propagation in item 3.

Primary targets: `neos17`, `timtab1`, `assign1-5-8`, and `neos5`.

Acceptance gate: at five minutes, improve at least two target gaps by 25%
relative without losing any current optimum incumbent, either proof, or more
than 20% node throughput on unaffected instances. Otherwise keep the pool
telemetry but revert the activation policy.

### 3. Structural presolve and reason-based learning

Goal: reduce the model or avoid repeated infeasible subtrees before spending
more time on generic separator families.

Priorities:

- singleton/doubleton substitution with stable postsolve maps;
- duplicate/dominated row detection;
- variable aggregation and stronger variable-bound substitution;
- clique extraction from implications and set-packing rows;
- conflict constraints learned only from reason-certified propagation
  infeasibility;
- probing schedules driven by observed deductions per unit work.

Primary targets: `timtab1`, `neos17`, `ic97_potential`, and `nidda`.

Acceptance gate: at least 20% root model reduction or a 2x node reduction on
one target, with byte-stable objective/feasibility on deterministic snapshots
and no five-minute incumbent regression.

## Later work

- Parallel B&B with worker-local LP state and atomic incumbent publication.
- Full individualization-refinement symmetry detection for block symmetries.
- More general flow-cover, zero-half, and aggregation separators after the cut
  lifecycle is cheap.
- GPU/first-order LP research for large relaxations.
- SOS, indicator, nonlinear, and broad SCIP-compatible modeling features.

## Validation checklist

For every accepted solver change:

```sh
/opt/homebrew/bin/go test ./...
/opt/homebrew/bin/go test -race ./...
/opt/homebrew/bin/go vet ./...
git diff --check
```

Then run a focused 30-second A/B, followed by the full five-minute suite with
`-parallel 3`. A result is not accepted until every candidate incumbent passes
the original-model feasibility verifier and the claimed change is visible in
machine-readable telemetry.
