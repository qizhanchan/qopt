# SCIP pipeline parity plan

This document turns the SCIP white-box work into a staged implementation
program. The key rule is to distinguish two questions:

1. Did q-opt reproduce the relevant SCIP mechanism at the same solver stage?
2. Does the completed mechanism improve matched end-to-end performance?

A negative answer to the second question is not evidence against the
algorithm while prerequisites from earlier stages are still missing. It only
shows that the current partial pipeline is not ready for promotion.

## Current pipeline gap

SCIP aggregation runs on the current presolved LP and participates in an
iterative separation loop:

```text
presolved LP rows
  -> row-side scoring from fractionality, duals, slack, and density
  -> repeated row aggregation
  -> variable-bound complementation
  -> best of cMIR / flow-cover / knapsack-cover
  -> numerical row cleanup
  -> efficacy and orthogonality selection
  -> LP or global cut pool
  -> row aging/removal
  -> dual reoptimization
  -> next separation round sees the augmented LP
```

The current experimental q-opt cMIR patch is intentionally much narrower:

```text
original model rows
  -> one positive two-row combination
  -> closest-bound complementation
  -> cMIR delta scan
  -> root cut selection
  -> permanent root LP row
```

It is useful as a formula and candidate-distribution probe, but it is not a
pipeline-parity implementation and must remain outside the recommended
profile.

## Stage 0: reproducible references

Keep one machine-readable reference for each focused target:

- q-opt recommended profile at 30 seconds and five minutes;
- SCIP JSON-input profile with the same budgets;
- root-only snapshots containing rows, columns, nonzeros, objective,
  iterations, fractional variables, and numerical counters;
- incumbent and global-bound timelines.

Every later stage compares against these references. Focused runs may use a
single instance, but promotion still requires the concurrent 13-instance
suite with `-parallel 3`.

## Stage 1: numerical status and repeatability

No separator work is trustworthy until the LP layer has a strict numerical
contract.

Required behavior:

- LU, refactor, and basis-update failures are numerical limits, never
  infeasibility proofs;
- repeated cold solves of the same cut-augmented LP return the same status
  and objective within tolerance;
- warm and cold solves agree whenever both terminate normally;
- root-cut validation and the production root solve cannot disagree without
  recording a numerical failure and retaining the unresolved bound;
- factor failures, residual retries, cold fallbacks, and rejected warm bases
  are exposed in benchmark output.

Acceptance gate:

- deterministic repeated-solve tests for 0, 20, 30, 50, and 100 added rows;
- no false `Infeasible` result from an injected singular basis update;
- stable status and objective across repeated cold solves on focused
  `neos17` cut batches.

## Stage 2: separator-independent row postprocessing

All cut families must pass through one numerical cleanup layer before
selection or LP insertion.

Implement, in this order:

1. Normalize row scale without changing the represented inequality.
2. Remove or round coefficients only by aggregating finite variable bounds
   into the row side, so the cleaned row is a valid relaxation of the cut.
3. Reject rows with non-finite data, unreliable activity, excessive
   coefficient ratio, or lost efficacy after cleanup.
4. Record pre/post nonzeros, coefficient ratio, efficacy loss, side
   relaxation, and rejection reason.

This should follow `SCIPcleanupRowprep()` and
`rowprepCleanupImproveCoefrange()` in
`3rd_repo/scip/src/scip/misc_rowprep.c`, not an absolute
`abs(coef) < epsilon` deletion rule.

Acceptance gate:

- property tests show every cleaned inequality is implied by the original
  cut plus the declared global bounds;
- selected cuts retain the configured minimum efficacy after cleanup;
- repeated-solve stability improves or remains unchanged on focused cut
  batches.

Current status (July 20, 2026): a conservative root rowprep is implemented.
It removes fixed terms exactly, eliminates range-damaging terms only with a
finite global bound and side relaxation, performs bound-compensated
near-integer rounding, uses power-of-two scaling, and enforces SCIP's `1e-4`
root efficacy threshold. Unlike SCIP, q-opt currently refuses a rounding when
the required bound is infinite instead of accepting a small validity error.
On `neos17`, this turns the previously unstable 30-cut batch into three
identical successful cold validations. Tree-level quality remains to be
measured separately. A first 30-second smoke shows no throughput collapse and
a slightly stronger final gap than the 20-cut batch, but a worse incumbent;
this keeps active-row selection and lifecycle ahead of further separator
expansion.

## Stage 3: separation rounds over the active LP

Build an explicit root separation state machine:

```text
solve -> generate -> clean -> select -> activate -> dual reoptimize -> repeat
```

Later rounds must read current active LP rows, including cuts accepted in
earlier rounds. This is the minimum change required to test the iterative
closure hypothesis. Original-row-only cMIR generation is not sufficient.

Start with root-only rounds and hard caps on generated, selected, and active
rows. Add tree separation only after root rounds are stable and measurable.

Acceptance gate:

- round telemetry matches the SCIP concepts of calls, tries, aggregations,
  generated cuts, selected cuts, and applied cuts;
- the second round demonstrably consumes at least one first-round cut row;
- each round reports objective gain, LP iterations, wall time, and numerical
  fallbacks.

Current status (July 20, 2026): the root control loop now runs the explicit
solve/generate/clean/pool/select/activate/validate sequence. Later rounds solve
the model with earlier accepted rows present, and each round records pre/post
objective, LP iterations, elapsed time, pool scans, family counts, selection
rejections, coefficient metrics, and validation status.

The two-round `neos17` root-only probe demonstrates retained closure rather
than another copy of the first-layer generator. Round 0 keeps the established
result: 2731 generated rows reduce to 811 scalar-distinct pool rows and 30
active rows. Round 1 starts with 516 rows instead of 486, retests 781
prior-round pool rows, finds 602 still violated, and selects 27 of its 30 cuts
from retained rows. All 2192 newly generated cMIR rows in round 1 are scalar
duplicates of existing pool rows; only 165 new Gomory rows enter the pool.
The 60-cut validation remains Optimal and strengthens the one-node dual bound
from `0.00350959` to `0.00713467`.

This passes the control-flow part of Stage 3, but not LP-engine parity. Each
round still rebuilds and cold-solves the augmented model instead of applying a
batch to a persistent LP and dual-reoptimizing it. The measured cut phase also
grows from about `0.94s` for one round to `3.98s` for two rounds, so the next
work is active-row lifecycle and persistent reoptimization, not another
separator family.

## Stage 4: cut store, pool, and active-row lifecycle

Separate three states that are currently conflated:

- candidate store: generated this round, not yet selected;
- cut pool: globally valid rows retained for later violation checks;
- active LP rows: the small subset currently present in the LP.

Extend the existing position-stable row slots. Add metadata for family,
rank, creation depth, efficacy, density, orthogonality, active age, pool age,
last violation, and last binding node.

Activation and deactivation should be batched so one accepted cut set causes
at most one refactor/reoptimization cycle. Pool age counts rounds where a cut
was neither active nor violated; active-row age counts consecutive nonbinding
LPs.

Acceptance gate:

- active-row count stays within a configured budget while the pool may grow;
- stale rows leave the matrix and violated pool rows can return;
- 50-100 retained pool cuts do not imply 50-100 active LP rows;
- node throughput on unaffected instances loses less than 20%.

Current status (July 20, 2026): the first lifecycle slice is implemented.
Generated candidates, globally retained pool rows, and active model rows now
have separate states. The pool removes exact positive scalar duplicates,
retests inactive rows against each new root LP point, ages rows only when they
are neither active nor violated, and can revive an expired row when a later
generator produces it again. Validation failure rolls back the model rows
without marking their pool entries active.

Second slice (July 20, 2026, later): the persistent priceAndCutLoop port
completes the remaining Stage 3/4 mechanics for the revised backend:

- one `canonCache` compiles the base model once; every round appends or
  reuses position-stable cut-row slots through the lp engine's RowSet path
  and re-solves with the previous round's warm handle (rounds 1+ typically
  take 50-400 dual iterations instead of a cold solve);
- round control follows SCIP's stall test (`objreldiff <= 1e-4` and the
  `(0.9 - 0.1*n)` fractional-count decay, `RootCutStallRounds`, SCIP default
  10) with `MaxCutRounds` as the hard cap;
- active rows age per solve and are emptied in place after
  `RootCutRowAgeLimit` consecutive nonbinding re-solves on SCIP's
  `(limit+1)/2 + 1` cadence; their pool entries become retestable, and
  emptied slots are reused by later same-sense cuts;
- a failed warm re-solve retries cold (SCIP lpSolveStable escalation)
  before the round rolls back;
- cut rows are equilibrated into the base Ruiz row-scale system (per-row
  max |coef| normalized to 1); without this, iterated cut-on-cut rows
  drifted orders of magnitude from the base scaling and broke the
  factorization around round 10;
- rowprep now enforces SCIP's `separating/maxcoefratio` (1e4) as a hard
  acceptance gate on final cut dynamism;
- after the loop, SCIP's `lp/cleanuprowsroot` is applied: rows nonbinding
  at the final root optimum are dropped before the tree inherits the model;
- a zero-gain gate drops the entire cut set when the whole loop moved the
  root objective by less than 1e-4 relative (measured requirement: without
  it, zero-gain cut rows cost `markshare_4_0` its five-minute proof and
  `glass4` its entire incumbent trajectory).

Measured root strength (SCIP-parameter run: 200-round cap, stall 10, age
limit 10, 30 cuts/round, cMIR on): `neos17` root bound `0.000681 ->
0.0375` in ten productive rounds and about 2 seconds (SCIP's root with the
aggregation family disabled reaches `0.0410`; its full root, `0.1177`,
needs the Stage 5 aggregation separator). `timtab1` root bound `28694 ->
415020` in 20 rounds (86% of SCIP's `480226`).

Matched five-minute suite effect (same machine, recommended profile plus
the cut options above, zero-gain gate active): `neos17` gap `9.09% ->
3.37%` with dual bound `0.0641 -> 0.1214` (above SCIP's final root) and
incumbent `0.15510` against the `0.15000` optimum; `timtab1` gap `57.6% ->
38.2%` with bound `352362 -> 488664`. Both proofs are retained and
slightly faster (`markshare_4_0` 137s vs 182s, `pk1` 113s vs 127s) because
the gate drops their zero-gain cut sets entirely. `glass4`, `mad`, and
`ic97_potential` also gate to zero cuts and match control; the ungated run
showed `ic97` `10.57% -> 8.08%` and `mad` `12.4% -> 9.3%` from
cut-induced incumbent-trajectory changes with no bound movement, a primal
side effect the gate deliberately gives up in exchange for protecting the
`markshare_4_0` proof and `glass4` coverage. `gen-ip002/054`, `neos5`,
`murg`, `nidda` are near parity. The one regression is `assign1-5-8`
(`8.99% -> 9.79%`; 67 permanent cut rows cost about 3x node throughput),
which is the motivation for in-tree lifecycle of the inherited root rows
rather than permanent baking.

Re-validation on the certified default (July 21, 2026 — post the Stage 4c
cadence fix and Stage 4d Farkas certificates; control =
`/tmp/qopt-miplib-5m-farkas-cert.json`): the paragraph above is STALE and
the profile no longer generalizes. Against the much faster certified
baseline, the same cut options keep only two effects: `neos17` gap
`8.37% -> 6.57%` (bound `0.0810 -> 0.1028`, q-opt's best measured
neos17 gap) and the `timtab1` BOUND `394k -> 505k` (above the July 20
figure) — but timtab1's incumbent regresses `802k -> 1069k` for a net gap
loss, `neos-3046615-murg` collapses (bound `421 -> 330`, nodes 437k ->
27k under 69 cut rows), `glass4` loses its improved incumbent
(`2.50e9 -> 3.64e9`) even though its cut set gates to zero (the root
separation phase alone perturbs the fragile incumbent trajectory), and
`assign1-5-8`, `gen-ip002/054`, `ic97_potential`, `nidda`, `mad`,
`neos5` are all flat-to-worse with node counts down 25-90%. Proofs remain
node-identical. Verdict: the root-cut options are a neos17-focused
specialist configuration, not a suite profile; the cut-augmented LP's
relative cost grew as the baseline engine got faster.

## Stage 5: aggregation parity before family expansion

Only after stages 1-4 should q-opt expand the aggregation heuristic.

Match SCIP features incrementally:

1. Score both finite sides of current LP rows using fractionality, dual,
   slack, and density.
2. Support negative start scaling where valid.
3. Aggregate repeatedly up to the configured root depth, eliminating the
   continuous variable farthest from a useful bound.
4. Include variable-bound information and implicit-integer treatment.
5. Run cMIR alone first; then compare flow-cover and knapsack-cover as
   competing methods on the same aggregated row.
6. Add objective-cutoff aggregation only after ordinary LP-row aggregation
   is understood.

SCIP A/B runs should separately control the cMIR, flow-cover, and
knapsack-cover frequencies used by `sepa_aggregation.c`, then scan
`maxaggrsroot`, `maxroundsroot`, `maxsepacutsroot`, and pool usage.

Acceptance gate:

- candidate distributions are compared at the same presolved LP stage:
  start rows, aggregation depth, density, coefficient ratio, efficacy,
  winning family, and rejection reason;
- q-opt reproduces the direction of SCIP parameter sweeps before tuning for
  final gap;
- promotion requires matched-time improvement, but a parity implementation
  may remain experimental while later lifecycle or propagation stages are
  completed.

## Stage 4b: tree separation groundwork (July 20, 2026)

The per-node cut chain now uses a global tree-slot registry: every accepted
tree cut occupies one stable slot index shared by all node LPs, a node
activates the subset on its own ancestor chain (local rows leave the LP for
foreign subtrees, as masks rather than shape changes), and warm handles
survive cross-subtree jumps through the lp engine's same-shape retarget
path. Default behavior is unchanged (the registry is empty when tree cuts
are off), and the persistent root profile is measurement-identical.

Negative result: this alone does not make tree Gomory affordable. The
30-second `timtab1` probe with tree separation (stride 5 or 10, age limit
10) still collapses node throughput (about 200-300 nodes versus 2900 with
root-only cuts) and loses the incumbent entirely. The dominant costs are
now structural: every node LP carries the registry width, reliability
probes multiply the wider-LP cost, and the cut-augmented vertex breaks the
rounding/diving incumbent path on this instance. The SCIP mechanism this
still lacks is the in-place focus-node LP: SCIP appends rows to the LP
object it just solved and dual-reoptimizes in place, paying no per-node
rebuild, re-validation, or warm-classification cost at all. That is an lp
engine change (persistent solver state across the tree walk plus
apply-rows-and-reoptimize), and it is the next port target; the slot
registry is its search-side prerequisite.

## Stage 4c: focus-node LP session (July 20, 2026)

`lp.Session` (opt-in: `RevisedFocusSession`, bench `-revised-focus-session`)
ports SCIP's focus-node LP switching onto the revised engine: one persistent,
already-assembled solver per canonical body follows the tree walk, node
solves rewrite the RHS and structural bounds in place, and the caller's
parent handle is installed only when it differs from the basis already in
the factorization (plunge children skip the install refactor entirely).
Shape changes and unsound in-place outcomes fall back to the stateless path
— cold when a dual attempt already ran — and the fallback's solver is
adopted as the new live state. `Solution.Solver` becomes a detached snapshot
so cut generation and reduced-cost fixing survive later in-place solves.

Two design lessons were measured, not assumed:

- "Continue from whatever was solved last" — SCIP's default posture — is a
  poor warm start under q-opt's traversal. Best-estimate jumps and
  reliability probes interleave relaxations that drag the live basis away
  from the next node's optimum; even tightening-only continues cost more
  dual iterations and stalls than the skipped setup saved (neos17 nodes
  −24% in the 30-second probe). The caller's parent handle is the right
  start; the session's win is skipping the rebuild/classify/refactor, not
  changing the start.
- A refactor-and-retry escalation on uncertified "no entering column" exits
  is a pure loss: the retried dual almost always repeats the exit (0 and 15
  certificates against 92 and 511 retries on neos17/timtab1), doubling the
  failed work in front of the cold Phase I reference.

The port also exposed a latent engine-wide soundness gap. The stateless
path's certified warm-reopt shortcut (dual-feasibility trusted by
construction on tightening-only handle chains) can accept a primal-feasible
but dual-infeasible vertex after long chains — compounded dual ratio-test
tolerance — and the overstated node bound then prunes optimal subtrees. On
pk1 a certified install returned −9.941 against the true LP optimum −9.929,
the stateless warm path reproduces the same wrong value from the same
handle (`QOPT_SESSION_VERIFY` harness), and the session run "proved"
objective 12 against the true 11. The session therefore verifies every
accepted optimum with an explicit dual-feasibility pass; the default
stateless path carried the latent gap until the chain-cadence fix below.

Matched five-minute suite (same machine, gated profile ± the flag): proofs
retained with correct objectives (markshare_4_0 167s, pk1 152s), 13/13
incumbents, and no new proof — the acceptance target (a neos17 or timtab1
proof) is not met. Primal quality splits: better on mad (0.124 → 0.0784),
ic97_potential (4325 → 4195), nidda (15834 → 14831), assign1-5-8
(215.9 → 212), neos17 incumbent (0.15510 → 0.15455); worse on timtab1
(0.791M → 1.177M) and glass4 (3.24e9 → 3.69e9, nodes 160k → 548). The
telemetry explains the losses: the verification pass fails and escalates to
a full primal resume almost everywhere on the ultra-degenerate instances
(glass4: 288 of 300 seconds in Phase II resumes, 5.6M iterations), i.e. the
control arm's throughput there is built on unverified certified accepts.
The flag stays off; the next lever is making verification cheap (bounded
re-verification cadence, or repairing only the violated columns) rather
than abandoning it.

## Stage 4c follow-up: certified-chain cadence (July 21, 2026)

The stateless soundness gap is closed by default. Every warm handle now
carries `certChain`, the number of consecutive certificate-trusted accepts
behind its basis; a certified Optimal is trusted only while the chain is
below `certChainLimit` (default 8, `QOPT_CERT_CHAIN` overrides; 0 = always
verify, negative = legacy unlimited). At the limit the accept runs the
explicit dual-feasibility pass (one BTRAN + pricing), and a passed check
resets the chain.

A failed check does NOT pay the primal resume by default. The same pricing
pass yields the dual-infeasibility gap Σ |dⱼ|·(uⱼ−lⱼ) over violated
columns, and vertex objective + gap is a valid dual bound
(Neumaier–Shcherbina shape), so the solve returns its primal-feasible
vertex with the objective weakened by the gap — sound bound, zero extra
work, chain kept growing so descendants keep re-verifying. Only when the
correction exceeds `certCorrectionRelTol` (1e-4, `QOPT_CERT_CORRTOL`)
relative to the vertex objective — a uselessly weak bound — or a violated
column has infinite range does the solve fall back to the primal resume,
which also repairs the basis for the subtree.

Three-arm five-minute suite (same machine, recommended profile;
control = `QOPT_CERT_CHAIN=-1`):

- Control's certified shortcut is systemically unsound, not rare-unlucky:
  chain-limit checks found dual-infeasible bases on 11 of 13 instances
  (pk1: 34.7k of 35.7k checks failed in the resume-only arm; single
  corrections reached the objective's own scale on timtab1/gen-ip002).
- Verify + resume only (no correction): proofs kept, but glass4 collapses
  to 870 nodes and loses its incumbent — its throughput was built on
  unverified accepts, and every failure paid a degenerate Phase II resume.
- Verify + uncapped correction: 13/13 incumbents, zero resumes, but
  pathological corrections wreck reported bounds (gen-ip002 gap 0.35% →
  9.2%).
- Shipped hybrid (capped correction, resume past the cap): 13/13
  incumbents, markshare_4_0 and pk1 proofs retained, dual bounds at
  effective parity (gen-ip002 0.375% vs 0.346%), timtab1 incumbent better
  (808k vs 831k). Resuming at detection also stops dual-infeasible bases
  propagating to children: gen-ip002's verify failures drop from 473k
  (correction-only) to 58. Costs: roughly 10–25% node throughput on most
  instances and worse incumbents on ic97_potential (4325 vs 4128), mad
  (0.133 vs 0.124), and glass4 (3.63e9 vs 3.24e9) — measured against a
  control arm whose bounds the telemetry shows were not trustworthy.

## Stage 4c follow-up 2: session adopts the corrected accept (July 21, 2026)

`lp.Session` now uses the same verify-and-repair shape instead of paying a
full primal resume on every failed check. Two matched five-minute A/Bs
against a fresh control baseline rebuilt on the committed cadence fix
(13/13, markshare_4_0 and pk1 proved, gen-ip002 0.366%, timtab1 808k):

- Session under the stateless K=8 trust window: pk1's FALSE PROOF RETURNS
  (claimed optimal 12, control proves 11). The K=8 window is measured-safe
  only for the stateless trajectory, whose cold rebuilds at fallback points
  act as drift firebreaks; the session walks 300k–1.8M consecutive in-place
  solves per instance (pk1: 533k reopts vs 392 fallbacks) and one
  overstated bound inside an unverified window pruned the optimal subtree.
  pk1's drift (~1.2e-2 absolute) also exceeds the 1e-4 correction cap, so
  every caught failure resumed (614) — but the window between checks is
  what let one through. This is a standing caution for the default path
  too: K=8 is empirically safe on the stateless walk, not proven safe.
- Session verifying every accept (`sessionCertChainLimit` 0, override
  `QOPT_SESSION_CERT_CHAIN`) with the capped correction as the repair:
  sound again — 13/13 incumbents, both proofs kept (pk1 at 318k nodes vs
  control's 306k, i.e. the per-accept pass is affordable). glass4's July 20
  resume collapse is gone (25 222 corrections vs 119 resumes; obj 3.64e9 vs
  control 3.58e9, ~parity). Clear wins: ic97_potential 4107 / gap 5.82% vs
  4325 / 10.57%, neos-3046615-murg gap 77.7% vs 81.1%, and small gap
  improvements on assign1-5-8, gen-ip002, gen-ip054, mad. Still worse:
  timtab1 incumbent 1141k vs 808k (nodes 4× higher — the in-place walk
  changes the heuristic trajectory) and the neos17 incumbent 0.198 vs
  0.155. No new proof.

Verdict: the flag stays off and out of the experimental profile — the
timtab1/neos17 regressions and the absence of a new proof do not pay for
the wins. The sound per-accept form is the baseline for any future session
work; an intermediate trust window (1–7) is unexplored but any window > 0
readmits the pk1 exposure class. The QOPT_SESSION_VERIFY harness now
compares against the stateless warm solve of the same request (drop-in
parity contract) and expects QOPT_CERT_CHAIN=0 so the reference verifies as
strictly as the session.

The neos17 conflict lever was also re-measured on the sound baseline
(30-second focused A/B, `-lp-conflicts`): gap 16.31% → 16.22%, dual bound
worse (0.0463 → 0.0387), fewer nodes — effectively identical to the July 18
verdict, so the certified-chain fix does not change that story. The
stop-condition from scip-neos17-conflict-ab.md stands: the missing SCIP
mechanism is reason-certified logicor/setppc propagation with
repropagation, not raw branch-path no-goods.

## Stage 4d: SCIP-shape Farkas validation of infeasible exits (July 21, 2026)

`neos17`'s node-cost profile pointed at Phase I: 114 of 300 seconds in cold
Phase I re-proofs, with 1980 dual "no entering column" exits and 1325
warm-dual iteration caps each escalating cold. SCIP never pays this — an
infeasible node LP is accepted from the dual simplex's Farkas ray, validated
by `SCIPlpGetDualfarkas` under `lp/checkfarkas = TRUE` (the default), and
`lpSolveStable`'s escalation ladder reaches "solve from scratch" only as the
last resort (`lp.c`).

q-opt's opt-in certificate (`-revised-certify-infeasible`) existed but hit
2 of 376 candidates on a 30-second `neos17` probe. Telemetry isolated the
blocker: 374 rejections on the infinite-width gate. The old certificate
validated the proof's DERIVATION — it omitted basic columns from the
tableau row, required a near-exact identity-row residual to justify the
omission, and rejected any helpful-direction column with infinite width.
But a "no entering column" exit means every helpful column carries only a
sub-candidacy-tolerance coefficient, so the gate rejected exactly the
noise-on-unbounded-column pattern SCIP treats as zero
(`SCIPsetIsDualfeasZero` skips such coefficients even on unbounded
columns).

The certificate is rewritten in SCIP's shape: validate the proof's
CONCLUSION. Aggregate the ray over every column of the equality-form model,
`coef_j = y·A_j` with `y = -sigmaR·rho`, and accept when the box-maximal
activity provably stays below `y·b`. No basis-residual precondition (the
check is self-contained regardless of how `y` was produced); coefficients
at or below the ratio test's own candidacy tolerance are treated as zero;
q-opt stays more conservative than SCIP in one place — a skipped column
with finite bounds still contributes its worst-case activity to the
acceptance margin, so only the noise-times-unbounded-column judgment is
taken on tolerance. Artificial columns are the Phase I device and
contribute exactly zero (the model is the artificial == 0 slice).

30-second `neos17` probe: certificates 634 of 634 (was 2 of 376), Phase I
time halved, nodes 588 → 1488.

Promoted to DEFAULT the same day (SCIP parity: `lp/checkfarkas` is on by
default and infeasible LPs are routinely accepted from the ray). The public
option inverts to `RevisedNoCertifyInfeasible` / bench
`-revised-no-certify-infeasible` for A/B control. Gates for the flip: the
matched suite below, the toy suite against the committed OR-Tools baseline
(11 ok / 3 warn / 0 fail — warns are the known degenerate-optimum and
GLOP-status classes, plus one degenerate-optimum warn on `mip_set_cover`
with matching objective), both toy infeasible problems still detected, and
byte-identical flag-on/flag-off toy objectives. Deterministic snapshots
change with the default, as expected for a default-behavior change; detsnap
remains two-run byte-identical on the new default.

Matched five-minute suite (gated profile ± the flag, sound f214ad9
control): no regression on any instance. `markshare_4_0` and `pk1` never
fire the certificate (all-bounded columns) and reproduce their control
runs node-for-node — both proofs retained. `neos17` gap 9.06% → 8.37%
with nodes 2486 → 10880 and bound 0.0644 → 0.0810; `timtab1` gap 58.1% →
50.9% with nodes 150k → 752k and bound 339k → 394k; `glass4` incumbent
3.58e9 → 2.50e9 (gap 68.0%); `murg` gap 81.1% → 74.9%; `ic97_potential`
4118 / 6.07% vs 4325 / 10.57%; `gen-ip002` 0.337%, `gen-ip054` 0.598%,
`assign1-5-8` 8.36% all better; `mad` and `neos5` at parity. 194k+
certificates fired across eight instances with zero rejects; all 13
incumbents pass the original-model verifier.

## Stage 4d follow-up: probe iteration-cap escalation (July 21, 2026, REJECTED)

The remaining neos17 cold-Phase-I source after the Farkas certificates is
the reliability probes: each probe runs its warm dual under a
200-iteration budget, and a capped exit falls back to a cold Phase I under
the same 200-iteration budget. SCIP's postures differ twice over:
`lp/resolveiterfac` defaults to -1 (node resolves are UNCAPPED — set.c),
and a strong-branching probe stopped at its iteration limit yields its
current dual objective as a valid child bound rather than any from-scratch
re-solve.

Both parity forms were implemented (a probe-mode lp option returning a
certified weak-duality bound at the cap — verified by the dualInfeasGap
pass with the standard capped correction — plus skipping the redundant
cold confirmation of certificate-carrying infeasible probes) and both were
REJECTED on measurement:

- Scoring branching candidates with the capped bounds: 30-second neos17
  gap 15.86% → 17.29%. The bounds are optimistic, so the degradations they
  imply are systematic underestimates that pollute selection.
- Merely skipping the probes' cold rebuilds (bounds discarded): the
  30-second neos17 signal was positive (gap 15.09%, cold starts 847 → 32),
  but the matched five-minute suite regressed nearly everywhere — glass4
  incumbent 2.50e9 → 3.58e9, ic97_potential 4118 → 4201, neos17 gap 8.37%
  → 10.14%, murg and nidda worse, node counts down 10–30%.
- Skipping only the cold confirmation of certified-infeasible probes
  reproduced the same suite regression almost value-for-value, isolating
  it as the dominant cause: the "redundant" confirmations consume probe
  budget, and without them every node runs more probes and node throughput
  drops ~15%.

Verdict: q-opt's probe economics are not SCIP's. The probes' cold rebuilds
complete often enough within their 200-iteration budget to convert into
real scores and infeasibility results that feed the reliability machinery,
and the probe budget's early exhaustion is load-bearing. All code reverted;
re-attempts must treat the probe iteration economy (budget size, confirm
costs, and per-node probe counts) as one coupled system, not per-exit
mechanics.

## Stage 4e: native strong-branch LP sessions (July 21, 2026, ACCEPTED)

The rejected Stage 4d variants had copied individual iteration-limit exits,
not SCIP's strong-branch state machine. The accepted implementation moves the
boundary into the LP engine:

- one immutable, dual-feasible parent solver state is captured per branching
  node; every side clones that exact basis/LU/eta state and is discarded after
  probing;
- iteration-limit results carry separate `estimate` and `valid bound` flags;
  the raw current objective drives candidate scoring, while child bounds are
  updated only after an explicit dual-feasibility pass;
- aggregate-Farkas infeasibility is the only native signal that deletes a
  child; uncertified no-entering exits remain unknown;
- the parent basis is restored conceptually after every side, and probe bases
  are never handed to real children;
- the per-side cap follows SCIP's auto `inititer` shape (twice average node-LP
  iterations, early-node inflation, clamped to 10..500). The total budget
  remains q-opt-calibrated at `1600 + node_iterations/2`, with at most eight
  candidates; copying SCIP's 100k/100-candidate constants over-invested in
  q-opt's slower Go LP kernel and regressed primal search.

The implementation also fixed a latent bounded-variable error: zero-width
non-basic columns were treated as pricing candidates and as dual-feasibility
violations even though they cannot move. Primal pricing, dual ratio testing,
dual-feasibility checks, and the safe-bound gap now consistently skip them.

Matched 30-second profile, 13-instance suite: 35,221 native probe sides and
1,773 compatibility fallbacks. Of 10,611 iteration-limit exits, 10,531 kept a
valid dual bound; zero uncertified infeasibility exits were used for pruning.
Fallbacks are concentrated on `glass4` and `neos-3754480-nidda`, whose node
LP path sometimes returns a sound corrected bound on a slightly
dual-infeasible basis. A one-time primal repair of those parent bases reduced
fallbacks to near zero but was REJECTED: `timtab1` and `neos17` proof bounds
regressed because the repaired vertex changed branching and child warm-start
trajectories.

Representative matched 30-second results against the pre-change 5971fe0
control: `neos17` improves from objective/bound `0.20672 / 0.03718` to
`0.15505 / 0.07822` in the focused run; `timtab1` from
`825322 / 287130` to `793311 / 288162`; `ic97_potential` from `4325` to
`4183`; `gen-ip054` from `6859.08 / 6790.99` to
`6853.08 / 6790.46`. `neos-3754480-nidda` trades a weaker incumbent
(`15890.76` vs `15517.47`) for a substantially stronger bound and gap
(`36.68%` vs `42.57%`).

The retained proof gate is stronger than before: under the five-minute
profile, `markshare_4_0` proves objective 1 in 39.6 seconds and `pk1` proves
objective 11 in 72.4 seconds. Both runs use native probes only, retain all
original-model feasibility checks, and finish with exact global bounds.

## Stage 4f: inherited root-cut lifecycle (July 21, 2026, INFRASTRUCTURE ONLY)

Validated global root cuts can now leave the fixed model body and occupy the
same position-stable tree RowSet used by local cuts. The root bundle is
inherited by every child; path-local ages empty a nonbinding inequality slot
after `CutRowAgeLimit`, and every pooled row is retested against each solved LP
point so tight or violated rows reactivate for the next generation. Native
strong branching clones the exact active parent RowSet and cannot mutate ages.

Moving rows out of the fixed body initially changed the numerical experiment:
the base cache no longer included root cuts in Ruiz column scaling or domain
propagation. The retained implementation therefore derives base-column,
base-row, and root-slot row scales from the complete validated cut model and
includes active slots in `tightenBounds`. A deterministic unit test compares
the baked and slotted root LP objectives, values, and scale vectors. Continuous
equality aggregation remains on the baked path because adding cuts can change
which variables qualify for aggregation and therefore invalidate a shared map.

The activation policy did not pass promotion. In the final matched 30-second
run after scaling and propagation parity, age 10 reduced average active rows
to `59.0/67` on `assign1-5-8`, `67.1/100` on `neos17`, and `92.3/140` on
`timtab1`. Against permanent rows, `assign1-5-8` regressed from objective/bound
`218.83 / 189.51` to `220.78 / 188.54`, and `neos17` from
`0.15675 / 0.12066` to `0.17358 / 0.10508`. `timtab1` improved only modestly
in gap (`61.14% -> 58.70%`) while processing fewer nodes. An earlier 30-row
hard cap caused severe row churn (`neos17` processed only 63 nodes), and
delaying aging by 100 or 500 nodes did not recover a cross-target win. Those
policy knobs were removed rather than retained as public options.

Verdict: retain the slot lifecycle, scaling/propagation parity, telemetry, and
existing opt-in `CutRowAgeLimit`; do not add it to the recommended profile.
The next attempt needs cut importance that is stable across degenerate
reliability bases, not another global age/cap sweep. Machine-readable evidence:
`bench/results/root-cut-tree-lifecycle-20260721.json`.

## Stage 4g: dual-proof conflict pool (July 22, 2026, OPT-IN)

SCIP's default infeasible-LP conflict path is not the FUIP graph but the dual
proof (`conflict/useinflp = 'b'`, dual-ray analysis first —
`conflict_general.c` / `conflict_dualproofanalysis.c`): the Farkas ray is
aggregated into one globally valid row, stored in a 100-entry conflictstore,
and reused as a propagate-only linear constraint. The July 18 rejection
covered branch-path no-goods, a different mechanism.

q-opt now ports the prune half. The certified aggregate Farkas exit (Stage
4d) already computes the proof row; `certifyDualLeavingRow` exports it with
identical acceptance arithmetic — structural coefficients exactly, slack
columns folded into the RHS through their global ranges, and only the
noise-on-unbounded-column judgment taken on the certificate's own candidacy
tolerance. The node cache translates it to model space (colScale division,
node-lower shift cancellation), pools up to 100 proofs with last-hit
eviction and SCIP's `maxvarsfac 0.15` density filter, and prunes any later
node or native strong-branch probe whose post-propagation box provably
violates a pooled proof — before the LP is even prepared. Proofs never
enter an LP, matching both SCIP's default and the July 18 conflict-row
verdict. Options: `UseDualProofs` / bench `-dual-proofs`;
`QOPT_DUALPROOF_VERIFY=1` shadows every prune with the real LP solve.

Soundness evidence: shadow verification contradicted 0 of 6593 timtab1 and
0 of 1434 neos17 prunes; the toy suite is unchanged (11 ok / 3 warn / 0
fail); markshare_4_0 and pk1 reproduce their control runs node-for-node in
the matched five-minute suite (the certificate never fires there).

Matched five-minute suite (26696c5 control, foreground batches): prunes
fire heavily — neos17 10.6k, neos-3046615-murg 41k, timtab1 36k — and the
target signals split. neos17 improves slightly (bound `0.0992 -> 0.1000`,
gap `5.55% -> 5.50%`, nodes +25%) and murg clearly (bound `345 -> 383`,
gap `79.8% -> 78.1%`, nodes 2.6x). timtab1 regresses consistently at both
budgets (five-minute incumbent `788306 -> 801982`, bound `375k -> 340k`;
the 30-second probe loses the incumbent entirely) — trajectory
perturbation on the known-fragile instance, not unsoundness. Remaining
instances move within run-to-run noise. The promotion gate (two target
gaps improved 25% relative) is not met, so the flag stays opt-in.

Two measured non-levers and one open lever: exact pool dedup never fires
(re-derived proofs differ bitwise through the node-lower arithmetic), the
pool churns on active instances (neos17 evicts 4.5k for 100 slots), and
the port covers only the prune outcome — SCIP's dominant cheap outcome,
the size-1 proof becoming a global bound change with repropagation
(`tightenSingleVar`), plus long-proof one-shot propagation
(`propagateLongProof`), remains unimplemented and is the next slice.
Machine-readable evidence: `bench/results/dualproof-conflicts-20260722.json`.

Measurement caution recorded the same day: harness background tasks on
macOS run QoS-throttled (~28% of one core for a `-parallel 3` suite), which
silently costs proofs and inflates apparent regressions. All promotion-grade
suites must run in the foreground; the five-minute arms above ran as five
sequential foreground batches of at most three instances with identical
batch composition in both arms.

## Stage 6: branching, propagation, and learning

Reliability branching parity is a separate workstream after the LP/cut loop
is stable. A probe should return more than objective degradation:

- propagated bound changes;
- infeasible-side reasons;
- implication and conflict information;
- reusable inference scores;
- reduced-cost deductions from the solved child LP.

Primary target is `timtab1`, where presolve and inference are more likely to
matter than additional generic root cuts. `neos17` remains the primary cut
closure and LP-stability target.

## Promotion criteria

The long-term target is q-opt within 1.5x SCIP wall time on proofs where SCIP
finishes inside the budget, or a comparable incumbent and gap at the same
five-minute limit. Intermediate changes are promoted only when they preserve
correctness and current 13-instance coverage.

Retain this vector for every experiment rather than only root objective:

- root gain and root LP iterations;
- active and pooled cut counts by family;
- first 100 and 1000 node LP iterations;
- nodes per second and LP iterations per second;
- incumbent timeline and source;
- global-bound slope;
- refactors, factor failures, cold fallbacks, and unresolved nodes.
