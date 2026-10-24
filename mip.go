package qopt

import (
	"container/heap"
	"context"
	"math"
	"strings"
	"time"
)

type mipNode struct {
	bounds []boundConstraint
	depth  int
	// branchLiterals contains only integral decisions made by the B&B tree.
	// It deliberately excludes reduced-cost and conflict-propagated bounds so
	// LP conflict learning can reconstruct a globally valid assumption set.
	branchLiterals []boundLiteral
	// subtreeBound is a valid inherited LP bound for this whole subtree.
	// Normally it is the parent's LP objective; reliability branching can
	// replace it with the stronger child LP bound already proved by a probe.
	// It is used for pre-LP pruning and DFS/global-bound accounting only.
	subtreeBound      float64
	subtreeBoundKnown bool
	// warm is the LP basis handed down from this node's parent. The root
	// node has warm == nil and triggers a cold Phase I / II solve. Once
	// the LP backend supports warm starts (revised simplex), the child
	// nodes carry their parent's optimal basis here and the LP is re-
	// optimized via dual simplex after the new bound is applied.
	warm *lpWarmStart
	// cuts is the chain of Gomory cuts that apply to this subtree. Each
	// link holds the cuts emitted at one ancestor (or at this node); the
	// LP at this node sees the flattened set as extra constraints on top
	// of work.constraints. nil at the root and whenever per-node cuts are
	// disabled.
	cuts *cutBundle
	// Pseudocost provenance: which variable was branched to create this
	// node, in which direction, and the LP-objective state right before
	// the branch. Used to update the pseudocost table after this node's
	// LP is solved. Root nodes have parentVar = −1.
	parentVar   int
	parentDown  bool
	parentStep  float64
	parentLPObj float64
	// extraBounds carries reduced-cost-derived bound tightenings computed at
	// this node's own LP solve, forwarded to both children alongside their
	// branching bound once this node is branched. Populated only by
	// best-bound search — DFS builds children in the same loop iteration it
	// solves the node in, so it folds these in locally without needing a
	// field (see solveMIP).
	extraBounds []boundConstraint
	// cutAges is index-aligned with cuts.flatten()'s ancestor-first order:
	// cutAges[i] is how many consecutive checks along this path cut i has
	// sat non-binding. Only populated/consulted when Options.CutRowAgeLimit
	// > 0; see cutaging.go.
	cutAges []int
}

// cutBundle is a copy-on-extend linked chain of node-local Gomory cut sets.
// Each node carries a pointer to the chain inherited from its parent;
// emitting new cuts wraps the parent in a fresh link rather than copying
// the underlying slice, so siblings share storage.
//
// Every link's cuts also occupy a contiguous range [base, base+len(cuts)) in
// the searchContext's global tree-slot registry. All node LPs are built at
// the registry's width with non-path slots emptied (SCIP's local rows leave
// the LP when the focus node exits their subtree), so every node in the tree
// shares one stable ext-row layout: warm starts survive cross-subtree jumps
// through the lp engine's same-shape retarget path instead of falling to a
// cold Phase I whenever two paths' cut chains have different lengths.
type cutBundle struct {
	parent *cutBundle
	cuts   []Constraint
	// base is this link's slot offset in searchContext.treeSlots.
	base int
	// flat memoizes flatten(): the chain is immutable after creation, and a
	// stable flattened slice is what lets canonCache key its compiled cut
	// RowSets on slice identity instead of recompiling per node.
	flat []Constraint
}

func (b *cutBundle) size() int {
	n := 0
	for x := b; x != nil; x = x.parent {
		n += len(x.cuts)
	}
	return n
}

func (b *cutBundle) flatten() []Constraint {
	if b == nil {
		return nil
	}
	if b.flat != nil {
		return b.flat
	}
	out := make([]Constraint, 0, b.size())
	var walk func(x *cutBundle)
	walk = func(x *cutBundle) {
		if x == nil {
			return
		}
		walk(x.parent)
		out = append(out, x.cuts...)
	}
	walk(b)
	b.flat = out
	return out
}

// modelWithExtraConstraints returns a shallow-copy Model whose constraints
// are work.constraints followed by extras. vars/objective/direction are
// shared with the input. The returned model is read-only by convention —
// callers must not AddConstraint into it.
func modelWithExtraConstraints(m *Model, extras []Constraint) *Model {
	if len(extras) == 0 {
		return m
	}
	aug := *m
	aug.constraints = make([]Constraint, 0, len(m.constraints)+len(extras))
	aug.constraints = append(aug.constraints, m.constraints...)
	aug.constraints = append(aug.constraints, extras...)
	return &aug
}

// perNodeCutCap bounds the number of cuts emitted from a single B&B node,
// and cutPathCap bounds the cumulative inherited+local cut count along any
// root-to-node path. These keep per-node cut generation from running away
// on instances where every fractional row yields a valid GMI.
const (
	perNodeCutCap = 20
	cutPathCap    = 200
	// treeSlotCap bounds the global tree-cut slot registry. Every node LP
	// carries one (mostly empty) ext row per registered slot, so unbounded
	// registry growth would widen every LP in the tree.
	treeSlotCap = 200
)

// Adaptive cut cadence — CBC's `maximumCutPasses_` halving idea ported to the
// depth-stride model. When a per-node cut batch leaves the LP objective
// essentially unchanged (relative change < cutImprovementTol), it counts as
// a "miss"; after cutMissesToDecay consecutive misses we double the effective
// stride (capped at 1<<maxExtraStrideShift) so further nodes fire cuts less
// often, autotuning toward sparseness on instances where extra cuts buy
// nothing. We never lower extraStride once it has been raised — recent dud
// history is the lesson.
const (
	cutImprovementTol   = 1e-4
	cutMissesToDecay    = 2
	maxExtraStrideShift = 4
	// The budget shape matches SCIP (fixed offset plus half of ordinary node-LP
	// work), but the offset is q-opt-calibrated. SoPlex's 100k default spends
	// too much wall time in q-opt's Go revised simplex; matched-suite A/B keeps
	// the existing 1600 offset with the native no-cold probe path.
	reliabilityInitialIterationBudget = 1600
	reliabilityPrimalIterationBudget  = 1600
)

func reliabilityProbeIterationBudget(nodeLPIters, strongBranchIters int, primalUrgency bool, primalOffset int) int {
	offset := reliabilityInitialIterationBudget
	if primalOffset > 0 {
		offset = primalOffset
	} else if primalUrgency {
		offset = reliabilityPrimalIterationBudget
	}
	allowed := offset + nodeLPIters/2
	if remaining := allowed - strongBranchIters; remaining > 0 {
		return remaining
	}
	return 0
}

func reliabilityThresholdForNode(opts Options, nodeLPIters, strongBranchIters, nodes int) int {
	if opts.ReliabilityThreshold > 0 {
		return opts.ReliabilityThreshold
	}
	// SCIP interpolates between minreliable=1 and maxreliable=5 according
	// to how much strong-branch budget remains. Early search therefore asks
	// for five observations; near exhaustion, one observation is enough.
	nodeQuota := nodeLPIters / 2
	offset := reliabilityInitialIterationBudget
	if opts.ReliabilityPrimalIterationBudget > 0 {
		offset = opts.ReliabilityPrimalIterationBudget
	}
	maxStrong := offset + nodeQuota
	denom := float64(strongBranchIters + 1)
	prio := float64(maxStrong-strongBranchIters) / denom
	if alt := float64(nodeQuota-strongBranchIters) / denom; alt > prio {
		prio = alt
	}
	if prio < 0 {
		prio = 0
	} else if prio > 1 {
		prio = 1
	}
	reliable := int(math.Ceil(1 + 4*prio))
	if opts.ReliabilityAdaptiveThreshold && nodes >= 64 && float64(nodeLPIters)/float64(nodes) <= 20 && reliable > 2 {
		reliable = 2
	}
	return reliable
}

// reliabilityProbeIterationLimit mirrors relpscost/inititer=0: twice the
// observed average node resolve work, inflated at the first few important
// nodes, then clamped to [10, 500].
func reliabilityProbeIterationLimit(nodeLPIters, nodes int) int {
	if nodes <= 0 || nodeLPIters <= 0 {
		return 500
	}
	limit := 2 * nodeLPIters / nodes
	limit = int(float64(limit) * (1 + 20/float64(nodes)))
	if limit < 10 {
		return 10
	}
	if limit > 500 {
		return 500
	}
	return limit
}

func hasGeneralIntegerVars(vars []VarSpec) bool {
	for _, spec := range vars {
		if spec.Type == Integer {
			return true
		}
	}
	return false
}

func (sc *searchContext) shouldRunIncumbentRepair() bool {
	if !sc.haveIncumbent || sc.nodes-lastIncumbentNode(sc.stats) < 200 {
		return false
	}
	if !sc.opts.RepairAdaptiveCadence {
		return sc.nodes%200 == 0
	}
	interval := sc.repairInterval
	if interval <= 0 {
		interval = 200
	}
	return sc.nodes-sc.repairLastNode >= interval
}

func (sc *searchContext) recordRepairResult(improved bool) {
	if !sc.opts.RepairAdaptiveCadence || !sc.haveIncumbent {
		return
	}
	sc.repairLastNode = sc.nodes
	if improved {
		sc.repairInterval = 200
		return
	}
	if sc.repairInterval < 200 {
		sc.repairInterval = 200
	}
	sc.repairInterval *= 2
	if sc.repairInterval > 3200 {
		sc.repairInterval = 3200
	}
}

type cutCadenceState struct {
	misses      int
	extraStride int // 0 ≡ 1 (no multiplier)
}

func (s *cutCadenceState) effectiveStrideMultiplier() int {
	if s == nil || s.extraStride < 1 {
		return 1
	}
	return s.extraStride
}

// observe records whether a cut batch moved the LP objective. preObj is the
// LP relaxation value at this node before the cuts; postObj is the validation
// LP objective after the cuts. Cuts are valid by construction so postObj can
// only equal or tighten preObj; the magnitude of the change is the signal.
func (s *cutCadenceState) observe(preObj, postObj float64) {
	if s == nil {
		return
	}
	rel := math.Abs(postObj-preObj) / math.Max(1, math.Abs(preObj))
	if rel < cutImprovementTol {
		s.misses++
		if s.misses >= cutMissesToDecay {
			cur := s.extraStride
			if cur < 1 {
				cur = 1
			}
			if cur < (1 << maxExtraStrideShift) {
				s.extraStride = cur * 2
				debugf("gomory cadence: misses=%d → extra stride %d", s.misses, s.extraStride)
			}
			s.misses = 0
		}
		return
	}
	s.misses = 0
}

// shouldCutAtDepth is the per-node cut firing predicate. Cuts fire when
// depth ≤ MaxCutDepth and depth is a multiple of the effective stride
// (user-set CutDepthStride scaled by any adaptive multiplier the cadence
// state accumulated). Used by both DFS and best-bound solvers so the gating
// policy stays in one place.
func shouldCutAtDepth(depth int, opts Options, cadence *cutCadenceState) bool {
	if opts.MaxCutDepth <= 0 || depth > opts.MaxCutDepth {
		return false
	}
	stride := opts.CutDepthStride
	if stride < 1 {
		stride = 1
	}
	stride *= cadence.effectiveStrideMultiplier()
	return depth%stride == 0
}

// solveNodeOrFull solves a node LP via the canonCache fast path when the
// cache is present and matches the model being solved. extras are the
// per-node cut rows appended to the cache's base model — callers pass the
// same slice they built model from (model == modelWithExtraConstraints(
// cache.model, extras)), so a cut-augmented node LP also rides the cache via
// the lp engine's RowSet path instead of the generic canonicalize+scale
// builder. Falls back to the generic builder for foreign models (tableau
// backend, reduced models).
func solveNodeOrFull(ctx context.Context, cache *canonCache, model *Model, extras []Constraint, bounds []boundConstraint, opts Options, warm *lpWarmStart) (Result, lpResponse, error) {
	return solveNodeOrFullWithStructure(ctx, cache, model, extras, extras, bounds, opts, warm)
}

// solveNodeOrFullWithStructure is solveNodeOrFull with stable identity and
// senses for extra-row slots. extras supplies the model-visible active rows;
// structure supplies the immutable cut-pool ordering used by the revised
// cache across aging/reactivation changes.
func solveNodeOrFullWithStructure(ctx context.Context, cache *canonCache, model *Model, extras, structure []Constraint, bounds []boundConstraint, opts Options, warm *lpWarmStart) (Result, lpResponse, error) {
	if cache != nil {
		if cache.model == model {
			res, resp := cache.solveNode(ctx, bounds, opts, warm)
			return res, resp, nil
		}
		if len(extras) > 0 && len(model.constraints) == len(cache.model.constraints)+len(extras) {
			res, resp := cache.solveNodeRowsWithStructure(ctx, bounds, structure, extras, opts, warm)
			return res, resp, nil
		}
	}
	res, resp, _, err := solveLPModelWithBackend(ctx, model, bounds, opts, warm)
	return res, resp, err
}

func solveMIP(ctx context.Context, m *Model, opts Options) (Result, error) {
	start := time.Now()
	if opts.TimeLimit > 0 {
		var cancel context.CancelFunc
		ctx, cancel = context.WithTimeout(ctx, opts.TimeLimit)
		defer cancel()
	}
	work := m
	symmetryChanged := false
	structureChanged := false
	var rootImplications []boundImplication
	var rootRelations []variableBoundRelation
	var inferenceDown, inferenceUp []int
	cutRounds := 0
	cutsAdded := 0
	cutElapsed := time.Duration(0)
	var rootCutRows []Constraint
	var rootCutScaleModel *Model
	stats := &SolverStats{}

	// Symmetry-breaking: detect verified variable-permutation orbits and add
	// a lex-ordering constraint chain per orbit before anything else touches
	// the model, so root cuts and every B&B node LP see the tightened
	// formulation. See symmetry.go for the verification argument.
	if opts.DetectSymmetry {
		if orbits := detectSymmetryOrbits(m); len(orbits) > 0 {
			work = cloneModel(m)
			addSymmetryBreakingConstraints(work, orbits)
			symmetryChanged = true
		}
	}

	// Structural root cuts are derived directly from the original rows and
	// therefore remain valid for every B&B node. Keep them separate from
	// tableau cuts so they do not depend on a particular LP backend.
	if opts.DetectStructure {
		conflicts := detectBinaryRowConflicts(work)
		bigms := detectSingleBinaryBigMTightenings(work)
		fixings := detectPartitioningOrbitopes(m)
		probes := probeBinaryImplications(work, opts.FeasibilityTol)
		rootImplications = probes.Implications
		rootRelations = probes.Relations
		inferenceDown, inferenceUp = implicationInferenceCounts(len(work.vars), probes.Implications)
		stats.Structural.ConflictCuts = len(conflicts)
		stats.Structural.BigMTightenings = len(bigms)
		stats.Structural.OrbitopeFixings = len(fixings)
		seenVBRows := make(map[int]struct{}, len(probes.Relations))
		for _, rel := range probes.Relations {
			seenVBRows[rel.SourceRow] = struct{}{}
		}
		stats.Structural.VariableBoundRows = len(seenVBRows)
		stats.Structural.Implications = len(probes.Implications)
		stats.Structural.ProbeSides = probes.ProbeSides
		stats.Structural.InfeasibleSides = probes.InfeasibleSides
		stats.Structural.GlobalTightenings = probes.GlobalTightened
		if len(conflicts)+len(bigms)+len(fixings)+probes.GlobalTightened > 0 {
			structureChanged = true
			work = cloneModel(work)
			for _, tightening := range bigms {
				work.constraints[tightening.row] = tightening.cut
			}
			if len(bigms) > 0 {
				work.rowsSorted = nil
			}
			for _, c := range conflicts {
				_ = work.AddNamedConstraint(c.Name, c.Coeffs, c.Sense, c.RHS)
			}
			for _, fixing := range fixings {
				for v := range fixing.Coeffs {
					zero := 0.0
					work.vars[v].Upper = &zero
				}
			}
			for v := range work.vars {
				if probes.GlobalLower != nil && probes.GlobalLower[v] > *work.vars[v].Lower+opts.FeasibilityTol {
					b := probes.GlobalLower[v]
					work.vars[v].Lower = &b
				}
				if probes.GlobalUpper != nil && probes.GlobalUpper[v] < *work.vars[v].Upper-opts.FeasibilityTol {
					b := probes.GlobalUpper[v]
					work.vars[v].Upper = &b
				}
			}
			debugf("structure: conflicts=%d big-M=%d orbitopes=%d vb-rows=%d implications=%d global-bounds=%d", len(conflicts), len(bigms), len(fixings), len(seenVBRows), len(probes.Implications), probes.GlobalTightened)
		}
	}

	// Root cuts run through an explicit solve/generate/select/activate/
	// validate loop. The state machine records each candidate's family and
	// disposition while preserving the existing cut formulas and selection
	// behavior. Later rounds solve the already-augmented work model, so their
	// input row count exposes whether iterative closure is actually occurring.
	if opts.MaxCutRounds > 0 {
		rootCutBase := work
		separated, err := runRootSeparation(ctx, work, opts, stats)
		if err != nil {
			return Result{}, err
		}
		manageRootCutsInTree := opts.LPBackend == LPBackendRevised &&
			!equalityAggregationEnabled(opts) && opts.CutRowAgeLimit > 0 &&
			len(separated.rows) > 0
		if manageRootCutsInTree {
			work = rootCutBase
			rootCutRows = append([]Constraint(nil), separated.rows...)
			rootCutScaleModel = separated.model
		} else {
			work = separated.model
		}
		cutRounds = separated.rounds
		cutsAdded = separated.cuts
		cutElapsed = separated.elapsed
	}

	cadence := &cutCadenceState{}

	// Cross-node canonicalize cache (revised backend only). Built once against
	// the post-root-cut `work` model; reused for every node LP whose model is
	// `work` (i.e. whenever per-node cuts are inactive). cleanCache serves the
	// no-cuts "clean" LP solve that runs against the original `m`.
	var workCache, cleanCache *canonCache
	if opts.LPBackend == LPBackendRevised {
		buildCache := buildCanonCache
		if equalityAggregationEnabled(opts) {
			buildCache = buildAggregatedCanonCache
		}
		if rootCutScaleModel != nil {
			workCache = buildCanonCacheScaledLike(work, rootCutScaleModel)
		} else {
			workCache = buildCache(work)
		}
		if m == work {
			cleanCache = workCache
		} else {
			cleanCache = buildCache(m)
		}
		if opts.UseDualProofs && workCache != nil && !opts.RevisedNoCertifyInfeasible {
			workCache.dualProofs = newDualProofPool(work, opts.FeasibilityTol)
		}
	}

	sc := &searchContext{
		ctx:              ctx,
		original:         m,
		work:             work,
		opts:             opts,
		start:            start,
		stats:            stats,
		cadence:          cadence,
		workCache:        workCache,
		cleanCache:       cleanCache,
		structureChanged: structureChanged,
		// Global root cuts are part of work's fixed relaxation, so branch and
		// primal search directly from that LP. Re-solving the original model at
		// every node doubled the LP path and discarded the cut-induced vertex;
		// SCIP's measured timtab1 gain instead comes from letting Gomory cuts
		// steer the tree. Symmetry rows retain the historical clean solve because
		// they encode an arbitrary representative rather than a relaxation cut.
		cleanEveryNode:  symmetryChanged,
		pcosts:          newPseudocostState(len(work.vars)),
		hasGeneralInts:  hasGeneralIntegerVars(work.vars),
		implications:    rootImplications,
		relations:       rootRelations,
		inferenceDown:   inferenceDown,
		inferenceUp:     inferenceUp,
		cutRounds:       cutRounds,
		cutsAdded:       cutsAdded,
		cutElapsed:      cutElapsed,
		limitStatus:     Optimal,
		unresolvedBound: math.NaN(),
		tag:             "mip",
		cutLabel:        "node",
	}
	if len(rootCutRows) > 0 {
		sc.treeSlots = append(sc.treeSlots, rootCutRows...)
		sc.rootSlotCount = len(rootCutRows)
		sc.rootCuts = &cutBundle{cuts: rootCutRows, base: 0}
		sc.stats.CutLifecycle.RootRows = len(rootCutRows)
	}
	sc.heuristics = newHeuristicScheduler(ctx, opts, stats)
	var q nodeQueue
	if opts.MIPNodeSelection != MIPDepthFirst {
		if opts.MIPNodeSelection == MIPBestEstimate {
			sc.tag = "mip-est"
			sc.cutLabel = "estimate node"
		} else {
			sc.tag = "mip-bb"
			sc.cutLabel = "bb node"
		}
		open := &mipBestBoundHeap{direction: work.direction, selection: opts.MIPNodeSelection}
		heap.Init(open)
		q = &bestBoundQueue{open: open}
	} else {
		q = &dfsQueue{}
	}
	res, err := runMIPSearch(sc, q)
	if workCache != nil && workCache.dualProofs != nil {
		stats.DualProofs = workCache.dualProofs.stats
	}
	return res, err
}

// searchContext carries all state shared by every B&B node evaluation — the
// models, options, caches, incumbent, counters, and limit status. It is what
// the old DFS loop body and solveMIPBestBound's evaluate closure each held as
// a private copy of the same ~20 locals.
type searchContext struct {
	ctx        context.Context
	original   *Model // the user's model (pre symmetry / root-cut augmentation)
	work       *Model // original + symmetry-breaking rows + root cuts
	opts       Options
	start      time.Time
	stats      *SolverStats
	cadence    *cutCadenceState
	workCache  *canonCache
	cleanCache *canonCache
	// Structural reformulations get one clean original-model LP at the root
	// for primal heuristics. Generic cuts and symmetry retain the historical
	// per-node clean solve because they also steer branching at every depth.
	structureChanged bool
	cleanEveryNode   bool
	pcosts           *pseudocostState
	hasGeneralInts   bool
	implications     []boundImplication
	relations        []variableBoundRelation
	inferenceDown    []int
	inferenceUp      []int
	heuristics       *heuristicScheduler
	neighborhoods    neighborhoodScheduler
	lpConflicts      lpConflictPool
	rins             rinsState
	localBranching   localBranchingState
	proximity        proximityState
	repairLastNode   int
	repairInterval   int
	nextBoundEvent   time.Duration

	cutRounds  int
	cutsAdded  int
	cutElapsed time.Duration

	// treeSlots is the global tree-cut slot registry (see cutBundle). Slots
	// are append-only and capped at treeSlotCap; a node's LP activates the
	// subset on its own chain and sees every other slot as an empty row of
	// the same sense, keeping the ext-row layout identical tree-wide.
	treeSlots []Constraint
	// rootCuts is the globally valid prefix installed as inherited slots when
	// tree lifecycle management is enabled. rootSlotCount separates that
	// prefix from subtree-local cuts for active caps and clean-LP decoupling.
	rootCuts      *cutBundle
	rootSlotCount int

	bestObj       float64
	bestValues    []float64
	haveIncumbent bool
	nodes         int
	iterations    int
	// nodeLPIterations excludes strong-branch probes. Reliability probing gets
	// a SCIP-style global budget that grows as a fraction of this real search
	// work, preventing probes from dominating shallow/large instances.
	nodeLPIterations  int
	strongBranchIters int
	limitStatus       Status
	// unresolvedBound retains a valid inherited/LP bound for subtrees that
	// were deliberately not completed (gap pruning or a numerical LP failure).
	// Without it, removing such a node from the queue can falsely collapse the
	// reported global bound to the incumbent. unresolvedUnknown marks the root
	// case where no LP bound was obtained at all.
	unresolvedBound   float64
	unresolvedUnknown bool
	gapPruned         bool
	// pumpedRoot tracks whether we've already run the feasibility pump on
	// the root LP solution. Pump is expensive (≤fpMaxIters LP solves) so
	// once per solve is the right cadence.
	pumpedRoot             bool
	divedRoot              bool
	separatedStructureRoot bool

	// Debug-log identity: "mip"/"node" under DFS, "mip-bb"/"bb node" under
	// best-bound, preserving the historical per-mode message prefixes.
	tag      string
	cutLabel string
}

// treeSlotViews materializes a node's cut rows at the global registry width.
// structure aliases the registry (stable slot identity and senses for the
// revised cache and warm-start classification); extras holds the same rows
// with every slot outside the node's own chain emptied — those cuts were
// derived under other subtrees' local bounds and are not valid here. With an
// empty registry both views are nil and the node LP is the plain base model.
func (sc *searchContext) treeSlotViews(chain *cutBundle) (structure, extras []Constraint) {
	width := len(sc.treeSlots)
	if width == 0 {
		return nil, nil
	}
	structure = sc.treeSlots[:width:width]
	extras = make([]Constraint, width)
	for i := range extras {
		extras[i] = Constraint{Name: structure[i].Name, Sense: structure[i].Sense}
	}
	for link := chain; link != nil; link = link.parent {
		for j := range link.cuts {
			if idx := link.base + j; idx < width {
				extras[idx] = link.cuts[j]
			}
		}
	}
	return structure, extras
}

// registerTreeSlots appends newly accepted cuts to the global registry and
// returns their base offset, or -1 when the registry cap has no room.
func (sc *searchContext) registerTreeSlots(cuts []Constraint) int {
	if len(sc.treeSlots)+len(cuts) > sc.rootSlotCount+treeSlotCap {
		return -1
	}
	base := len(sc.treeSlots)
	sc.treeSlots = append(sc.treeSlots, cuts...)
	return base
}

func (sc *searchContext) withStats(res Result) Result {
	res.CutRounds = sc.cutRounds
	res.Cuts = sc.cutsAdded
	res.CutElapsed = sc.cutElapsed
	res.NodeSelection = sc.opts.MIPNodeSelection
	res.Stats = sc.stats
	// Derive the optimality gap from the dual bound stamped on res.DualBound
	// and the incumbent (Objective; Values != nil signals an incumbent exists).
	res.RelativeGap = mipRelativeGap(res.DualBound, res.Objective, res.Values != nil)
	return res
}

func (sc *searchContext) recordUnresolvedBound(bound float64) {
	if math.IsNaN(bound) {
		sc.unresolvedUnknown = true
		return
	}
	if math.IsNaN(sc.unresolvedBound) ||
		(sc.work.direction == Minimize && bound < sc.unresolvedBound) ||
		(sc.work.direction == Maximize && bound > sc.unresolvedBound) {
		sc.unresolvedBound = bound
	}
}

func (sc *searchContext) recordUnresolvedNode(node mipNode) {
	if node.parentVar < 0 {
		sc.unresolvedUnknown = true
		return
	}
	if node.subtreeBoundKnown {
		sc.recordUnresolvedBound(node.subtreeBound)
		return
	}
	sc.recordUnresolvedBound(node.parentLPObj)
}

// pruneByBound centralizes incumbent pruning so a node closed only because it
// meets MIPGap remains represented in the final dual bound and status.
func (sc *searchContext) pruneByBound(bound float64) bool {
	switch boundPruneReason(sc.work.direction, bound, sc.bestObj, sc.opts) {
	case boundPruneExact:
		return true
	case boundPruneGap:
		sc.gapPruned = true
		sc.recordUnresolvedBound(bound)
		return true
	default:
		return false
	}
}

func (sc *searchContext) globalDualBound(frontier float64) float64 {
	if sc.unresolvedUnknown {
		return math.NaN()
	}
	best := frontier
	if !math.IsNaN(sc.unresolvedBound) {
		if math.IsNaN(best) ||
			(sc.work.direction == Minimize && sc.unresolvedBound < best) ||
			(sc.work.direction == Maximize && sc.unresolvedBound > best) {
			best = sc.unresolvedBound
		}
	}
	if math.IsNaN(best) {
		if sc.haveIncumbent {
			return sc.bestObj // no open or unresolved subtree: exact proof
		}
		return math.NaN()
	}
	if sc.haveIncumbent {
		if sc.work.direction == Minimize && best > sc.bestObj {
			best = sc.bestObj
		}
		if sc.work.direction == Maximize && best < sc.bestObj {
			best = sc.bestObj
		}
	}
	return best
}

type evalOutcome int

const (
	evalPruned evalOutcome = iota // node is dead (infeasible / pruned / integer leaf)
	evalBranch                    // candidate ready to branch on
	evalStop                      // limit or fatal LP status; stop the search
)

type nextStatus int

const (
	nextBranch nextStatus = iota // candidate returned
	nextEmpty                    // queue exhausted — the tree is proven
	nextStop                     // a limit tripped while waiting for the next node
)

// nodeQueue abstracts the open-node container and, with it, the one real
// difference between depth-first and best-bound search: WHEN a node is
// evaluated. DFS stores raw nodes and evaluates at pop time (next);
// best-bound must evaluate at push time (add) because its heap is keyed on
// the node's own LP objective.
type nodeQueue interface {
	// add enqueues the root or a branched pair of children. Returns false
	// when the search must stop (limit tripped or fatal LP status during a
	// push-time evaluation).
	add(sc *searchContext, nodes ...mipNode) (bool, error)
	// next returns the next evaluated candidate to branch on.
	next(sc *searchContext) (mipCandidate, nextStatus, error)
	// relaxationModel returns the (model, extras) reliability probes run
	// against for cand. Historical per-mode difference preserved: DFS probes
	// the node's own age-filtered, pre-new-cuts LP; best-bound probes the
	// full post-cut set the children will inherit.
	relaxationModel(sc *searchContext, cand mipCandidate) (*Model, []Constraint, []Constraint)
	// dualBound computes the global dual bound over the remaining open nodes.
	dualBound(sc *searchContext) float64
}

// runMIPSearch is the single B&B driver shared by both node-selection modes:
// seed the root, then repeatedly take the next evaluated candidate from the
// queue, pick a branching variable, and enqueue the two children. All
// node-level work (LP + safeguards + heuristics + cuts) lives in
// searchContext.evaluate; all ordering policy lives in the queue.
func runMIPSearch(sc *searchContext, q nodeQueue) (Result, error) {
	root := mipNode{parentVar: -1, cuts: sc.rootCuts}
	if sc.rootSlotCount > 0 {
		root.cutAges = make([]int, sc.rootSlotCount)
	}
	cont, err := q.add(sc, root)
	if err != nil {
		return Result{}, err
	}
	sc.recordBoundEvent(q, true)
	for cont {
		cand, st, nerr := q.next(sc)
		if nerr != nil {
			return Result{}, nerr
		}
		if st != nextBranch {
			break
		}
		branchVar, ok := pickBranchVar(sc.work.vars, cand.values, sc.opts, sc.pcosts)
		if !ok {
			continue
		}
		// Reliability branching: refine the variable choice with on-the-fly
		// strong-branching probes for not-yet-reliable candidates. The probes
		// warm-start from this candidate's own basis and update the
		// pseudocost table.
		preferDown := true
		choice := reliabilityChoice{varIndex: branchVar, ok: true, preferDown: true}
		if sc.opts.MIPBranching == MIPBranchReliability {
			relModel, relStructure, relExtras := q.relaxationModel(sc, cand)
			budget := reliabilityProbeIterationBudget(sc.nodeLPIterations, sc.strongBranchIters, !sc.haveIncumbent && sc.hasGeneralInts, sc.opts.ReliabilityPrimalIterationBudget)
			relOpts := sc.opts
			relOpts.ReliabilityThreshold = reliabilityThresholdForNode(sc.opts, sc.nodeLPIterations, sc.strongBranchIters, sc.nodes)
			probeIterLimit := reliabilityProbeIterationLimit(sc.nodeLPIterations, sc.nodes)
			if sc.opts.ReliabilityAdaptiveThreshold && sc.opts.ReliabilityThreshold <= 0 {
				if relOpts.ReliabilityThreshold < defaultReliabilityThreshold {
					sc.stats.Branching.ReliabilityAdaptiveLowNodes++
				} else {
					sc.stats.Branching.ReliabilityAdaptiveHighNodes++
				}
			}
			choice = chooseBranchVarReliability(sc.ctx, sc.work, relModel, relExtras, cand.bounds, cand.values, cand.objective, cand.childWarm, cand.probeState, relOpts, sc.pcosts, sc.workCache, relStructure, sc.inferenceDown, sc.inferenceUp, budget, probeIterLimit)
			sc.strongBranchIters += choice.iterations
			sc.iterations += choice.iterations
			sc.stats.addRevised(choice.revised)
			sc.stats.Branching.ReliabilityCandidates += choice.candidates
			sc.stats.Branching.ReliabilityLPSolves += choice.lpSolves
			sc.stats.Branching.ReliabilityIterations += choice.iterations
			sc.stats.Branching.ReliabilityInfeasibleSides += choice.infeasibleSides
			sc.stats.Branching.ReliabilityDomainReductions += choice.domainReductions
			sc.stats.Branching.ReliabilitySelectedProbed += choice.selectedProbed
			sc.stats.Branching.ReliabilityDiscardedProbes += choice.discardedProbes
			sc.stats.Branching.ReliabilityNativeProbes += choice.nativeProbes
			sc.stats.Branching.ReliabilityFallbackProbes += choice.fallbackProbes
			sc.stats.Branching.ReliabilityIterationLimits += choice.iterationLimits
			sc.stats.Branching.ReliabilityValidLimitBounds += choice.validLimitBounds
			sc.stats.Branching.ReliabilityUncertInfeasible += choice.uncertInfeasible
			sc.stats.Branching.ReliabilityLookaheadStops += choice.lookaheadStops
			if choice.ok {
				branchVar = choice.varIndex
				preferDown = choice.preferDown
				if inferenceAt(sc.inferenceDown, branchVar)+inferenceAt(sc.inferenceUp, branchVar) > 0 {
					sc.stats.Structural.InferenceBranches++
				}
			}
		}
		value := cand.values[branchVar]
		down := math.Floor(value)
		up := math.Ceil(value)
		if math.IsInf(down, 0) || math.IsInf(up, 0) {
			continue
		}
		// A certified infeasible probe is a proof for that branch, so do not
		// enqueue and solve it again. Valid probe objectives are inherited as
		// stronger subtree bounds. Strong-branch bases are deliberately discarded:
		// real children start from the parent basis, matching SCIP's restore step.
		children := make([]mipNode, 0, 2)
		left := cand.child(branchVar, true, down, value-down, choice.down)
		right := cand.child(branchVar, false, up, up-value, choice.up)
		if sc.work.vars[branchVar].Type == Binary || sc.work.vars[branchVar].Type == Integer {
			left.branchLiterals = appendBoundLiteral(cand.branchLiterals, boundLiteral{varIndex: branchVar, sense: LessOrEqual, rhs: down})
			right.branchLiterals = appendBoundLiteral(cand.branchLiterals, boundLiteral{varIndex: branchVar, sense: GreaterOrEqual, rhs: up})
		}
		if !choice.down.infeasible {
			children = append(children, left)
		} else {
			sc.stats.Branching.ReliabilityChildrenSkipped++
		}
		if !choice.up.infeasible {
			children = append(children, right)
		} else {
			sc.stats.Branching.ReliabilityChildrenSkipped++
		}
		for _, child := range children {
			if child.subtreeBoundKnown && child.subtreeBound != cand.objective {
				sc.stats.Branching.ReliabilityBoundReuses++
			}
		}
		if len(children) == 0 {
			continue
		}
		if preferDown {
			cont, err = q.add(sc, children...)
		} else {
			if len(children) == 2 {
				children[0], children[1] = children[1], children[0]
			}
			cont, err = q.add(sc, children...)
		}
		if err != nil {
			return Result{}, err
		}
		sc.recordBoundEvent(q, false)
	}
	if sc.limitStatus == Unbounded {
		return sc.withStats(Result{Status: Unbounded, Nodes: sc.nodes, Iterations: sc.iterations, DualBound: math.NaN()}), nil
	}
	dualBound := q.dualBound(sc)
	sc.recordBoundEvent(q, true)
	status := sc.limitStatus
	if status == Optimal && sc.gapPruned {
		// A later incumbent can close bounds that were originally gap-pruned.
		// Claim exact optimality only if the retained global bound now proves it.
		exactOpts := sc.opts
		exactOpts.MIPGap = 0
		if !sc.haveIncumbent || math.IsNaN(dualBound) ||
			!boundCannotImprove(sc.work.direction, dualBound, sc.bestObj, exactOpts) {
			status = GapLimit
		} else {
			// The improved incumbent now closes every retained bound within the
			// solver's exact-pruning tolerance; preserve Optimal => gap 0.
			dualBound = sc.bestObj
		}
	}
	return mipResultFromSearch(status, sc.haveIncumbent, sc.bestObj, sc.bestValues, sc.nodes, sc.iterations, dualBound, sc.withStats), nil
}

func (sc *searchContext) recordBoundEvent(q nodeQueue, force bool) {
	elapsed := time.Since(sc.start)
	if !force && elapsed < sc.nextBoundEvent {
		return
	}
	bound := q.dualBound(sc)
	if math.IsNaN(bound) || math.IsInf(bound, 0) {
		return
	}
	last := len(sc.stats.Bounds) - 1
	if last >= 0 && sc.stats.Bounds[last].Node == sc.nodes && sc.stats.Bounds[last].DualBound == bound {
		return
	}
	sc.stats.Bounds = append(sc.stats.Bounds, BoundEvent{
		Elapsed: elapsed, Node: sc.nodes, Iterations: sc.iterations, DualBound: bound,
	})
	sc.nextBoundEvent = elapsed + time.Second
}

// evaluate runs the node pipeline shared by both search modes: the LP solve
// with warm start and the numerical safeguards, the pseudocost update, bound
// pruning, the clean-LP decoupling, rounding / feasibility-pump / RINS
// incumbent work, the integer-leaf check, per-node cut generation, and
// reduced-cost fixing. On evalBranch the returned candidate carries
// everything the branching step needs.
//
// checkLimits mirrors the historical difference between the two drivers:
// best-bound checked limits at the top of its evaluate (needed for the two
// children evaluated inline right after a pop), while DFS checked once per
// pop in its loop head — dfsQueue.next does that itself and passes false so
// a node is never dropped from the stack by a limit (dfsDualBound must still
// see it as open).
func (sc *searchContext) evaluate(node mipNode, checkLimits bool) (mipCandidate, evalOutcome, error) {
	var cand mipCandidate
	opts := sc.opts
	if opts.ObjectiveCutoffPropagation && sc.haveIncumbent {
		opts.objectiveCutoffActive = true
		opts.objectiveCutoff = sc.bestObj
	}
	if checkLimits {
		if opts.MaxNodes > 0 && sc.nodes >= opts.MaxNodes {
			sc.limitStatus = NodeLimit
			sc.recordUnresolvedNode(node)
			return cand, evalStop, nil
		}
		if err := sc.ctx.Err(); err != nil {
			sc.limitStatus = TimeLimit
			sc.recordUnresolvedNode(node)
			return cand, evalStop, nil
		}
		if opts.TimeLimit > 0 && time.Since(sc.start) >= opts.TimeLimit {
			sc.limitStatus = TimeLimit
			sc.recordUnresolvedNode(node)
			return cand, evalStop, nil
		}
	}
	sc.nodes++

	if opts.UseLPConflictLearning {
		var conflictPruned bool
		node.bounds, conflictPruned = sc.applyLPConflicts(node.bounds)
		if conflictPruned {
			return cand, evalPruned, nil
		}
	}

	// Reliability probes may already have solved this exact child relaxation.
	// Its objective is a valid subtree bound even when the basis cannot be
	// reused (for example because new local cuts changed the LP shape).
	if node.subtreeBoundKnown && sc.haveIncumbent && sc.pruneByBound(node.subtreeBound) {
		return cand, evalPruned, nil
	}

	// Pseudo-objective pre-LP prune: a box-relaxation bound computed
	// from this node's own bounds with every constraint row dropped.
	// Cheaper than even canonCache's recompute, so worth checking
	// before doing any of that work.
	if opts.PseudoObjPropagation && sc.haveIncumbent {
		if pb := pseudoObjectiveBound(sc.work, node.bounds); sc.pruneByBound(pb) {
			return cand, evalPruned, nil
		}
	}

	// nodeModel = work + inherited per-node cuts. The LP, cold-solve
	// verify, and any cut generation done at this node read from this
	// augmented model so the cut rows are present and gomoryCuts can
	// resolve their slack columns back to user x-space consistently.
	//
	// Cut row aging (opt-in via CutRowAgeLimit): nodeExtras keeps every row in
	// registry order but replaces stale cuts by empty redundant slots.
	// allExtras is this node's registry-width active view; slotStructure is
	// the registry itself and provides the stable row identity/senses shared
	// by every node LP in the tree (see treeSlotViews).
	slotStructure, allExtras := sc.treeSlotViews(node.cuts)
	nodeExtras := allExtras
	if opts.CutRowAgeLimit > 0 {
		nodeExtras = relaxAgedCuts(sc.work, allExtras, node.cutAges, opts.CutRowAgeLimit)
	}
	if sc.rootSlotCount > 0 {
		active := countActiveCutPrefix(nodeExtras, sc.rootSlotCount)
		sc.stats.CutLifecycle.NodeSamples++
		sc.stats.CutLifecycle.ActiveRowsTotal += active
		sc.stats.CutLifecycle.PoolRetests += sc.rootSlotCount - active
		if active > sc.stats.CutLifecycle.PeakActiveRows {
			sc.stats.CutLifecycle.PeakActiveRows = active
		}
	}
	nodeModel := modelWithExtraConstraints(sc.work, nodeExtras)

	lpRes, lpResp, err := solveNodeOrFullWithStructure(sc.ctx, sc.workCache, nodeModel, nodeExtras, slotStructure, node.bounds, opts, node.warm)
	if err != nil {
		return cand, evalStop, err
	}
	sc.iterations += lpRes.Iterations
	sc.nodeLPIterations += lpRes.Iterations
	sc.stats.addRevised(lpResp.revised)
	sc.stats.Branching.ObjectiveCutoffReductions += lpResp.objectiveCutoffReductions
	if lpResp.objectiveCutoffPruned {
		sc.stats.Branching.ObjectiveCutoffPrunes++
	}
	if sc.nodes == 1 && lpRes.Stats != nil {
		sc.stats.Presolve = lpRes.Stats.Presolve
	}

	// Cold-solve verify: the revised backend's warm-start path can
	// occasionally return Infeasible (or, more dangerously, an Optimal
	// with the wrong objective) when the dual reoptimize lands on a
	// drift-shifted basis. The backend's internal Phase I safeguard
	// catches the cold-Phase-I case but not all warm-start cases. If a
	// warm-started LP returns Infeasible at a B&B node, re-solve cold
	// to confirm.
	if lpRes.Status == Infeasible && node.warm != nil && !lpResp.infeasibilityCertified {
		coldRes, coldResp, cerr := solveNodeOrFullWithStructure(sc.ctx, sc.workCache, nodeModel, nodeExtras, slotStructure, node.bounds, opts, nil)
		if cerr == nil && coldRes.Status != Infeasible {
			debugf("%s: warm-start LP reported Infeasible but cold solve returned %s; using cold result", sc.tag, coldRes.Status)
			lpRes = coldRes
			lpResp = coldResp
			sc.iterations += coldRes.Iterations
			sc.stats.addRevised(coldResp.revised)
		}
	}

	// Unbounded at a non-root B&B node must be numerical noise. Every
	// canonicalized variable has an explicit upper-bound row in the
	// tableau (canonical.go:275) plus the lower bound y ≥ 0, so the LP
	// is bounded by construction; branching only tightens bounds
	// further. Observed on gen-ip054 around node ~2M with the dense
	// tableau's Phase II ratio test failing to find a leave row due
	// to drift. Try a cold solve first in case warm-start was the
	// culprit; if it persists, prune as if infeasible and keep the
	// rest of the tree alive.
	if lpRes.Status == Unbounded && node.parentVar >= 0 {
		coldRes, coldResp, cerr := solveNodeOrFullWithStructure(sc.ctx, sc.workCache, nodeModel, nodeExtras, allExtras, node.bounds, opts, nil)
		if cerr == nil && coldRes.Status == Optimal {
			debugf("%s: LP reported Unbounded at non-root node (depth=%d); cold solve returned Optimal — using cold", sc.tag, node.depth)
			lpRes = coldRes
			lpResp = coldResp
			sc.iterations += coldRes.Iterations
			sc.stats.addRevised(coldResp.revised)
		} else if cerr == nil && coldRes.Status == Infeasible {
			debugf("%s: LP reported Unbounded at non-root node (depth=%d); cold solve proved Infeasible — pruning", sc.tag, node.depth)
			sc.learnLPConflict(node, coldResp.infeasibility)
			return cand, evalPruned, nil
		} else {
			// The subtree is mathematically bounded (it descends from an optimal
			// bounded root), but a numerical failure is not an infeasibility proof.
			// Continue elsewhere while retaining the inherited bound and a
			// non-optimal final status.
			debugf("%s: LP reported Unbounded at non-root node (depth=%d); cold solve = %v err=%v — retaining unresolved bound", sc.tag, node.depth, coldRes.Status, cerr)
			sc.recordUnresolvedNode(node)
			if coldRes.Status == TimeLimit || sc.ctx.Err() != nil {
				sc.limitStatus = TimeLimit
				return cand, evalStop, nil
			}
			sc.limitStatus = IterationLimit
			return cand, evalPruned, nil
		}
	}

	switch lpRes.Status {
	case Infeasible:
		debugf("%s: LP infeasible depth=%d parentVar=%d parentDown=%v bounds=%v", sc.tag, node.depth, node.parentVar, node.parentDown, node.bounds)
		sc.learnLPConflict(node, lpResp.infeasibility)
		return cand, evalPruned, nil
	case Unbounded:
		sc.limitStatus = Unbounded
		return cand, evalStop, nil
	case TimeLimit:
		sc.limitStatus = lpRes.Status
		sc.recordUnresolvedNode(node)
		return cand, evalStop, nil
	case IterationLimit:
		// A non-root node whose LP won't converge must not abort the whole
		// search and discard the incumbent. glass4's extreme degeneracy can
		// defeat the revised simplex's Bland anti-cycling on BOTH LU factors
		// at some deep nodes (float noise → tiny-but-nonzero steps that never
		// terminate within MaxIterations). Skip just this node and keep
		// exploring, like the non-root Unbounded prune above. We record the
		// numerical limit so we never claim optimality if the tree later
		// empties; the usual outcome is that the time budget is hit first and
		// the status becomes time_limit. The root LP failing is still fatal.
		sc.limitStatus = lpRes.Status
		sc.recordUnresolvedNode(node)
		if node.parentVar < 0 {
			return cand, evalStop, nil
		}
		return cand, evalPruned, nil
	case Optimal:
	default:
		sc.limitStatus = lpRes.Status
		sc.recordUnresolvedNode(node)
		return cand, evalStop, nil
	}

	// Root cover separation is LP-point driven: only violated covers are
	// candidates, and each is installed through an Optimal validation solve.
	// Validation is incremental so one numerically troublesome combination
	// cannot discard covers already shown safe (nidda's two static covers are
	// individually solvable but singular together on the revised backend).
	if node.parentVar < 0 && opts.DetectStructure && !sc.separatedStructureRoot {
		sc.separatedStructureRoot = true
		preStructureRes, preStructureResp := lpRes, lpResp
		coverCandidates := separateKnapsackCoverCuts(sc.work, lpRes.Values)
		impliedCandidates := separateImpliedBoundCuts(sc.work, sc.implications, lpRes.Values, opts.FeasibilityTol)
		flowCandidates := separateFlowCoverCuts(sc.work, sc.relations, sc.implications, lpRes.Values, opts.FeasibilityTol)
		cap := perNodeCutCap
		if opts.PerNodeCutCap > 0 {
			cap = opts.PerNodeCutCap
		}
		coverCandidates = selectCuts(coverCandidates, lpRes.Values, cap)
		impliedCandidates = selectCuts(impliedCandidates, lpRes.Values, cap)
		flowCandidates = selectCuts(flowCandidates, lpRes.Values, cap)
		candidates := append(append(coverCandidates, impliedCandidates...), flowCandidates...)
		seen := make(map[string]bool, len(sc.work.constraints)+len(nodeExtras))
		for _, c := range sc.work.constraints {
			seen[rowSignaturePermutation(c, nil)] = true
		}
		for _, c := range nodeExtras {
			seen[rowSignaturePermutation(c, nil)] = true
		}
		accepted := make([]Constraint, 0, len(candidates))
		for _, cut := range candidates {
			key := rowSignaturePermutation(cut, nil)
			if seen[key] {
				continue
			}
			trialExtras := append(append(append([]Constraint(nil), nodeExtras...), accepted...), cut)
			trialModel := modelWithExtraConstraints(sc.work, trialExtras)
			trialRes, trialResp, verr := solveNodeOrFull(sc.ctx, sc.workCache, trialModel, trialExtras, node.bounds, opts, nil)
			if verr != nil {
				return cand, evalStop, verr
			}
			sc.iterations += trialRes.Iterations
			sc.stats.addRevised(trialResp.revised)
			if trialRes.Status != Optimal {
				debugf("structure: rejected root cover %s after validation returned %s", cut.Name, trialRes.Status)
				if trialRes.Status == TimeLimit || sc.ctx.Err() != nil {
					sc.limitStatus = TimeLimit
					sc.recordUnresolvedNode(node)
					return cand, evalStop, nil
				}
				continue
			}
			accepted = append(accepted, cut)
			seen[key] = true
			lpRes, lpResp = trialRes, trialResp
		}
		if len(accepted) > 0 {
			hasClassicalCover := false
			for _, cut := range accepted {
				if cut.Name != "structure_implied_bound" && !strings.HasPrefix(cut.Name, "structure_flow_cover_") {
					hasClassicalCover = true
					break
				}
			}
			// Implied-bound/flow-cover batches are useful here only when they
			// strengthen the root objective. On glass4, 20 individually valid
			// cuts merely walked across degenerate vertices, left the bound
			// bit-identical, and made every child LP larger. Do not attach such
			// a batch to the tree. Keep the established classical cover behavior
			// (markshare proof win) independent of this new family gate.
			gain := pseudocostDegradation(sc.work.direction, preStructureRes.Objective, lpRes.Objective)
			minGain := 1e-8 * math.Max(1, math.Abs(preStructureRes.Objective))
			if !hasClassicalCover && gain <= minGain {
				debugf("structure: rejected %d implied/flow cuts: root bound gain %g <= %g", len(accepted), gain, minGain)
				accepted = nil
				lpRes, lpResp = preStructureRes, preStructureResp
			}
		}
		if len(accepted) > 0 {
			// Root structure covers bypass the registry cap: lpRes already
			// reflects an LP containing these rows, so they must be attached
			// for the branching state to stay consistent. They are few and
			// fire once, at the root.
			base := len(sc.treeSlots)
			sc.treeSlots = append(sc.treeSlots, accepted...)
			node.cuts = &cutBundle{parent: node.cuts, cuts: accepted, base: base}
			slotStructure, allExtras = sc.treeSlotViews(node.cuts)
			nodeExtras = allExtras
			if opts.CutRowAgeLimit > 0 {
				nodeExtras = relaxAgedCuts(sc.work, allExtras, node.cutAges, opts.CutRowAgeLimit)
			}
			nodeModel = modelWithExtraConstraints(sc.work, nodeExtras)
			for _, cut := range accepted {
				if cut.Name == "structure_implied_bound" {
					sc.stats.Structural.ImpliedBoundCuts++
				} else if strings.HasPrefix(cut.Name, "structure_flow_cover_") {
					sc.stats.Structural.FlowCoverCuts++
				} else {
					sc.stats.Structural.CoverCuts++
				}
			}
			sc.cutsAdded += len(accepted)
			debugf("structure: accepted %d/%d violated root cover cuts, LP bound=%g", len(accepted), len(candidates), lpRes.Objective)
		}
	}

	// Pseudocost update: now that we have this child's LP optimum,
	// retroactively record the gain/loss observed for the branch
	// that led here. Skip for the root and for negative degradations
	// (numerical drift). The bound-tightening presolve can sometimes
	// improve the LP bound at a child relative to its parent — those
	// readings are noise from the pseudocost's perspective.
	if node.parentVar >= 0 {
		deg := pseudocostDegradation(sc.work.direction, node.parentLPObj, lpRes.Objective)
		if deg > 0 {
			if node.parentDown {
				sc.pcosts.observeDown(node.parentVar, deg, node.parentStep)
			} else {
				sc.pcosts.observeUp(node.parentVar, deg, node.parentStep)
			}
		}
	}

	if sc.haveIncumbent && sc.pruneByBound(lpRes.Objective) {
		return cand, evalPruned, nil
	}

	// "Clean" LP relaxation values, used for incumbent search
	// (rounding, FP) and for branching decisions when node-local cuts or
	// symmetry rows are active. A cut-augmented LP's optimum can sit on a different
	// (often degenerate) vertex whose rounding produces strictly
	// worse integer feasibles than the un-cut LP would —
	// empirically: root MIR cuts dropped markshare_4_0 incumbent
	// from obj=1 to obj=5, similar regression on gen-ip054.
	// Solving the clean LP (no cuts, same branching bounds) lets node-local
	// cuts tighten lpRes.Objective for pruning while the search trajectory and
	// rounding heuristic stay on the no-cuts path. Global root cuts are part of
	// the fixed work model and intentionally do not take this path: branching
	// from their strengthened vertex avoids an extra LP solve at every node and
	// matches the sustained tree-separation mechanism measured in SCIP.
	//
	// Reduced-cost bound tightenings deliberately stay IN node.bounds
	// here too (tried stripping them the way cut rows are stripped —
	// measured worse on the 13-instance MIPLIB suite: 0 wins vs. 3
	// losses including a new mad regression, vs. leaving them in
	// which nets 5 wins/3 losses/5 ties. Cut rows hurt incumbents via
	// an LP-vertex shift decoupling fixes; redcost bounds hurt mainly
	// via DFS-order-changing pruning, which decoupling does not
	// undo — it just erases the wins without fixing the losses. Do
	// not re-attempt this decoupling for redcost bounds.
	cleanLPValues := lpRes.Values
	cleanLPWarm := lpResp.warm
	cleanLPModel := nodeModel
	cleanLPExtras := nodeExtras
	cleanLPCache := sc.workCache
	rootClean := node.parentVar < 0 && sc.structureChanged
	hasLocalCuts := node.cuts != nil && node.cuts.size() > sc.rootSlotCount
	cleanEveryNode := sc.cleanEveryNode || (hasLocalCuts && !opts.CutRowsDriveBranching)
	if rootClean || cleanEveryNode {
		cleanStarted := time.Now()
		cleanRes, cleanResp, cerr := solveNodeOrFull(sc.ctx, sc.cleanCache, sc.original, nil, node.bounds, opts, nil)
		if cerr == nil && cleanRes.Status == Optimal {
			cleanLPValues = cleanRes.Values
			cleanLPWarm = cleanResp.warm
			cleanLPModel = sc.original
			cleanLPExtras = nil
			cleanLPCache = sc.cleanCache
			sc.iterations += cleanRes.Iterations
			sc.stats.addRevised(cleanResp.revised)
		}
		if rootClean {
			sc.stats.Structural.RootCleanSolves++
			sc.stats.Structural.RootCleanElapsed += time.Since(cleanStarted)
		}
	}
	// Structural root decoupling is deliberately primal-only. Reliability
	// probes, the branch variable, child warm starts, and pruning must continue
	// from the strengthened root relaxation. Node-local cut decoupling keeps
	// its historical clean branching point at non-structural nodes.
	branchLPValues := cleanLPValues
	if rootClean {
		branchLPValues = lpRes.Values
	}

	{
		candObj, candValues, ok := roundedIncumbent(sc.original, cleanLPValues, opts)
		if ok && (!sc.haveIncumbent || betterObjective(sc.work.direction, candObj, sc.bestObj, opts.FeasibilityTol)) {
			sc.haveIncumbent = true
			sc.bestObj = candObj
			sc.bestValues = candValues
			sc.stats.recordIncumbent(time.Since(sc.start), sc.nodes, sc.bestObj, "rounding")
		}
	}

	// Shift-and-propagate repair starts from the same clean LP point as plain
	// rounding, but can move variables across violated rows instead of accepting
	// only an immediately feasible rounded vector.
	shouldRepair := node.parentVar < 0 || (!sc.haveIncumbent && sc.nodes%25 == 0) || sc.shouldRunIncumbentRepair()
	if shouldRepair {
		if hctx, finish, allowed := sc.heuristics.begin("repair", 0); allowed {
			_ = hctx // repair is CPU-only; begin still enforces the shared deadline.
			candObj, candValues, ok := repairRoundedIncumbent(sc.original, cleanLPValues, opts)
			finish(ok, 0)
			improved := ok && (!sc.haveIncumbent || betterObjective(sc.work.direction, candObj, sc.bestObj, opts.FeasibilityTol))
			if improved {
				sc.haveIncumbent = true
				sc.bestObj = candObj
				sc.bestValues = candValues
				sc.stats.recordIncumbent(time.Since(sc.start), sc.nodes, sc.bestObj, "repair")
			}
			sc.recordRepairResult(improved)
		}
	}

	// Once at the root, test a handful of complete discrete roundings while an
	// LP restores all continuous variables together. This complements the
	// coordinate repair on equality-heavy models with continuous recourse.
	if node.parentVar < 0 && !sc.haveIncumbent {
		if hctx, finish, allowed := sc.heuristics.begin("round_and_fix", roundFixMaxLPSolves); allowed {
			values, ok, lpSolves := runRoundAndFix(hctx, sc.original, cleanLPValues, opts, sc.cleanCache)
			finish(ok, lpSolves)
			if ok {
				if obj, feasible := feasibleIncumbent(sc.original, values, opts); feasible {
					sc.haveIncumbent = true
					sc.bestObj = obj
					sc.bestValues = values
					sc.stats.recordIncumbent(time.Since(sc.start), sc.nodes, sc.bestObj, "round_and_fix")
				}
			}
		}
	}

	// Feasibility pump: at the root only, if rounding above didn't
	// land on an incumbent, try the LP-objective-swapping iteration
	// to chase one down. Skipped when rounding already worked
	// (incumbent already attained) or when the pump already ran on
	// this solve.
	if !sc.pumpedRoot && !sc.haveIncumbent {
		sc.pumpedRoot = true
		if hctx, finish, allowed := sc.heuristics.begin("feasibility_pump", fpMaxIters); allowed {
			fpValues, ok, lpSolves := runFeasibilityPump(hctx, sc.original, cleanLPModel, cleanLPValues, opts, sc.start)
			finish(ok, lpSolves)
			if ok {
				fpObj, fpFeas := feasibleIncumbent(sc.original, fpValues, opts)
				if fpFeas {
					sc.haveIncumbent = true
					sc.bestObj = fpObj
					sc.bestValues = fpValues
					sc.stats.recordIncumbent(time.Since(sc.start), sc.nodes, sc.bestObj, "feasibility_pump")
				}
			}
		}
	}

	// Coefficient diving batches lock-guided bound changes between LP solves.
	// It reaches a substantially deeper integer path than one-variable diving
	// under the same LP budget, which is important for big-M scheduling models.
	if node.parentVar < 0 && !sc.haveIncumbent {
		if hctx, finish, allowed := sc.heuristics.begin("coefficient_diving", coefficientDivingMaxLPSolves); allowed {
			values, ok, lpSolves := runCoefficientDiving(hctx, sc.original, cleanLPModel, cleanLPExtras, node.bounds, cleanLPValues, cleanLPWarm, opts, cleanLPCache)
			finish(ok, lpSolves)
			if ok {
				if obj, feasible := feasibleIncumbent(sc.original, values, opts); feasible {
					sc.haveIncumbent = true
					sc.bestObj = obj
					sc.bestValues = values
					sc.stats.recordIncumbent(time.Since(sc.start), sc.nodes, sc.bestObj, "coefficient_diving")
				}
			}
		}
	}

	// One bounded root dive complements the pump: it preserves LP feasibility
	// after every fixing and exploits revised-simplex warm starts. It runs before
	// RENS so a long pump cannot consume the quota needed for the inexpensive
	// warm-start path. Run once even when an incumbent exists, because the
	// objective-aware path can improve it.
	if node.parentVar < 0 && !sc.divedRoot {
		sc.divedRoot = true
		if hctx, finish, allowed := sc.heuristics.begin("pseudocost_diving", divingMaxLPSolves); allowed {
			divValues, ok, lpSolves := runPseudocostDiving(hctx, sc.original, cleanLPModel, cleanLPExtras, node.bounds, cleanLPValues, cleanLPWarm, opts, sc.pcosts, cleanLPCache)
			finish(ok, lpSolves)
			if ok {
				if obj, feasible := feasibleIncumbent(sc.original, divValues, opts); feasible &&
					(!sc.haveIncumbent || betterObjective(sc.work.direction, obj, sc.bestObj, opts.FeasibilityTol)) {
					sc.haveIncumbent = true
					sc.bestObj = obj
					sc.bestValues = divValues
					sc.stats.recordIncumbent(time.Since(sc.start), sc.nodes, sc.bestObj, "pseudocost_diving")
				}
			}
		}
	}

	// RENS uses any root-neighborhood budget left after the direct dive. Fixing
	// variables already integral in the LP preserves that LP's feasibility and
	// lets a small sub-MIP explore several paths rather than committing to one.
	if node.parentVar < 0 && !sc.haveIncumbent {
		if hctx, finish, allowed := sc.heuristics.begin("rens", 20); allowed {
			rensValues, ok, lpSolves := runRENS(hctx, sc.original, cleanLPValues, opts)
			finish(ok, lpSolves)
			if ok {
				if obj, feasible := feasibleIncumbent(sc.original, rensValues, opts); feasible {
					sc.haveIncumbent = true
					sc.bestObj = obj
					sc.bestValues = rensValues
					sc.stats.recordIncumbent(time.Since(sc.start), sc.nodes, sc.bestObj, "rens")
				}
			}
		}
	}

	// Incumbent neighborhoods share one adaptive portfolio when more than one
	// is enabled. Each cadence runs at most one eligible sub-MIP; standalone
	// heuristic behavior keeps the same gates and budgets.
	sc.tryIncumbentNeighborhood(node, cleanLPValues)

	// Pseudocost warmup at the root: strong-branch each fractional
	// integer variable once so the first real branch isn't a blind
	// pick. node.parentVar == -1 identifies the root call. The
	// warmup probes use lpResp.warm (matched to the augmented LP)
	// so the values it consumes must be from the same LP — keep
	// lpRes.Values here.
	if node.parentVar < 0 && opts.MIPBranching == MIPBranchPseudocost {
		warmupPseudocost(sc.ctx, sc.work, lpRes.Values, lpRes.Objective, lpResp.warm, opts, sc.start, sc.pcosts)
	}

	if _, ok := pickBranchVar(sc.work.vars, branchLPValues, opts, sc.pcosts); !ok {
		candObj, feasible := feasibleIncumbent(sc.original, branchLPValues, opts)
		if !feasible {
			debugf("mip: rejected numerically infeasible incumbent at node depth=%d obj=%g", node.depth, lpRes.Objective)
			return cand, evalPruned, nil
		}
		if !sc.haveIncumbent || betterObjective(sc.work.direction, candObj, sc.bestObj, opts.FeasibilityTol) {
			sc.haveIncumbent = true
			sc.bestObj = candObj
			sc.bestValues = append([]float64(nil), branchLPValues...)
			sc.stats.recordIncumbent(time.Since(sc.start), sc.nodes, sc.bestObj, "integer")
		}
		return cand, evalPruned, nil
	}

	// Cut row aging: re-test every inherited cut (including ones this
	// node's own LP excluded as aged-out) against the LP point just
	// solved, so a cut that drifts back into being tight reactivates
	// for the children about to be built. See cutaging.go.
	manageCutRows := opts.CutRowAgeLimit > 0
	childCutAges := node.cutAges
	if manageCutRows {
		childCutAges = updateCutAges(allExtras, node.cutAges, lpRes.Values, opts.FeasibilityTol)
	}
	if sc.rootSlotCount > 0 && manageCutRows {
		nextExtras := allExtras
		if opts.CutRowAgeLimit > 0 {
			nextExtras = relaxAgedCuts(sc.work, allExtras, childCutAges, opts.CutRowAgeLimit)
		}
		for i := 0; i < sc.rootSlotCount && i < len(nodeExtras) && i < len(nextExtras); i++ {
			wasActive := len(nodeExtras[i].Coeffs) > 0 || nodeExtras[i].Sense == Equal
			willActive := len(nextExtras[i].Coeffs) > 0 || nextExtras[i].Sense == Equal
			switch {
			case wasActive && !willActive:
				sc.stats.CutLifecycle.RowsDeactivated++
			case !wasActive && willActive:
				sc.stats.CutLifecycle.RowsReactivated++
			}
		}
	}

	// Per-node Gomory cut generation. Off by default; opted in via
	// opts.MaxCutDepth. generatePerNodeCuts self-supplies the tableau state
	// when the node ran on the revised backend, so we attempt cuts whenever
	// the depth gate fires regardless of which backend produced lpResp.state.
	childCuts := node.cuts
	localCutCount := 0
	if node.cuts != nil {
		localCutCount = node.cuts.size() - sc.rootSlotCount
	}
	if shouldCutAtDepth(node.depth, opts, sc.cadence) && localCutCount < cutPathCap {
		newCuts, postObj, accepted := generatePerNodeCuts(sc.ctx, sc.workCache, sc.work, nodeModel, node.bounds, lpResp.state, lpResp.revisedState, nodeExtras, lpRes.Values, opts)
		base := -1
		if accepted && len(newCuts) > 0 {
			if base = sc.registerTreeSlots(newCuts); base < 0 {
				debugf("gomory: %s depth=%d dropping %d cuts: tree slot registry full (%d)", sc.cutLabel, node.depth, len(newCuts), len(sc.treeSlots))
			}
		}
		if accepted && len(newCuts) > 0 && base >= 0 {
			childCuts = &cutBundle{parent: node.cuts, cuts: newCuts, base: base}
			if manageCutRows {
				childCutAges = append(append([]int(nil), childCutAges...), make([]int, len(newCuts))...)
			}
			sc.cutsAdded += len(newCuts)
			sc.cutRounds++
			if !math.IsNaN(postObj) {
				sc.cadence.observe(lpRes.Objective, postObj)
			}
			debugf("gomory: %s depth=%d added %d cuts (subtree total %d), LP %g→%g", sc.cutLabel, node.depth, len(newCuts), childCuts.size(), lpRes.Objective, postObj)
		}
	}

	// Reduced-cost bound fixing: derive extra tightenings from this
	// node's own LP-optimal reduced costs (revised backend only). Valid
	// for the whole polytope this node's LP relaxed, so both children
	// inherit the same tightenings alongside their own branching bound.
	var redcostExtra []boundConstraint
	if opts.ReducedCostFixing && sc.haveIncumbent {
		redcostExtra = reducedCostBounds(sc.work, lpResp.revisedState, sc.work.direction, lpRes.Objective, sc.bestObj, opts)
	}

	cand = mipCandidate{
		mipNode:      node,
		objective:    lpRes.Objective,
		values:       branchLPValues,
		childWarm:    lpResp.warm,
		probeState:   lpResp.revisedState,
		relModel:     nodeModel,
		relStructure: slotStructure,
		relExtras:    nodeExtras,
	}
	cand.cuts = childCuts
	cand.cutAges = childCutAges
	cand.extraBounds = redcostExtra
	cand.estimate = mipCandidateEstimate(sc.work.direction, lpRes.Objective, sc.work.vars, branchLPValues, sc.pcosts, opts.IntegralityTol)
	return cand, evalBranch, nil
}

// dfsQueue is the depth-first open-node container: a LIFO stack of raw
// (unevaluated) nodes, evaluated at pop time.
type dfsQueue struct {
	stack []mipNode
}

func (q *dfsQueue) add(sc *searchContext, nodes ...mipNode) (bool, error) {
	// Push in reverse so the first-listed child (the down branch) is
	// explored first deterministically. Children inherit the parent's
	// optimal basis as their warm start; the revised backend
	// dual-reoptimizes from there. The tableau backend produces no warm
	// handle, so that is a no-op when LPBackend = Tableau.
	for i := len(nodes) - 1; i >= 0; i-- {
		q.stack = append(q.stack, nodes[i])
	}
	return true, nil
}

func (q *dfsQueue) next(sc *searchContext) (mipCandidate, nextStatus, error) {
	for len(q.stack) > 0 {
		if err := sc.ctx.Err(); err != nil {
			sc.limitStatus = TimeLimit
			return mipCandidate{}, nextStop, nil
		}
		if sc.opts.TimeLimit > 0 && time.Since(sc.start) >= sc.opts.TimeLimit {
			sc.limitStatus = TimeLimit
			return mipCandidate{}, nextStop, nil
		}
		if sc.opts.MaxNodes > 0 && sc.nodes >= sc.opts.MaxNodes {
			sc.limitStatus = NodeLimit
			return mipCandidate{}, nextStop, nil
		}
		node := q.stack[len(q.stack)-1]
		q.stack = q.stack[:len(q.stack)-1]
		cand, outcome, err := sc.evaluate(node, false)
		if err != nil {
			return mipCandidate{}, nextStop, err
		}
		switch outcome {
		case evalPruned:
			continue
		case evalStop:
			// Keep the remaining stack: every node still carries a valid
			// inherited parent bound and must remain in the global dual bound.
			return mipCandidate{}, nextStop, nil
		default:
			return cand, nextBranch, nil
		}
	}
	return mipCandidate{}, nextEmpty, nil
}

func (q *dfsQueue) relaxationModel(sc *searchContext, cand mipCandidate) (*Model, []Constraint, []Constraint) {
	return cand.relModel, cand.relStructure, cand.relExtras
}

func (q *dfsQueue) dualBound(sc *searchContext) float64 {
	// Global dual bound combines the remaining DFS stack with subtrees retained
	// outside the queue after gap pruning or numerical failure.
	frontier := dfsDualBound(q.stack, sc.work.direction, false, 0)
	return sc.globalDualBound(frontier)
}

// bestBoundQueue is the best-bound open-node container: candidates are
// evaluated at push time and heaped on their own LP objective, so the pop
// order follows the tightest bound and the heap front is an exact global
// dual bound.
type bestBoundQueue struct {
	open                     *mipBestBoundHeap
	plunge                   []mipCandidate
	selected                 int
	adaptiveIncumbentChecked bool
	lastAdaptiveIncumbent    float64
}

func (q *bestBoundQueue) add(sc *searchContext, nodes ...mipNode) (bool, error) {
	evaluated := make([]mipCandidate, 0, len(nodes))
	for i, node := range nodes {
		cand, outcome, err := sc.evaluate(node, true)
		if err != nil {
			return false, err
		}
		switch outcome {
		case evalPruned:
			continue
		case evalStop:
			for _, ready := range evaluated {
				heap.Push(q.open, ready)
			}
			// evaluate retained this node's inherited bound. Preserve every
			// sibling not yet evaluated as well; the parent has already left the
			// heap, so otherwise those subtrees would disappear from the proof.
			for _, pending := range nodes[i+1:] {
				sc.recordUnresolvedNode(pending)
			}
			return false, nil
		default:
			// Copy the LP values: the candidate may sit on the heap for a
			// long time; don't alias slices reused downstream.
			cand.values = append([]float64(nil), cand.values...)
			evaluated = append(evaluated, cand)
		}
	}
	// SCIP's best-estimate selector plunges through the preferred child before
	// returning to the global estimate heap. This is especially important
	// before the first incumbent: pure best-estimate can keep sampling shallow
	// nodes without ever reaching an integer leaf. The sibling remains on the
	// heap, so the global proof frontier is preserved.
	if q.open.selection == MIPBestEstimate && sc.hasGeneralInts && !sc.haveIncumbent && len(evaluated) > 0 {
		q.plunge = append(q.plunge, evaluated[0])
		evaluated = evaluated[1:]
	}
	for _, cand := range evaluated {
		heap.Push(q.open, cand)
	}
	return true, nil
}

func (q *bestBoundQueue) next(sc *searchContext) (mipCandidate, nextStatus, error) {
	for {
		if sc.haveIncumbent && len(q.plunge) > 0 {
			for _, cand := range q.plunge {
				heap.Push(q.open, cand)
			}
			q.plunge = q.plunge[:0]
		}
		if q.open.Len() == 0 && len(q.plunge) == 0 {
			return mipCandidate{}, nextEmpty, nil
		}
		q.maybeSwitchToProof(sc)
		if err := sc.ctx.Err(); err != nil {
			sc.limitStatus = TimeLimit
			return mipCandidate{}, nextStop, nil
		}
		if sc.opts.TimeLimit > 0 && time.Since(sc.start) >= sc.opts.TimeLimit {
			sc.limitStatus = TimeLimit
			return mipCandidate{}, nextStop, nil
		}
		if sc.opts.MaxNodes > 0 && sc.nodes >= sc.opts.MaxNodes {
			sc.limitStatus = NodeLimit
			return mipCandidate{}, nextStop, nil
		}
		var cand mipCandidate
		if !sc.haveIncumbent && len(q.plunge) > 0 {
			last := len(q.plunge) - 1
			cand = q.plunge[last]
			q.plunge = q.plunge[:last]
		} else if q.open.selection == MIPBestEstimate && sc.opts.BestEstimateBestBoundFreq > 0 &&
			(q.selected+1)%sc.opts.BestEstimateBestBoundFreq == 0 {
			cand = heap.Remove(q.open, q.bestBoundIndex()).(mipCandidate)
		} else {
			cand = heap.Pop(q.open).(mipCandidate)
		}
		// Staleness prune: the incumbent may have improved since this
		// candidate was evaluated and pushed.
		if sc.haveIncumbent && sc.pruneByBound(cand.objective) {
			continue
		}
		q.selected++
		return cand, nextBranch, nil
	}
}

func (q *bestBoundQueue) maybeSwitchToProof(sc *searchContext) {
	if q.open.selection != MIPBestEstimate || sc.opts.BestEstimateProofGap <= 0 || !sc.haveIncumbent || q.open.Len() == 0 {
		return
	}
	if q.adaptiveIncumbentChecked && sc.bestObj == q.lastAdaptiveIncumbent {
		return
	}
	q.adaptiveIncumbentChecked = true
	q.lastAdaptiveIncumbent = sc.bestObj
	frontier := heapDualBound(q.open, sc.work.direction, false, 0)
	gap := mipRelativeGap(frontier, sc.bestObj, true)
	if math.IsNaN(gap) || math.IsInf(gap, 0) || gap > sc.opts.BestEstimateProofGap {
		return
	}
	q.open.selection = MIPBestBound
	heap.Init(q.open)
	sc.stats.Branching.BestEstimateProofSwitches++
	sc.stats.Branching.BestEstimateProofSwitchNode = sc.nodes
	sc.stats.Branching.BestEstimateProofSwitchGap = gap
}

func (q *bestBoundQueue) bestBoundIndex() int {
	best := 0
	for i := 1; i < len(q.open.nodes); i++ {
		a, b := q.open.nodes[i], q.open.nodes[best]
		if q.open.direction == Minimize && a.objective < b.objective ||
			q.open.direction == Maximize && a.objective > b.objective {
			best = i
		}
	}
	return best
}

func (q *bestBoundQueue) relaxationModel(sc *searchContext, cand mipCandidate) (*Model, []Constraint, []Constraint) {
	structure, extras := sc.treeSlotViews(cand.cuts)
	if sc.opts.CutRowAgeLimit > 0 {
		extras = relaxAgedCuts(sc.work, extras, cand.cutAges, sc.opts.CutRowAgeLimit)
	}
	return modelWithExtraConstraints(sc.work, extras), structure, extras
}

func (q *bestBoundQueue) dualBound(sc *searchContext) float64 {
	frontier := heapDualBound(q.open, sc.work.direction, false, 0)
	for _, cand := range q.plunge {
		if math.IsNaN(frontier) || sc.work.direction == Minimize && cand.objective < frontier ||
			sc.work.direction == Maximize && cand.objective > frontier {
			frontier = cand.objective
		}
	}
	return sc.globalDualBound(frontier)
}

// mipCandidate is an evaluated node: the mipNode plus everything its LP
// solve produced that the branching step needs. The embedded mipNode's
// cuts/cutAges/extraBounds fields are the CHILD-facing versions (new cuts
// appended, ages updated, reduced-cost tightenings attached) — see
// searchContext.evaluate.
type mipCandidate struct {
	mipNode
	objective float64
	// estimate predicts the best integer objective in this subtree from the
	// current LP bound and per-variable pseudocosts. It affects best-estimate
	// ordering only; objective remains the valid pruning/dual bound.
	estimate float64
	values   []float64
	// childWarm is the basis from this candidate's own LP solve, handed
	// to both children when this candidate is branched.
	childWarm *lpWarmStart
	// probeState is the immutable final revised-simplex state of this node's
	// LP. Native strong branching clones it for every side and never adopts a
	// probe result as the next parent state.
	probeState *revisedGomoryState
	// relModel/relExtras are the node's own LP model and (age-filtered)
	// cut extras exactly as evaluated — what DFS reliability probes run
	// against (best-bound recomputes the post-cut set instead; see
	// nodeQueue.relaxationModel).
	relModel     *Model
	relStructure []Constraint
	relExtras    []Constraint
}

// child builds one branching child: the candidate's bounds plus any
// reduced-cost tightenings plus the new branching bound, inheriting the
// candidate's basis, cut chain, and cut ages.
func (c mipCandidate) child(branchVar int, down bool, bound, step float64, probe reliabilityProbeInfo) mipNode {
	sense := GreaterOrEqual
	if down {
		sense = LessOrEqual
	}
	warm := c.childWarm
	subtreeBound := c.objective
	if probe.validBound {
		subtreeBound = probe.objective
	}
	return mipNode{
		bounds:            appendBoundsAll(c.bounds, c.extraBounds, boundConstraint{varIndex: branchVar, sense: sense, rhs: bound}),
		depth:             c.depth + 1,
		branchLiterals:    c.branchLiterals,
		subtreeBound:      subtreeBound,
		subtreeBoundKnown: true,
		warm:              warm,
		cuts:              c.cuts,
		cutAges:           c.cutAges,
		parentVar:         branchVar,
		parentDown:        down,
		parentStep:        step,
		parentLPObj:       c.objective,
	}
}

type mipBestBoundHeap struct {
	direction Direction
	selection MIPNodeSelection
	nodes     []mipCandidate
}

func (h mipBestBoundHeap) Len() int { return len(h.nodes) }

func (h mipBestBoundHeap) Less(i, j int) bool {
	a, b := h.nodes[i], h.nodes[j]
	if h.selection == MIPBestEstimate {
		if h.direction == Minimize {
			if a.estimate != b.estimate {
				return a.estimate < b.estimate
			}
		} else if a.estimate != b.estimate {
			return a.estimate > b.estimate
		}
	}
	if h.direction == Minimize {
		if a.objective != b.objective {
			return a.objective < b.objective
		}
	} else if a.objective != b.objective {
		return a.objective > b.objective
	}
	if a.depth != b.depth {
		return a.depth > b.depth
	}
	return len(a.bounds) > len(b.bounds)
}

func (h mipBestBoundHeap) Swap(i, j int) {
	h.nodes[i], h.nodes[j] = h.nodes[j], h.nodes[i]
}

func (h *mipBestBoundHeap) Push(x any) {
	h.nodes = append(h.nodes, x.(mipCandidate))
}

func (h *mipBestBoundHeap) Pop() any {
	n := len(h.nodes)
	x := h.nodes[n-1]
	h.nodes = h.nodes[:n-1]
	return x
}

// mipCandidateEstimate predicts the best integer objective reachable below
// an LP-optimal node. For each fractional integer variable it charges the
// cheaper of its estimated down/up branch degradations. This is SCIP's core
// best-estimate formula, expressed in q-opt's native objective direction.
func mipCandidateEstimate(direction Direction, lpObjective float64, vars []VarSpec, values []float64, pcosts *pseudocostState, tol float64) float64 {
	if pcosts == nil || len(values) != len(vars) {
		return lpObjective
	}
	if tol <= 0 {
		tol = 1e-8
	}
	penalty := 0.0
	for i, spec := range vars {
		if spec.Type != Integer && spec.Type != Binary {
			continue
		}
		f := values[i] - math.Floor(values[i])
		if f < tol || f > 1-tol {
			continue
		}
		down := math.Max(0, f*pcosts.downGain(i))
		up := math.Max(0, (1-f)*pcosts.upGain(i))
		penalty += math.Min(down, up)
	}
	if direction == Maximize {
		return lpObjective - penalty
	}
	return lpObjective + penalty
}
func mipResultFromSearch(limitStatus Status, haveIncumbent bool, bestObj float64, bestValues []float64, nodes, iterations int, dualBound float64, withStats func(Result) Result) Result {
	if haveIncumbent {
		status := Optimal
		if limitStatus != Optimal {
			status = limitStatus
		}
		return withStats(Result{
			Status:     status,
			Objective:  bestObj,
			Values:     bestValues,
			Iterations: iterations,
			Nodes:      nodes,
			DualBound:  dualBound,
		})
	}
	if limitStatus != Optimal {
		return withStats(Result{Status: limitStatus, Nodes: nodes, Iterations: iterations, DualBound: dualBound})
	}
	return withStats(Result{Status: Infeasible, Nodes: nodes, Iterations: iterations, DualBound: dualBound})
}

// mipRelativeGap returns |incumbent − dualBound| / max(1, |incumbent|), the
// optimality gap. +Inf when there is no incumbent or no finite dual bound yet.
func mipRelativeGap(dualBound, incumbent float64, haveIncumbent bool) float64 {
	if !haveIncumbent || math.IsNaN(dualBound) || math.IsInf(dualBound, 0) {
		return math.Inf(1)
	}
	return math.Abs(incumbent-dualBound) / math.Max(1, math.Abs(incumbent))
}

// dfsDualBound computes the global dual bound from the DFS open set. Each open
// stack node's subtree optimum is no better than its parent's LP relaxation
// (branching only tightens bounds), so parentLPObj is a valid bound for that
// node; the global bound is the least-tight of them (min for Minimize, max for
// Maximize). An empty stack means the tree was exhausted, so the bound meets
// the incumbent (a proof). NaN when nothing is known yet (only the unsolved
// root remains). Clamped to the incumbent so the reported gap is never
// negative from numerical slack.
func dfsDualBound(stack []mipNode, direction Direction, haveIncumbent bool, incumbent float64) float64 {
	if len(stack) == 0 {
		if haveIncumbent {
			return incumbent
		}
		return math.NaN()
	}
	best := math.NaN()
	for _, nd := range stack {
		if nd.parentVar < 0 {
			continue // unsolved root contributes no bound
		}
		b := nd.parentLPObj
		if nd.subtreeBoundKnown {
			b = nd.subtreeBound
		}
		switch {
		case math.IsNaN(best):
			best = b
		case direction == Minimize && b < best:
			best = b
		case direction == Maximize && b > best:
			best = b
		}
	}
	if math.IsNaN(best) {
		return math.NaN()
	}
	if haveIncumbent {
		if direction == Minimize && best > incumbent {
			best = incumbent
		}
		if direction == Maximize && best < incumbent {
			best = incumbent
		}
	}
	return best
}

// heapDualBound computes the global dual bound from an evaluated-node heap.
// Under best-bound the heap front is the answer. Best-estimate is ordered by
// a heuristic score, so scan the open candidates' authoritative LP objectives.
func heapDualBound(open *mipBestBoundHeap, direction Direction, haveIncumbent bool, incumbent float64) float64 {
	if open.Len() == 0 {
		if haveIncumbent {
			return incumbent
		}
		return math.NaN()
	}
	best := open.nodes[0].objective
	if open.selection == MIPBestEstimate {
		for _, cand := range open.nodes[1:] {
			if direction == Minimize && cand.objective < best {
				best = cand.objective
			}
			if direction == Maximize && cand.objective > best {
				best = cand.objective
			}
		}
	}
	if haveIncumbent {
		if direction == Minimize && best > incumbent {
			best = incumbent
		}
		if direction == Maximize && best < incumbent {
			best = incumbent
		}
	}
	return best
}

func chooseBranchVar(vars []VarSpec, values []float64, tol float64) (int, bool) {
	best := -1
	bestFrac := 0.0
	for i, spec := range vars {
		if spec.Type != Integer && spec.Type != Binary {
			continue
		}
		frac := math.Abs(values[i] - math.Round(values[i]))
		if frac > tol && frac > bestFrac+tol {
			best = i
			bestFrac = frac
		}
	}
	return best, best >= 0
}

// pickBranchVar dispatches between the deterministic most-fractional rule
// and the pseudocost-based rule based on opts.MIPBranching. The pseudocost
// state may be nil under MIPBranchMostFractional, in which case it is not
// consulted.
func pickBranchVar(vars []VarSpec, values []float64, opts Options, pcosts *pseudocostState) (int, bool) {
	if (opts.MIPBranching == MIPBranchPseudocost || opts.MIPBranching == MIPBranchReliability) && pcosts != nil {
		return chooseBranchVarPseudocost(vars, values, opts.IntegralityTol, pcosts)
	}
	return chooseBranchVar(vars, values, opts.IntegralityTol)
}

func betterObjective(direction Direction, candidate, incumbent, tol float64) bool {
	if direction == Minimize {
		return candidate < incumbent-tol
	}
	return candidate > incumbent+tol
}

type boundPruneKind int

const (
	boundPruneNone boundPruneKind = iota
	boundPruneExact
	boundPruneGap
)

func boundPruneReason(direction Direction, bound, incumbent float64, opts Options) boundPruneKind {
	if direction == Minimize {
		if bound >= incumbent-opts.FeasibilityTol {
			return boundPruneExact
		}
		if opts.MIPGap > 0 && (incumbent-bound)/math.Max(1, math.Abs(incumbent)) <= opts.MIPGap {
			return boundPruneGap
		}
		return boundPruneNone
	}
	if bound <= incumbent+opts.FeasibilityTol {
		return boundPruneExact
	}
	if opts.MIPGap > 0 && (bound-incumbent)/math.Max(1, math.Abs(incumbent)) <= opts.MIPGap {
		return boundPruneGap
	}
	return boundPruneNone
}

func boundCannotImprove(direction Direction, bound, incumbent float64, opts Options) bool {
	return boundPruneReason(direction, bound, incumbent, opts) != boundPruneNone
}

func appendBounds(src []boundConstraint, b boundConstraint) []boundConstraint {
	dst := make([]boundConstraint, len(src), len(src)+1)
	copy(dst, src)
	dst = append(dst, b)
	return dst
}

func cloneModel(src *Model) *Model {
	dst := &Model{
		direction:   src.direction,
		vars:        append([]VarSpec(nil), src.vars...),
		objective:   make(map[Var]float64, len(src.objective)),
		constraints: make([]Constraint, len(src.constraints)),
	}
	for v, c := range src.objective {
		dst.objective[v] = c
	}
	for i := range dst.vars {
		lower, upper := *src.vars[i].Lower, *src.vars[i].Upper
		dst.vars[i].Lower = &lower
		dst.vars[i].Upper = &upper
	}
	for i, c := range src.constraints {
		coeffs := make(map[Var]float64, len(c.Coeffs))
		for v, coef := range c.Coeffs {
			coeffs[v] = coef
		}
		dst.constraints[i] = Constraint{
			Coeffs: coeffs,
			Sense:  c.Sense,
			RHS:    c.RHS,
			Name:   c.Name,
		}
	}
	return dst
}

func feasibleIncumbent(m *Model, values []float64, opts Options) (float64, bool) {
	if len(values) != len(m.vars) {
		return 0, false
	}
	lower, upper := effectiveBounds(m, nil)
	tol := opts.FeasibilityTol
	obj := 0.0
	for i, x := range values {
		if math.IsNaN(x) || math.IsInf(x, 0) {
			return 0, false
		}
		if x < lower[i]-tol || x > upper[i]+tol {
			return 0, false
		}
		if m.vars[i].Type == Integer || m.vars[i].Type == Binary {
			if math.Abs(x-math.Round(x)) > opts.IntegralityTol {
				return 0, false
			}
		}
		obj += m.objective[Var(i)] * x
	}
	rows := m.sortedRows()
	for ci, c := range m.constraints {
		lhs := 0.0
		for _, e := range rows[ci] {
			lhs += e.coef * values[e.v]
		}
		switch c.Sense {
		case LessOrEqual:
			if lhs > c.RHS+tol {
				return 0, false
			}
		case GreaterOrEqual:
			if lhs < c.RHS-tol {
				return 0, false
			}
		case Equal:
			if math.Abs(lhs-c.RHS) > tol {
				return 0, false
			}
		}
	}
	return obj, true
}

func roundedIncumbent(m *Model, values []float64, opts Options) (float64, []float64, bool) {
	if len(values) != len(m.vars) {
		return 0, nil, false
	}
	type roundMode int
	const (
		roundNearest roundMode = iota
		roundObjective
		roundDown
		roundUp
	)
	modes := []roundMode{roundNearest, roundObjective, roundDown, roundUp}
	var bestObj float64
	var bestValues []float64
	found := false

	for _, mode := range modes {
		cand := append([]float64(nil), values...)
		for i, spec := range m.vars {
			if spec.Type != Integer && spec.Type != Binary {
				continue
			}
			v := values[i]
			switch mode {
			case roundNearest:
				v = math.Round(v)
			case roundObjective:
				coef := m.objective[Var(i)]
				if (m.direction == Maximize && coef > 0) || (m.direction == Minimize && coef < 0) {
					v = math.Ceil(v)
				} else {
					v = math.Floor(v)
				}
			case roundDown:
				v = math.Floor(v)
			case roundUp:
				v = math.Ceil(v)
			}
			if spec.Type == Binary {
				if v < 0 {
					v = 0
				}
				if v > 1 {
					v = 1
				}
			}
			cand[i] = v
		}
		obj, ok := feasibleIncumbent(m, cand, opts)
		if !ok {
			continue
		}
		if !found || betterObjective(m.direction, obj, bestObj, opts.FeasibilityTol) {
			found = true
			bestObj = obj
			bestValues = cand
		}
	}
	return bestObj, bestValues, found
}
