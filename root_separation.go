package qopt

import (
	"context"
	"math"
	"time"
)

type rootSeparationResult struct {
	model *Model
	// rows is the validated, surviving global cut set appended to model.
	// The MIP driver may keep these rows outside the fixed model body and
	// install them as inherited position-stable tree slots when in-tree cut
	// lifecycle management is enabled.
	rows    []Constraint
	rounds  int
	cuts    int
	elapsed time.Duration
}

func runRootSeparation(ctx context.Context, work *Model, opts Options, stats *SolverStats) (rootSeparationResult, error) {
	if opts.MaxCutRounds <= 0 {
		return rootSeparationResult{model: work}, nil
	}
	// The persistent loop needs the revised engine (RowSet append + warm dual
	// re-solve + native GMI state). The tableau backend keeps the historical
	// clone-and-cold-solve loop.
	if opts.LPBackend == LPBackendRevised {
		return runRootSeparationPersistent(ctx, work, opts, stats)
	}
	return runRootSeparationTableau(ctx, work, opts, stats)
}

// rootCutSlot is one position-stable cut row in the persistent root LP.
// Deactivation empties the slot's coefficients in place (sense and position
// survive, so the standard-form layout and warm-start handles stay valid) and
// a later cut of the same sense may reuse the slot.
type rootCutSlot struct {
	cons    Constraint // current cut; identity for the RowSet structure
	poolIdx int        // owning rootCutPool entry, -1 if none
	age     int        // consecutive re-solves the row stayed nonbinding
	round   int        // creation round of the occupying cut
	active  bool
}

// rootSlotViews materializes the (structure, activeRows) slice pair
// solveNodeRowsWithStructure consumes: structure keeps every slot's stable
// sense, activeRows carries the live coefficients with deactivated slots
// emptied. Fresh slices are returned each call so the canonCache MRU keying
// by slice identity can never serve a stale transform after in-place slot
// mutation.
func rootSlotViews(slots []rootCutSlot) (structure, activeRows []Constraint) {
	if len(slots) == 0 {
		return nil, nil
	}
	structure = make([]Constraint, len(slots))
	activeRows = make([]Constraint, len(slots))
	for i := range slots {
		structure[i] = slots[i].cons
		if slots[i].active {
			activeRows[i] = slots[i].cons
		} else {
			activeRows[i] = Constraint{Name: slots[i].cons.Name, Sense: slots[i].cons.Sense}
		}
	}
	return structure, activeRows
}

// rootCutMappableToCache reports whether every variable in the cut has a live
// LP column in the cache (aggregation-eliminated variables have none).
func rootCutMappableToCache(cache *canonCache, cut Constraint) bool {
	for v := range cut.Coeffs {
		idx := int(v)
		if idx >= len(cache.varToCol) || cache.varToCol[idx] < 0 {
			return false
		}
	}
	return true
}

func activeRootSlotCount(slots []rootCutSlot) int {
	n := 0
	for i := range slots {
		if slots[i].active {
			n++
		}
	}
	return n
}

// runRootSeparationPersistent is the SCIP priceAndCutLoop port
// (3rd_repo/scip/src/scip/solve.c). One canonCache compiles the base model
// once; every separation round appends/reuses cut-row slots in a RowSet and
// re-solves the same LP body with the previous round's warm handle instead of
// rebuilding and cold-solving an augmented model. Round control follows
// SCIP's stall test and row aging follows lp/rowagelimit semantics; both stay
// opt-in through RootCutStallRounds and RootCutRowAgeLimit so the fixed-round
// configuration remains reproducible.
func runRootSeparationPersistent(ctx context.Context, work *Model, opts Options, stats *SolverStats) (rootSeparationResult, error) {
	result := rootSeparationResult{model: work}
	start := time.Now()
	cutCtx := ctx
	var cancel context.CancelFunc
	if opts.CutTimeLimit > 0 {
		cutCtx, cancel = context.WithTimeout(ctx, opts.CutTimeLimit)
	}
	if cancel != nil {
		defer cancel()
	}

	buildCache := buildCanonCache
	if equalityAggregationEnabled(opts) {
		buildCache = buildAggregatedCanonCache
	}
	cache := buildCache(work)
	if cache.infeasible {
		return result, nil
	}

	rootCap := perNodeCutCap
	if opts.RootCutCap > 0 {
		rootCap = opts.RootCutCap
	} else if opts.PerNodeCutCap > 0 {
		rootCap = opts.PerNodeCutCap
	}
	ageLimit := opts.RootCutRowAgeLimit
	removalCadence := 0
	if ageLimit > 0 {
		// SCIP checks obsolete rows every (rowagelimit+1)/2 + 1 solved LPs.
		removalCadence = (ageLimit+1)/2 + 1
	}
	bindingTol := opts.FeasibilityTol
	if bindingTol <= 0 {
		bindingTol = 1e-6
	}

	pool := newRootCutPool()
	var slots []rootCutSlot
	lower, upper := effectiveBounds(work, nil)

	res, resp := cache.solveNodeRowsWithStructure(cutCtx, nil, nil, nil, opts, nil)
	if res.Stats != nil {
		stats.addRevised(res.Stats.Revised)
	}
	if res.Status != Optimal {
		stats.RootSeparation = append(stats.RootSeparation, RootSeparationRoundStats{
			Round: 0, InputRows: len(work.constraints), LPStatus: res.Status,
			LPObjective: res.Objective, LPIterations: res.Iterations, Elapsed: time.Since(start),
		})
		return result, nil
	}
	warm := resp.warm
	baseObj := res.Objective

	// SCIP stall state (solve.c: stalllpobjval / stallnfracs / nsepastallrounds).
	stallObj := math.NaN()
	stallFracs := math.MaxInt32
	nStall := 0
	solves := 1

	for round := 0; round < opts.MaxCutRounds; round++ {
		if cutCtx.Err() != nil {
			break
		}
		if opts.RootCutStallRounds > 0 && nStall >= opts.RootCutStallRounds {
			break
		}

		roundStart := time.Now()
		roundStats := RootSeparationRoundStats{
			Round:        round,
			InputRows:    len(work.constraints) + activeRootSlotCount(slots),
			LPStatus:     res.Status,
			LPObjective:  res.Objective,
			LPIterations: res.Iterations,
		}

		// Generate against the current active LP (base + live cut rows), so
		// later rounds read earlier cut rows exactly like SCIP's separators
		// read the augmented LP.
		_, activeRows := rootSlotViews(slots)
		genModel := modelWithExtraConstraints(work, activeRows)
		var cuts []Constraint
		if resp.revisedState != nil && len(cache.recoveries) == 0 {
			cuts = revisedGomoryCuts(genModel, lower, upper, resp.revisedState, opts)
		} else {
			// Aggregated bodies use ranged rows the revised GMI layout does
			// not cover; recover the tableau state for this round instead
			// (same fallback generatePerNodeCuts uses).
			tabOpts := opts
			tabOpts.LPBackend = LPBackendTableau
			if _, tabState, terr := solveLPModelWithState(cutCtx, genModel, nil, tabOpts); terr == nil && tabState != nil {
				cuts = gomoryCuts(genModel, lower, upper, tabState, opts)
			}
		}
		if opts.UseMIRCuts {
			cuts = append(cuts, mirCuts(work, res.Values, opts)...)
		}
		// The aggregated cache eliminates substituted continuous variables; a
		// cut referencing one (row MIR works on original rows) has no LP
		// column to land in, so it cannot enter this loop's pool or LP.
		if len(cache.recoveries) > 0 {
			mappable := cuts[:0]
			for _, cut := range cuts {
				if rootCutMappableToCache(cache, cut) {
					mappable = append(mappable, cut)
				}
			}
			cuts = mappable
		}
		generated := makePreparedCutCandidates(genModel, cuts, res.Values, opts, round)
		addStats := pool.addCandidates(generated)
		candidates, scanStats := pool.separationCandidates(res.Values, len(work.vars), round, cutRowprepMinEfficacy)
		selected := selectCutCandidates(candidates, rootCap)
		if len(selected) == 0 {
			roundStats.Elapsed = time.Since(roundStart)
			populateRootSeparationStats(&roundStats, generated, candidates, pool, addStats, scanStats)
			stats.RootSeparation = append(stats.RootSeparation, roundStats)
			debugRootSeparationRound(roundStats)
			break
		}

		// Install the selected cuts: fill emptied same-sense slots first so
		// deactivated rows do not leak auxiliary columns forever, then append
		// new slots (warm handles survive appends — the existing prefix stays
		// byte-identical).
		type slotRollback struct {
			idx      int
			prev     rootCutSlot
			appended bool
		}
		rollbacks := make([]slotRollback, 0, len(selected))
		for _, candidateIdx := range selected {
			cand := candidates[candidateIdx]
			slotIdx := -1
			for i := range slots {
				if !slots[i].active && slots[i].cons.Sense == cand.row.Sense {
					slotIdx = i
					break
				}
			}
			if slotIdx >= 0 {
				rollbacks = append(rollbacks, slotRollback{idx: slotIdx, prev: slots[slotIdx]})
				roundStats.SlotsReused++
			} else {
				slots = append(slots, rootCutSlot{})
				slotIdx = len(slots) - 1
				rollbacks = append(rollbacks, slotRollback{idx: slotIdx, appended: true})
			}
			slots[slotIdx] = rootCutSlot{
				cons:    cloneConstraint(cand.row),
				poolIdx: cand.poolIndex,
				round:   round,
				active:  true,
			}
		}

		structure, activeRows := rootSlotViews(slots)
		postRes, postResp := cache.solveNodeRowsWithStructure(cutCtx, nil, structure, activeRows, opts, warm)
		if postRes.Stats != nil {
			stats.addRevised(postRes.Stats.Revised)
		}
		solves++
		roundStats.WarmSolve = postResp.revised.WarmReopts > 0 && postResp.revised.WarmFailures == 0
		if postRes.Status != Optimal && cutCtx.Err() == nil && warm != nil {
			// SCIP's lpSolveStable escalation: on numerical trouble, retry the
			// same LP from scratch before declaring the round failed. The
			// warm dual reoptimize occasionally walks into a degenerate or
			// ill-conditioned basis a cold Phase I avoids.
			debugf("rootcuts: round=%d warm re-solve returned %s; retrying cold", round, postRes.Status)
			postRes, postResp = cache.solveNodeRowsWithStructure(cutCtx, nil, structure, activeRows, opts, nil)
			if postRes.Stats != nil {
				stats.addRevised(postRes.Stats.Revised)
			}
			roundStats.WarmSolve = false
		}
		roundStats.ValidationAttempted = true
		roundStats.ValidationStatus = postRes.Status
		roundStats.ValidationObjective = postRes.Objective
		roundStats.ValidationIterations = postRes.Iterations
		if postRes.Status != Optimal {
			// Numerical limit or worse: roll the slot mutations back and stop
			// separating. The bound from the previous round is retained.
			for i := len(rollbacks) - 1; i >= 0; i-- {
				rb := rollbacks[i]
				if rb.appended {
					slots = slots[:rb.idx]
				} else {
					slots[rb.idx] = rb.prev
				}
			}
			for _, idx := range selected {
				candidates[idx].state = cutCandidateRejectedValidation
			}
			roundStats.Elapsed = time.Since(roundStart)
			populateRootSeparationStats(&roundStats, generated, candidates, pool, addStats, scanStats)
			stats.RootSeparation = append(stats.RootSeparation, roundStats)
			debugRootSeparationRound(roundStats)
			break
		}

		for _, idx := range selected {
			candidates[idx].state = cutCandidateApplied
		}
		pool.setActive(candidates, selected, true)
		result.rounds++
		result.cuts += len(selected)
		roundStats.ObjectiveGain = rootSeparationObjectiveGain(work.direction, res.Objective, postRes.Objective)

		// Row aging (SCIP SCIPlpUpdateAges + lpRemoveObsoleteRows): binding
		// rows reset to age 0, nonbinding rows age by one per solve, and on
		// the removal cadence any over-age inequality row is emptied in place
		// and its pool entry becomes retestable again.
		if ageLimit > 0 {
			for i := range slots {
				if !slots[i].active {
					continue
				}
				if cutIsBinding(slots[i].cons, postRes.Values, bindingTol) {
					slots[i].age = 0
				} else {
					slots[i].age++
				}
			}
			if removalCadence > 0 && solves%removalCadence == 0 {
				for i := range slots {
					if !slots[i].active || slots[i].age <= ageLimit || slots[i].cons.Sense == Equal {
						continue
					}
					slots[i].active = false
					slots[i].age = 0
					pool.setActiveIndex(slots[i].poolIdx, false)
					roundStats.Deactivated++
				}
			}
		}

		// SCIP stall test: an optimal re-solve stalls when the objective
		// moved by at most 1e-4 relative and the fractional count did not
		// drop below (0.9 - 0.1*nsepastallrounds) of the previous count.
		fracs := countFractionalDiscrete(work, postRes.Values, opts.IntegralityTol)
		if !math.IsNaN(stallObj) {
			gain := rootSeparationObjectiveGain(work.direction, stallObj, postRes.Objective)
			scale := math.Max(math.Max(math.Abs(stallObj), math.Abs(postRes.Objective)), 1)
			stalling := gain/scale <= 1e-4 &&
				float64(fracs) >= (0.9-0.1*float64(nStall))*float64(stallFracs)
			if stalling {
				nStall++
			} else {
				nStall = 0
			}
		}
		stallObj = postRes.Objective
		stallFracs = fracs
		roundStats.Fractionals = fracs
		roundStats.StallRounds = nStall

		roundStats.Elapsed = time.Since(roundStart)
		populateRootSeparationStats(&roundStats, generated, candidates, pool, addStats, scanStats)
		stats.RootSeparation = append(stats.RootSeparation, roundStats)
		debugRootSeparationRound(roundStats)

		res, resp, warm = postRes, postResp, postResp.warm
	}

	// Bake the surviving active rows into the model the search will use, then
	// cold-validate once with the production backend (the Stage-1 contract:
	// the root loop and the production root solve must not disagree). On a
	// failed validation, drop the most recent creation rounds until the model
	// validates or no cuts remain.
	// Zero-gain gate: if the whole separation loop failed to move the root
	// objective by even the stall test's relative epsilon, the cuts are pure
	// LP ballast — they slow every node without strengthening the relaxation
	// the tree inherits (measured: they cost markshare_4_0's proof and
	// glass4's entire incumbent trajectory). SCIP survives such rows because
	// its LP and heuristics are hardier; q-opt drops them.
	if opts.RootCutStallRounds > 0 && result.rounds > 0 {
		totalGain := rootSeparationObjectiveGain(work.direction, baseObj, res.Objective)
		scale := math.Max(math.Max(math.Abs(baseObj), math.Abs(res.Objective)), 1)
		if totalGain/scale <= 1e-4 {
			debugf("rootcuts: total root gain %g (rel %g) below threshold; dropping all %d active cuts",
				totalGain, totalGain/scale, activeRootSlotCount(slots))
			result.elapsed = time.Since(start)
			result.cuts = 0
			return result, nil
		}
	}

	// SCIP lp/cleanuprowsroot (default TRUE): after root LP solving, rows
	// whose slack is basic (nonbinding at the final root optimum) are removed
	// from the LP the tree inherits. Gated on the same lifecycle opt-in as
	// aging so fixed-round configurations keep their historical cut sets.
	if ageLimit > 0 && res.Status == Optimal && res.Values != nil {
		removed := 0
		for i := range slots {
			if !slots[i].active || slots[i].cons.Sense == Equal {
				continue
			}
			if !cutIsBinding(slots[i].cons, res.Values, bindingTol) {
				slots[i].active = false
				pool.setActiveIndex(slots[i].poolIdx, false)
				removed++
			}
		}
		if removed > 0 {
			debugf("rootcuts: cleanuprowsroot removed %d nonbinding rows (%d remain active)", removed, activeRootSlotCount(slots))
		}
	}

	result.cuts = 0
	if activeRootSlotCount(slots) == 0 {
		result.elapsed = time.Since(start)
		return result, nil
	}
	for {
		final := cloneModel(work)
		applied := 0
		for i := range slots {
			if !slots[i].active {
				continue
			}
			c := slots[i].cons
			if err := final.AddNamedConstraint(c.Name, c.Coeffs, c.Sense, c.RHS); err != nil {
				return rootSeparationResult{}, err
			}
			applied++
		}
		if applied == 0 {
			break
		}
		coldRes, _, err := solveLPModelWithState(cutCtx, final, nil, opts)
		if err != nil {
			return rootSeparationResult{}, err
		}
		if coldRes.Stats != nil {
			stats.addRevised(coldRes.Stats.Revised)
		}
		if coldRes.Status == Optimal {
			result.model = final
			result.cuts = applied
			result.rows = append(result.rows[:0], final.constraints[len(work.constraints):]...)
			break
		}
		// Deactivate the newest creation round and retry.
		newest := -1
		for i := range slots {
			if slots[i].active && slots[i].round > newest {
				newest = slots[i].round
			}
		}
		debugf("rootcuts: final cold validation %s with %d rows; dropping round %d", coldRes.Status, applied, newest)
		for i := range slots {
			if slots[i].active && slots[i].round == newest {
				slots[i].active = false
				pool.setActiveIndex(slots[i].poolIdx, false)
			}
		}
	}
	if len(result.model.constraints) > len(work.constraints) {
		result.rows = append(result.rows[:0], result.model.constraints[len(work.constraints):]...)
	}
	result.elapsed = time.Since(start)
	return result, nil
}

func runRootSeparationTableau(ctx context.Context, work *Model, opts Options, stats *SolverStats) (rootSeparationResult, error) {
	result := rootSeparationResult{model: work}

	result.model = cloneModel(work)
	start := time.Now()
	cutCtx := ctx
	var cancel context.CancelFunc
	if opts.CutTimeLimit > 0 {
		cutCtx, cancel = context.WithTimeout(ctx, opts.CutTimeLimit)
	}
	if cancel != nil {
		defer cancel()
	}

	cutOpts := opts
	cutOpts.LPBackend = LPBackendTableau
	pool := newRootCutPool()
	for round := 0; round < opts.MaxCutRounds; round++ {
		if cutCtx.Err() != nil {
			break
		}

		roundStart := time.Now()
		roundStats := RootSeparationRoundStats{Round: round, InputRows: len(result.model.constraints)}
		lpRes, state, err := solveLPModelWithState(cutCtx, result.model, nil, cutOpts)
		if err != nil {
			return rootSeparationResult{}, err
		}
		if lpRes.Stats != nil {
			stats.addRevised(lpRes.Stats.Revised)
		}
		roundStats.LPStatus = lpRes.Status
		roundStats.LPObjective = lpRes.Objective
		roundStats.LPIterations = lpRes.Iterations
		if lpRes.Status != Optimal {
			roundStats.Elapsed = time.Since(roundStart)
			stats.RootSeparation = append(stats.RootSeparation, roundStats)
			break
		}

		var cuts []Constraint
		if state != nil {
			modelLower, modelUpper := effectiveBounds(result.model, nil)
			cuts = gomoryCuts(result.model, modelLower, modelUpper, state, opts)
		}
		if opts.UseMIRCuts {
			cuts = append(cuts, mirCuts(result.model, lpRes.Values, opts)...)
		}

		rootCap := perNodeCutCap
		if opts.RootCutCap > 0 {
			rootCap = opts.RootCutCap
		} else if opts.PerNodeCutCap > 0 {
			rootCap = opts.PerNodeCutCap
		}
		generated := makePreparedCutCandidates(result.model, cuts, lpRes.Values, opts, round)
		addStats := pool.addCandidates(generated)
		candidates, scanStats := pool.separationCandidates(
			lpRes.Values,
			len(result.model.vars),
			round,
			cutRowprepMinEfficacy,
		)
		selected := selectCutCandidates(candidates, rootCap)
		selectedCuts := selectedCandidateCuts(candidates, selected)
		if len(selectedCuts) == 0 {
			roundStats.Elapsed = time.Since(roundStart)
			populateRootSeparationStats(&roundStats, generated, candidates, pool, addStats, scanStats)
			stats.RootSeparation = append(stats.RootSeparation, roundStats)
			debugRootSeparationRound(roundStats)
			break
		}

		before := len(result.model.constraints)
		for _, cut := range selectedCuts {
			if err := result.model.AddNamedConstraint(cut.Name, cut.Coeffs, cut.Sense, cut.RHS); err != nil {
				return rootSeparationResult{}, err
			}
		}

		postRes, _, err := solveLPModelWithState(cutCtx, result.model, nil, opts)
		if err != nil {
			return rootSeparationResult{}, err
		}
		if postRes.Stats != nil {
			stats.addRevised(postRes.Stats.Revised)
		}
		roundStats.ValidationAttempted = true
		roundStats.ValidationStatus = postRes.Status
		roundStats.ValidationObjective = postRes.Objective
		roundStats.ValidationIterations = postRes.Iterations
		if postRes.Status != Optimal {
			for _, idx := range selected {
				candidates[idx].state = cutCandidateRejectedValidation
			}
			result.model.constraints = result.model.constraints[:before]
			roundStats.Elapsed = time.Since(roundStart)
			populateRootSeparationStats(&roundStats, generated, candidates, pool, addStats, scanStats)
			stats.RootSeparation = append(stats.RootSeparation, roundStats)
			debugRootSeparationRound(roundStats)
			break
		}
		roundStats.ObjectiveGain = rootSeparationObjectiveGain(result.model.direction, lpRes.Objective, postRes.Objective)

		for _, idx := range selected {
			candidates[idx].state = cutCandidateApplied
		}
		pool.setActive(candidates, selected, true)
		result.rounds++
		result.cuts += len(selectedCuts)
		roundStats.Elapsed = time.Since(roundStart)
		populateRootSeparationStats(&roundStats, generated, candidates, pool, addStats, scanStats)
		stats.RootSeparation = append(stats.RootSeparation, roundStats)
		debugRootSeparationRound(roundStats)
	}
	if len(result.model.constraints) > len(work.constraints) {
		result.rows = append(result.rows[:0], result.model.constraints[len(work.constraints):]...)
	}
	result.elapsed = time.Since(start)
	return result, nil
}

func rootSeparationObjectiveGain(direction Direction, before, after float64) float64 {
	if direction == Maximize {
		return before - after
	}
	return after - before
}

func populateRootSeparationStats(
	dst *RootSeparationRoundStats,
	generated []cutCandidate,
	candidates []cutCandidate,
	pool *rootCutPool,
	addStats rootCutPoolAddStats,
	scanStats rootCutPoolScanStats,
) {
	if dst == nil {
		return
	}
	dst.Generated = len(generated)
	dst.PoolSize = pool.size()
	dst.PoolAdded = addStats.added
	dst.PoolDuplicates = addStats.duplicates
	dst.PoolCandidates = scanStats.violated
	dst.PoolTested = scanStats.tested
	dst.PoolRetested = scanStats.retested
	dst.PoolReusedCandidates = scanStats.reusedCandidates
	dst.PoolAgedOut = scanStats.agedOut
	dst.ActiveRows = pool.activeCount()
	dst.Families = make(map[string]CutFamilyRoundStats)
	dst.Rejections = make(map[string]int)
	for _, candidate := range generated {
		family := dst.Families[candidate.family]
		family.Generated++
		if candidate.rowprepModified {
			dst.RowprepModified++
			family.RowprepModified++
		}
		dst.CoefficientsRemoved += candidate.rowprepRemoved
		dst.CoefficientsRounded += candidate.rowprepRounded
		dst.NonzerosBefore += candidate.preNonzeros
		dst.NonzerosAfter += len(candidate.row.Coeffs)
		family.CoefficientsRemoved += candidate.rowprepRemoved
		family.CoefficientsRounded += candidate.rowprepRounded
		if candidate.state != cutCandidateGenerated {
			dst.Rejections[string(candidate.state)]++
		}
		dst.MaxEfficacy = math.Max(dst.MaxEfficacy, candidate.efficacy)
		dst.MaxEfficacyBefore = math.Max(dst.MaxEfficacyBefore, candidate.preEfficacy)
		dst.MaxReferenceLoss = math.Max(dst.MaxReferenceLoss, candidate.rowprepLoss)
		dst.MaxDensity = math.Max(dst.MaxDensity, candidate.density)
		dst.MaxCoefficientBefore = math.Max(dst.MaxCoefficientBefore, candidate.preMaxCoefficient)
		dst.MaxCoefficient = math.Max(dst.MaxCoefficient, candidate.maxCoefficient)
		dst.MaxCoefficientRatioBefore = math.Max(dst.MaxCoefficientRatioBefore, candidate.preCoefficientRatio)
		dst.MaxCoefficientRatio = math.Max(dst.MaxCoefficientRatio, candidate.coefficientRatio)
		dst.Families[candidate.family] = family
	}
	for _, candidate := range candidates {
		family := dst.Families[candidate.family]
		switch candidate.state {
		case cutCandidateSelected:
			dst.Selected++
			family.Selected++
			if candidate.creationRound < dst.Round {
				dst.PoolSelectedRetained++
			}
		case cutCandidateApplied:
			dst.Selected++
			dst.Applied++
			family.Selected++
			family.Applied++
			if candidate.creationRound < dst.Round {
				dst.PoolSelectedRetained++
			}
		case cutCandidateRejectedValidation:
			dst.Selected++
			family.Selected++
			if candidate.creationRound < dst.Round {
				dst.PoolSelectedRetained++
			}
			dst.Rejections[string(candidate.state)]++
		default:
			dst.Rejections[string(candidate.state)]++
		}
		dst.Families[candidate.family] = family
	}
}

func debugRootSeparationRound(round RootSeparationRoundStats) {
	debugf("rootcuts: round=%d input_rows=%d generated=%d pool=%d added=%d duplicate=%d tested=%d retested=%d candidates=%d reused=%d retained_selected=%d agedout=%d active=%d selected=%d applied=%d rowprep=%d removed=%d rounded=%d nnz=%d->%d lp=%s obj=%g iter=%d validation=%s postobj=%g postiter=%d gain=%g elapsed=%s families=%v rejected=%v maxloss=%g maxeff=%g->%g maxdensity=%g maxcoef=%g->%g maxratio=%g->%g",
		round.Round, round.InputRows, round.Generated,
		round.PoolSize, round.PoolAdded, round.PoolDuplicates, round.PoolTested, round.PoolRetested,
		round.PoolCandidates, round.PoolReusedCandidates, round.PoolSelectedRetained, round.PoolAgedOut, round.ActiveRows,
		round.Selected, round.Applied,
		round.RowprepModified, round.CoefficientsRemoved, round.CoefficientsRounded, round.NonzerosBefore, round.NonzerosAfter,
		round.LPStatus, round.LPObjective, round.LPIterations,
		round.ValidationStatus, round.ValidationObjective, round.ValidationIterations, round.ObjectiveGain, round.Elapsed,
		round.Families, round.Rejections, round.MaxReferenceLoss,
		round.MaxEfficacyBefore, round.MaxEfficacy, round.MaxDensity,
		round.MaxCoefficientBefore, round.MaxCoefficient, round.MaxCoefficientRatioBefore, round.MaxCoefficientRatio)
	if round.Fractionals > 0 || round.StallRounds > 0 || round.Deactivated > 0 || round.SlotsReused > 0 || round.WarmSolve {
		debugf("rootcuts: round=%d fracs=%d stall=%d deactivated=%d reused_slots=%d warm=%t",
			round.Round, round.Fractionals, round.StallRounds, round.Deactivated, round.SlotsReused, round.WarmSolve)
	}
}
