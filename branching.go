package qopt

import (
	"context"
	"math"
	"sort"
	"time"
)

// Pseudocost branching for the B&B variable-selection step.
//
// For each integer/binary variable i the solver tracks the average LP-
// objective degradation per unit of "down-step" and "up-step" seen across
// past B&B children that branched on i. A child branching x_i ≤ floor(v)
// from a parent value v gives a down-step of (v − floor(v)); a child
// branching x_i ≥ ceil(v) gives an up-step of (ceil(v) − v). The
// observed degradation is normalized by the step size, so the unit is
// "objective units per unit of x movement".
//
// Until at least one observation exists globally, vars without local data
// fall back to a most-fractional measure f·(1−f). After data accumulates,
// uninitialized variables use the global average as a prior; well-observed
// variables use their own running average. The branching score combines
// the two directions multiplicatively (the classic Linderoth–Savelsbergh
// score), which empirically dominates a weighted-max score on small MIPs.

type pseudocostState struct {
	downSum   []float64
	downCount []float64
	upSum     []float64
	upCount   []float64

	globalDownSum   float64
	globalDownCount float64
	globalUpSum     float64
	globalUpCount   float64
}

func newPseudocostState(nvars int) *pseudocostState {
	return &pseudocostState{
		downSum:   make([]float64, nvars),
		downCount: make([]float64, nvars),
		upSum:     make([]float64, nvars),
		upCount:   make([]float64, nvars),
	}
}

// observeDown records a degradation on a down-branch (x_i ≤ floor(parent_v)).
// step is (parent_v − floor(parent_v)); degradation is the non-negative
// LP-objective movement from parent to child (parent_obj − child_obj for
// maximize, child_obj − parent_obj for minimize). The caller is expected
// to have normalized the sign.
func (p *pseudocostState) observeDown(varIdx int, degradation, step float64) {
	if step <= 1e-9 || degradation < 0 {
		return
	}
	gain := degradation / step
	p.downSum[varIdx] += gain
	p.downCount[varIdx]++
	p.globalDownSum += gain
	p.globalDownCount++
}

func (p *pseudocostState) observeUp(varIdx int, degradation, step float64) {
	if step <= 1e-9 || degradation < 0 {
		return
	}
	gain := degradation / step
	p.upSum[varIdx] += gain
	p.upCount[varIdx]++
	p.globalUpSum += gain
	p.globalUpCount++
}

// downGain returns the estimated down-direction gain for varIdx. When this
// variable has its own observations the running average wins; otherwise
// the global average is used as a Bayesian-ish prior. Returns 0 when no
// observations exist anywhere yet (callers handle that as "cold").
func (p *pseudocostState) downGain(varIdx int) float64 {
	if p.downCount[varIdx] > 0 {
		return p.downSum[varIdx] / p.downCount[varIdx]
	}
	if p.globalDownCount > 0 {
		return p.globalDownSum / p.globalDownCount
	}
	return 0
}

func (p *pseudocostState) upGain(varIdx int) float64 {
	if p.upCount[varIdx] > 0 {
		return p.upSum[varIdx] / p.upCount[varIdx]
	}
	if p.globalUpCount > 0 {
		return p.globalUpSum / p.globalUpCount
	}
	return 0
}

// score combines the down and up estimated losses for a fractional
// variable. f is the raw fractional part (v − floor(v)) ∈ (0, 1). The
// product form rewards variables that are constraining in BOTH directions,
// which is the textbook reliability heuristic.
func (p *pseudocostState) score(varIdx int, f float64) float64 {
	if f <= 0 || f >= 1 {
		return 0
	}
	const eps = 1e-6
	if p.globalDownCount == 0 && p.globalUpCount == 0 {
		// Cold start: most-fractional measure.
		return f * (1 - f)
	}
	d := math.Max(eps, f*p.downGain(varIdx))
	u := math.Max(eps, (1-f)*p.upGain(varIdx))
	return d * u
}

// chooseBranchVarPseudocost returns the fractional integer variable with
// the highest pseudocost score, or (-1, false) when no fractional integer
// var exists.
func chooseBranchVarPseudocost(vars []VarSpec, values []float64, tol float64, p *pseudocostState) (int, bool) {
	best := -1
	bestScore := -math.MaxFloat64
	for i, spec := range vars {
		if spec.Type != Integer && spec.Type != Binary {
			continue
		}
		f := values[i] - math.Floor(values[i])
		// Skip values too close to integer at either side.
		if f < tol || f > 1-tol {
			continue
		}
		score := p.score(i, f)
		if score > bestScore {
			bestScore = score
			best = i
		}
	}
	return best, best >= 0
}

// warmupPseudocost runs a strong-branching pass over the fractional integer
// variables at the root, recording the actual LP-objective degradation of
// each down/up probe in the pseudocost table. Without this, pure pseudocost
// branching cold-starts with no observations and the first few branches
// are essentially random — costing several MIPLIB instances their best
// incumbent. With warmup, every fractional-at-root variable has at least
// one real reading before B&B starts.
//
// Each probe re-solves the root LP with one added bound (x_j ≤ floor(v) or
// x_j ≥ ceil(v)). The revised backend's dual reoptimize warm-starts each
// probe from `rootWarm`, so a probe is roughly the cost of one or two dual
// pivots — much cheaper than a cold solve. The whole warmup is capped at
// `warmupDeadline` so we don't burn an unbounded fraction of the time
// budget; on remaining unprobed variables we fall back to the global
// pseudocost average at branching time.
func warmupPseudocost(
	ctx context.Context,
	work *Model,
	rootLPValues []float64,
	rootLPObj float64,
	rootWarm *lpWarmStart,
	opts Options,
	start time.Time,
	pcosts *pseudocostState,
) {
	if pcosts == nil || work == nil || len(rootLPValues) != len(work.vars) {
		return
	}
	tol := opts.IntegralityTol
	if tol <= 0 {
		tol = 1e-8
	}

	// Collect fractional integer vars; sort most-fractional first so we
	// probe the variables that matter even when the time cap truncates us.
	type cand struct {
		idx  int
		frac float64
	}
	var cands []cand
	for i, spec := range work.vars {
		if spec.Type != Integer && spec.Type != Binary {
			continue
		}
		v := rootLPValues[i]
		f := v - math.Floor(v)
		if f < tol || f > 1-tol {
			continue
		}
		cands = append(cands, cand{i, f})
	}
	if len(cands) == 0 {
		return
	}
	sort.Slice(cands, func(a, b int) bool {
		return math.Abs(cands[a].frac-0.5) < math.Abs(cands[b].frac-0.5)
	})

	// Per-probe iteration cap: tight, since each probe is a dual-reoptimize
	// from the root basis, not a full solve. 200 dual pivots almost always
	// suffice when one variable's bound just tightened.
	probeOpts := opts
	probeOpts.MaxIterations = 200
	probeOpts.TimeLimit = 0
	probeOpts.MaxCutRounds = 0

	// Warmup wall-clock budget: at most 10% of the overall time limit, and
	// at most 5 seconds. If no time limit is set, we use 5 seconds.
	budget := 5 * time.Second
	if opts.TimeLimit > 0 {
		if tenth := opts.TimeLimit / 10; tenth < budget {
			budget = tenth
		}
	}
	deadline := start.Add(budget)

	probes := 0
	for _, c := range cands {
		if time.Now().After(deadline) {
			break
		}
		if err := ctx.Err(); err != nil {
			break
		}
		j := c.idx
		v := rootLPValues[j]
		floor := math.Floor(v)
		ceil := math.Ceil(v)

		// Down probe: x_j ≤ floor(v).
		downBounds := []boundConstraint{{varIndex: j, sense: LessOrEqual, rhs: floor}}
		downRes, _, _, err := solveLPModelWithBackend(ctx, work, downBounds, probeOpts, rootWarm)
		if err == nil && downRes.Status == Optimal {
			deg := pseudocostDegradation(work.direction, rootLPObj, downRes.Objective)
			if deg > 0 {
				pcosts.observeDown(j, deg, v-floor)
			}
		}

		if time.Now().After(deadline) {
			break
		}

		// Up probe: x_j ≥ ceil(v).
		upBounds := []boundConstraint{{varIndex: j, sense: GreaterOrEqual, rhs: ceil}}
		upRes, _, _, err := solveLPModelWithBackend(ctx, work, upBounds, probeOpts, rootWarm)
		if err == nil && upRes.Status == Optimal {
			deg := pseudocostDegradation(work.direction, rootLPObj, upRes.Objective)
			if deg > 0 {
				pcosts.observeUp(j, deg, ceil-v)
			}
		}
		probes++
	}
	debugf("warmup: probed %d/%d fractional vars in %s", probes, len(cands), time.Since(start))
}

// Reliability branching uses SCIP's estimate/bound/proof semantics and
// lookahead stop, with q-opt-calibrated candidate width. SoPlex can afford a
// much wider initialization queue; q-opt's measured default remains eight.
const (
	defaultReliabilityThreshold = 5
	defaultReliabilityMaxProbes = 8
	reliabilityMaxLookahead     = 9
)

type reliabilityProbeInfo struct {
	objective     float64
	validBound    bool
	infeasible    bool
	estimate      float64
	validEstimate bool
}

type reliabilityChoice struct {
	varIndex         int
	ok               bool
	preferDown       bool
	down             reliabilityProbeInfo
	up               reliabilityProbeInfo
	candidates       int
	lpSolves         int
	iterations       int
	infeasibleSides  int
	domainReductions int
	selectedProbed   int
	discardedProbes  int
	nativeProbes     int
	fallbackProbes   int
	iterationLimits  int
	validLimitBounds int
	uncertInfeasible int
	lookaheadStops   int
	revised          RevisedStats
}

// chooseBranchVarReliability picks the branch variable using reliability
// branching: for each fractional candidate whose pseudocost is not yet
// reliable, it runs a down- and up-probe (warm-started LP solves on nodeModel
// with the branched bound applied) and folds the real degradation into the
// pseudocost table. After probing, the variable with the best (now possibly
// strong-branching-fresh) score wins. values/baseObj come from the node's own
// LP solve so probe degradations are measured against the right baseline; warm
// is that LP's basis so each probe is a cheap reoptimize on the revised
// backend (a cold solve on the tableau backend, where warm is nil).
func chooseBranchVarReliability(
	ctx context.Context,
	work, nodeModel *Model,
	nodeExtras []Constraint,
	baseBounds []boundConstraint,
	values []float64,
	baseObj float64,
	warm *lpWarmStart,
	parentState *revisedGomoryState,
	opts Options,
	pcosts *pseudocostState,
	cache *canonCache,
	nodeStructure []Constraint,
	inferenceDown, inferenceUp []int,
	iterationBudget int,
	probeIterationLimit int,
) reliabilityChoice {
	choice := reliabilityChoice{varIndex: -1, preferDown: true}
	tol := opts.IntegralityTol
	if tol <= 0 {
		tol = 1e-8
	}
	if probeIterationLimit <= 0 {
		probeIterationLimit = 10
	}
	remaining := iterationBudget
	native := cache.newStrongBranchSession(parentState, baseBounds, nodeStructure, nodeExtras, opts)

	type cand struct {
		idx  int
		frac float64
	}
	var cands []cand
	for i, spec := range work.vars {
		if spec.Type != Integer && spec.Type != Binary {
			continue
		}
		f := values[i] - math.Floor(values[i])
		if f < tol || f > 1-tol {
			continue
		}
		cands = append(cands, cand{i, f})
	}
	if len(cands) == 0 {
		return choice
	}
	downScores := make(map[int]float64, len(cands))
	upScores := make(map[int]float64, len(cands))
	for _, c := range cands {
		downScores[c.idx] = math.Max(0, c.frac*pcosts.downGain(c.idx))
		upScores[c.idx] = math.Max(0, (1-c.frac)*pcosts.upGain(c.idx))
	}

	// Probe the most promising candidates first (current score), so that when
	// maxReliabilityProbes truncates we've spent the budget where it matters.
	sort.Slice(cands, func(a, b int) bool {
		return inferenceAdjustedScore(pcosts.score(cands[a].idx, cands[a].frac), inferenceAt(inferenceDown, cands[a].idx), inferenceAt(inferenceUp, cands[a].idx)) >
			inferenceAdjustedScore(pcosts.score(cands[b].idx, cands[b].frac), inferenceAt(inferenceDown, cands[b].idx), inferenceAt(inferenceUp, cands[b].idx))
	})

	probeOpts := opts
	probeOpts.TimeLimit = 0
	probeOpts.MaxCutRounds = 0
	probeOpts.MaxCutDepth = 0

	probeLimit := opts.ReliabilityMaxProbes
	if probeLimit <= 0 {
		probeLimit = defaultReliabilityMaxProbes
	}
	reliable := opts.ReliabilityThreshold
	if reliable <= 0 {
		reliable = defaultReliabilityThreshold
	}
	probeResultsDown := make(map[int]reliabilityProbeInfo, len(cands))
	probeResultsUp := make(map[int]reliabilityProbeInfo, len(cands))
	probeInferenceDown := make(map[int]int, len(cands))
	probeInferenceUp := make(map[int]int, len(cands))
	probedCandidates := make(map[int]bool, len(cands))
	runProbe := func(bounds []boundConstraint) (reliabilityProbeInfo, int, int, RevisedStats, bool) {
		if remaining <= 0 {
			return reliabilityProbeInfo{}, 0, 0, RevisedStats{}, false
		}
		limit := probeIterationLimit
		if remaining < limit {
			limit = remaining
		}
		if native != nil {
			result, reductions, ok := native.probe(ctx, bounds, probeOpts, limit)
			if !ok {
				native = nil
			} else {
				info := reliabilityProbeInfo{
					estimate: result.Estimate, validEstimate: result.EstimateValid,
					objective: result.Bound, validBound: result.BoundValid,
					infeasible: result.InfeasibleCertified,
				}
				remaining -= result.Iterations
				choice.nativeProbes++
				if result.IterationLimit {
					choice.iterationLimits++
					if result.BoundValid {
						choice.validLimitBounds++
					}
				}
				if result.InfeasibleUncertified {
					choice.uncertInfeasible++
				}
				return info, result.Iterations, reductions, result.Stats, true
			}
		}
		probeOpts.MaxIterations = limit
		res, resp, err := solveNodeOrFullWithStructure(ctx, cache, nodeModel, nodeExtras, nodeStructure, bounds, probeOpts, warm)
		if err != nil {
			return reliabilityProbeInfo{}, 0, 0, RevisedStats{}, false
		}
		remaining -= res.Iterations
		choice.fallbackProbes++
		info := reliabilityProbeInfo{}
		switch res.Status {
		case Optimal:
			info.estimate = res.Objective
			info.validEstimate = true
			info.objective = res.Objective
			info.validBound = true
		case Infeasible:
			// The ordinary solve path never exposes an uncertified warm-dual
			// infeasibility: it either returns the aggregate certificate or
			// completes the cold Phase-I reference internally.
			info.infeasible = true
		}
		return info, res.Iterations, resp.propagationReductions, resp.revised, true
	}
	probed := 0
	bestProbedScore := -math.MaxFloat64
	lookahead := 0
	for _, c := range cands {
		if probed >= probeLimit || remaining <= 0 || lookahead >= reliabilityMaxLookahead {
			if lookahead >= reliabilityMaxLookahead {
				choice.lookaheadStops++
			}
			break
		}
		if pcosts.downCount[c.idx] >= float64(reliable) && pcosts.upCount[c.idx] >= float64(reliable) {
			continue
		}
		if ctx.Err() != nil {
			break
		}
		j := c.idx
		v := values[j]
		floor := math.Floor(v)
		ceil := math.Ceil(v)

		downBounds := appendBounds(baseBounds, boundConstraint{varIndex: j, sense: LessOrEqual, rhs: floor})
		if downInfo, iters, reductions, stats, solved := runProbe(downBounds); solved {
			choice.lpSolves++
			choice.iterations += iters
			choice.revised.Add(stats)
			choice.domainReductions += reductions
			probeInferenceDown[j] = reductions
			if downInfo.infeasible {
				downScores[j] = math.Inf(1)
				probeResultsDown[j] = downInfo
				choice.infeasibleSides++
			} else if downInfo.validEstimate {
				deg := math.Max(0, pseudocostDegradation(work.direction, baseObj, downInfo.estimate))
				downScores[j] = deg
				probeResultsDown[j] = downInfo
				if downInfo.validBound && deg > 0 {
					pcosts.observeDown(j, deg, v-floor)
				}
			}
		}
		upBounds := appendBounds(baseBounds, boundConstraint{varIndex: j, sense: GreaterOrEqual, rhs: ceil})
		if upInfo, iters, reductions, stats, solved := runProbe(upBounds); solved {
			choice.lpSolves++
			choice.iterations += iters
			choice.revised.Add(stats)
			choice.domainReductions += reductions
			probeInferenceUp[j] = reductions
			if upInfo.infeasible {
				upScores[j] = math.Inf(1)
				probeResultsUp[j] = upInfo
				choice.infeasibleSides++
			} else if upInfo.validEstimate {
				deg := math.Max(0, pseudocostDegradation(work.direction, baseObj, upInfo.estimate))
				upScores[j] = deg
				probeResultsUp[j] = upInfo
				if upInfo.validBound && deg > 0 {
					pcosts.observeUp(j, deg, ceil-v)
				}
			}
		}
		probedCandidates[j] = true
		probed++
		candidateScore := strongBranchScore(downScores[j], upScores[j])
		if candidateScore > bestProbedScore {
			bestProbedScore = candidateScore
			lookahead = 0
		} else {
			lookahead++
		}
	}
	choice.candidates = probed

	best := -1
	bestScore := -math.MaxFloat64
	for _, c := range cands {
		downInference := inferenceAt(inferenceDown, c.idx)
		upInference := inferenceAt(inferenceUp, c.idx)
		if opts.ReliabilityProbeInference {
			downInference += probeInferenceDown[c.idx]
			upInference += probeInferenceUp[c.idx]
		}
		s := inferenceAdjustedScore(strongBranchScore(downScores[c.idx], upScores[c.idx]), downInference, upInference)
		if s > bestScore {
			bestScore = s
			best = c.idx
		}
	}
	if best < 0 {
		return choice
	}
	// Explore the branch with less LP degradation first; if one side was
	// proved infeasible, this sends DFS directly into the surviving side.
	choice.varIndex = best
	choice.ok = true
	choice.preferDown = downScores[best] <= upScores[best]
	choice.down = probeResultsDown[best]
	choice.up = probeResultsUp[best]
	if probedCandidates[best] {
		choice.selectedProbed = 1
	}
	choice.discardedProbes = len(probedCandidates) - choice.selectedProbed
	return choice
}

func inferenceAt(counts []int, v int) int {
	if v < 0 || v >= len(counts) {
		return 0
	}
	return counts[v]
}

// inferenceAdjustedScore gives a bounded, logarithmic preference to branch
// variables whose two sides trigger more domain deductions. LP degradation
// remains the primary signal; inference cannot turn a zero/weak pseudocost
// into an arbitrarily dominant score.
func inferenceAdjustedScore(base float64, down, up int) float64 {
	if down <= 0 && up <= 0 {
		return base
	}
	balanced := math.Min(float64(down), float64(up))
	total := float64(down + up)
	return base * (1 + 0.05*math.Log1p(total) + 0.10*math.Log1p(balanced))
}

func strongBranchScore(down, up float64) float64 {
	if math.IsInf(down, 1) && math.IsInf(up, 1) {
		return math.MaxFloat64
	}
	if math.IsInf(down, 1) {
		return 1e100 + math.Min(up, 1e99)
	}
	if math.IsInf(up, 1) {
		return 1e100 + math.Min(down, 1e99)
	}
	const eps = 1e-6
	return math.Max(eps, down) * math.Max(eps, up)
}

// pseudocostDegradation returns the non-negative objective movement from a
// parent LP solution to a child LP solution, normalized to "more positive
// = worse for the search direction". For Maximize this is parent − child;
// for Minimize it is child − parent. The caller filters out negative
// values (which would indicate numerical drift or a re-optimization that
// found a better solution because of a heuristic, not the branch).
func pseudocostDegradation(direction Direction, parentObj, childObj float64) float64 {
	if direction == Maximize {
		return parentObj - childObj
	}
	return childObj - parentObj
}
