package lp

import (
	"context"
	"math"
)

// StrongBranchResult is one side of an LP strong-branch call. Estimate is
// the current LP objective even when the iteration limit was reached and is
// intended only for branching scores. Bound is safe for pruning only when
// BoundValid is true. InfeasibleCertified is an independently checked proof
// and is the only infeasibility signal callers may use to delete a child.
type StrongBranchResult struct {
	Estimate              float64
	EstimateValid         bool
	Bound                 float64
	BoundValid            bool
	InfeasibleCertified   bool
	InfeasibleUncertified bool
	IterationLimit        bool
	Iterations            int
	Stats                 Stats
}

// StrongBranchSession owns an immutable optimal parent LP state. Every Probe
// clones that exact state, applies one side's RHS/bounds, runs bounded dual
// simplex, and discards the clone. No probe can become the next probe's start
// state or trigger the ordinary cold Phase-I fallback.
type StrongBranchSession struct {
	inc         *Incremental
	parent      *Solver
	ext         *RowSet
	phaseIICost []float64
}

// NewStrongBranchSession returns nil unless parent is a clean, dual-feasible
// optimum on this Incremental and exact RowSet identity. Keeping this gate
// narrow makes every probe a pure bound/RHS reoptimization from one certified
// parent basis; callers can fall back to the ordinary solve path otherwise.
func (inc *Incremental) NewStrongBranchSession(parent *Solver, ext *RowSet) *StrongBranchSession {
	if inc == nil || parent == nil || parent.form == nil {
		debugf("strongbranch: unavailable: missing incremental/parent/form")
		return nil
	}
	if parent.form.ext != ext || parent.n != inc.n || parent.m != inc.m+rowSetLen(ext) {
		debugf("strongbranch: unavailable: shape mismatch parent=%d/%d ext=%p request=%d/%d ext=%p",
			parent.n, parent.m, parent.form.ext, inc.n, inc.m+rowSetLen(ext), ext)
		return nil
	}
	if _, ok := parent.checkArtificialResidual(); !ok {
		debugf("strongbranch: unavailable: parent artificial residual")
		return nil
	}
	cost := makePhaseIICost(parent)
	if !parent.dualFeasibleAt(cost) {
		gap, finite := parent.dualInfeasGap(cost)
		debugf("strongbranch: unavailable: parent dual infeasible gap=%g finite=%v", gap, finite)
		return nil
	}
	return &StrongBranchSession{inc: inc, parent: parent, ext: ext, phaseIICost: cost}
}

func rowSetLen(ext *RowSet) int {
	if ext == nil {
		return 0
	}
	return len(ext.Senses)
}

// Probe evaluates one tightened child LP. b is in the Incremental's external
// row orientation, upper is in structural-column space, and iterLimit is the
// per-side strong-branch pivot limit.
func (sb *StrongBranchSession) Probe(ctx context.Context, opts Options, b, upper []float64, iterLimit int) StrongBranchResult {
	if sb == nil || iterLimit <= 0 || len(b) != sb.parent.m {
		return StrongBranchResult{}
	}
	for j := 0; j < sb.parent.n; j++ {
		u := math.Inf(1)
		if j < len(upper) {
			u = upper[j]
		}
		// Strong branching is a child-domain tightening. A relaxation would
		// invalidate the parent's dual-feasibility lineage.
		if u > sb.parent.upper[j]+opts.FeasibilityTol {
			return StrongBranchResult{}
		}
	}

	bNorm := make([]float64, len(b))
	sb.inc.orientRHS(b, bNorm, sb.parent.form.flip)
	s := sb.parent.cloneForStrongBranch(ctx, opts, bNorm, upper)
	status := s.runDualSimplexLimited(sb.phaseIICost, iterLimit, 0)

	result := StrongBranchResult{
		Estimate:      s.maxObjectiveValue(),
		EstimateValid: true,
		Iterations:    s.iterations,
		Stats:         s.stats,
	}
	if math.IsNaN(result.Estimate) || math.IsInf(result.Estimate, 0) {
		result.EstimateValid = false
	}

	switch status {
	case Optimal:
		if _, ok := s.checkArtificialResidual(); ok && s.dualFeasibleAt(sb.phaseIICost) {
			result.Bound = result.Estimate
			result.BoundValid = result.EstimateValid
		}
	case Infeasible:
		result.InfeasibleCertified = s.dualInfeasibilityCertified
		result.InfeasibleUncertified = !result.InfeasibleCertified
	case IterationLimit:
		result.IterationLimit = true
		// SCIP keeps the limit objective as an estimate, but promotes it to a
		// child bound only when the LP interface still certifies dual
		// feasibility. The explicit pass is q-opt's equivalent LPI query.
		if s.dualFeasibleAt(sb.phaseIICost) {
			result.Bound = result.Estimate
			result.BoundValid = result.EstimateValid
		}
	}
	return result
}

func (s *Solver) cloneForStrongBranch(ctx context.Context, opts Options, bNorm, upper []float64) *Solver {
	c := *s
	c.ctx = ctx
	c.opts = opts
	c.start = opts.Start
	c.bNorm = append([]float64(nil), bNorm...)
	c.upper = append([]float64(nil), s.upper...)
	c.basicCols = append([]int(nil), s.basicCols...)
	c.status = append([]ColStatus(nil), s.status...)
	c.xBasic = append([]float64(nil), s.xBasic...)
	c.etaUpdates = append([]etaUpdate(nil), s.etaUpdates...)
	c.iterations = 0
	c.stats = Stats{LPSolves: 1, WarmReopts: 1}
	c.dualInfeasibilityCertified = false
	c.farkasProof = nil
	c.onDenseRescueOutcome = nil
	c.initScratch()
	c.setStructuralUppers(upper)
	c.clampNonBasicStatuses()
	c.recomputeBasicValues()
	return &c
}

func (s *Solver) maxObjectiveValue() float64 {
	obj := 0.0
	for j := 0; j < s.n; j++ {
		obj += s.cost[j] * s.XValue(j)
	}
	return obj
}
