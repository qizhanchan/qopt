package qopt

import (
	"math"
	"sort"
	"strconv"
	"strings"

	"github.com/qizhanchan/qopt/lp"
)

const (
	defaultLPConflictMaxLiterals = 12
	lpConflictProbeIterations    = 2000
	lpConflictPoolLimit          = 10000
)

// boundLiteral is one integral branch decision used as an assumption in an
// LP infeasibility proof: x <= k or x >= k for integral k.
type boundLiteral struct {
	varIndex int
	sense    Sense
	rhs      float64
}

type binaryConflict struct {
	literals []boundLiteral
}

type lpConflictPool struct {
	conflicts []binaryConflict
	seen      map[string]struct{}
}

func appendBoundLiteral(src []boundLiteral, literal boundLiteral) []boundLiteral {
	dst := make([]boundLiteral, len(src), len(src)+1)
	copy(dst, src)
	return append(dst, literal)
}

func literalBounds(literals []boundLiteral) []boundConstraint {
	bounds := make([]boundConstraint, 0, len(literals))
	for _, literal := range literals {
		bounds = append(bounds, boundConstraint{varIndex: literal.varIndex, sense: literal.sense, rhs: literal.rhs})
	}
	return bounds
}

func conflictKey(literals []boundLiteral) string {
	canonical := append([]boundLiteral(nil), literals...)
	sort.Slice(canonical, func(i, j int) bool {
		if canonical[i].varIndex != canonical[j].varIndex {
			return canonical[i].varIndex < canonical[j].varIndex
		}
		if canonical[i].sense != canonical[j].sense {
			return canonical[i].sense < canonical[j].sense
		}
		return canonical[i].rhs < canonical[j].rhs
	})
	var b strings.Builder
	for _, literal := range canonical {
		b.WriteString(strconv.Itoa(literal.varIndex))
		b.WriteByte(':')
		b.WriteString(strconv.Itoa(int(literal.sense)))
		b.WriteByte(':')
		b.WriteString(strconv.FormatFloat(literal.rhs, 'g', -1, 64))
		b.WriteByte(';')
	}
	return b.String()
}

func conflictSubset(a, b []boundLiteral) bool {
	if len(a) > len(b) {
		return false
	}
	set := make(map[boundLiteral]struct{}, len(b))
	for _, literal := range b {
		set[literal] = struct{}{}
	}
	for _, literal := range a {
		if _, ok := set[literal]; !ok {
			return false
		}
	}
	return true
}

func normalizeConflict(literals []boundLiteral) ([]boundLiteral, bool) {
	seen := make(map[boundLiteral]struct{}, len(literals))
	out := make([]boundLiteral, 0, len(literals))
	for _, literal := range literals {
		if literal.sense != LessOrEqual && literal.sense != GreaterOrEqual {
			return nil, false
		}
		if _, exists := seen[literal]; exists {
			continue
		}
		seen[literal] = struct{}{}
		out = append(out, literal)
	}
	return out, len(out) > 0
}

func (p *lpConflictPool) add(literals []boundLiteral) bool {
	literals, ok := normalizeConflict(literals)
	if !ok || len(p.conflicts) >= lpConflictPoolLimit {
		return false
	}
	key := conflictKey(literals)
	if p.seen != nil {
		if _, exists := p.seen[key]; exists {
			return false
		}
	}
	for _, conflict := range p.conflicts {
		if conflictSubset(conflict.literals, literals) {
			return false
		}
	}
	kept := p.conflicts[:0]
	for _, conflict := range p.conflicts {
		if conflictSubset(literals, conflict.literals) {
			delete(p.seen, conflictKey(conflict.literals))
			continue
		}
		kept = append(kept, conflict)
	}
	p.conflicts = append(kept, binaryConflict{literals: append([]boundLiteral(nil), literals...)})
	if p.seen == nil {
		p.seen = make(map[string]struct{})
	}
	p.seen[key] = struct{}{}
	return true
}

// applyLPConflicts performs no-good propagation to a fixed point. A conflict
// says that not all of its literals may hold simultaneously. If all but one
// hold, the last literal is forced false; if all hold, the node is infeasible.
func (sc *searchContext) applyLPConflicts(bounds []boundConstraint) ([]boundConstraint, bool) {
	if len(sc.lpConflicts.conflicts) == 0 {
		return bounds, false
	}
	tol := sc.opts.IntegralityTol
	if tol <= 0 {
		tol = 1e-8
	}
	for {
		lower, upper := effectiveBounds(sc.work, bounds)
		changed := false
		for _, conflict := range sc.lpConflicts.conflicts {
			unknown := 0
			var last boundLiteral
			satisfied := false
			for _, literal := range conflict.literals {
				state := integralLiteralState(literal, lower[literal.varIndex], upper[literal.varIndex], tol)
				if state == literalUnknown {
					unknown++
					last = literal
					continue
				}
				if state == literalFalse {
					satisfied = true
					break
				}
			}
			if satisfied {
				continue
			}
			switch unknown {
			case 0:
				sc.stats.Branching.LPConflictPrunes++
				return bounds, true
			case 1:
				bounds = appendBounds(bounds, negateIntegralLiteral(last))
				sc.stats.Branching.LPConflictPropagations++
				changed = true
			}
			if changed {
				break
			}
		}
		if !changed {
			return bounds, false
		}
	}
}

type literalState uint8

const (
	literalUnknown literalState = iota
	literalTrue
	literalFalse
)

func integralLiteralState(literal boundLiteral, lower, upper, tol float64) literalState {
	if literal.sense == LessOrEqual {
		if upper <= literal.rhs+tol {
			return literalTrue
		}
		if lower >= literal.rhs+1-tol {
			return literalFalse
		}
		return literalUnknown
	}
	if lower >= literal.rhs-tol {
		return literalTrue
	}
	if upper <= literal.rhs-1+tol {
		return literalFalse
	}
	return literalUnknown
}

func negateIntegralLiteral(literal boundLiteral) boundConstraint {
	if literal.sense == LessOrEqual {
		return boundConstraint{varIndex: literal.varIndex, sense: GreaterOrEqual, rhs: literal.rhs + 1}
	}
	return boundConstraint{varIndex: literal.varIndex, sense: LessOrEqual, rhs: literal.rhs - 1}
}

func nodeBoundsAreOnlyBranchLiterals(node mipNode) bool {
	if len(node.bounds) != len(node.branchLiterals) {
		return false
	}
	want := make(map[boundLiteral]int, len(node.branchLiterals))
	for _, literal := range node.branchLiterals {
		want[literal]++
	}
	for _, bound := range node.bounds {
		if bound.sense != LessOrEqual && bound.sense != GreaterOrEqual {
			return false
		}
		literal := boundLiteral{varIndex: bound.varIndex, sense: bound.sense, rhs: bound.rhs}
		if want[literal] == 0 {
			return false
		}
		want[literal]--
	}
	return true
}

func (sc *searchContext) solveLPConflictTrial(literals []boundLiteral) Status {
	probeOpts := sc.opts
	probeOpts.TimeLimit = 0
	probeOpts.MaxCutRounds = 0
	probeOpts.MaxCutDepth = 0
	probeOpts.MaxIterations = lpConflictProbeIterations
	probeOpts.DisablePrimalHeuristics = true
	res, resp, err := solveNodeOrFull(sc.ctx, sc.workCache, sc.work, nil, literalBounds(literals), probeOpts, nil)
	if err != nil {
		return IterationLimit
	}
	sc.iterations += res.Iterations
	sc.stats.addRevised(resp.revised)
	sc.stats.Branching.LPConflictLPSolves++
	sc.stats.Branching.LPConflictIterations += res.Iterations
	return res.Status
}

// learnLPConflict extracts a small no-good from a cold-verified infeasible
// node. Only integral tree decisions are candidates. If the node also contains
// inferred bounds or local rows, the decision-only assumptions are verified
// again against the global work model before a conflict is accepted.
func (sc *searchContext) lpConflictCertificateProves(cert *lp.InfeasibilityCertificate, literals []boundLiteral) bool {
	if cert == nil || sc.workCache == nil || sc.workCache.model != sc.work {
		return false
	}
	sc.stats.Branching.LPConflictCertificateTests++
	bound, ok := sc.workCache.lpConflictCertificateBound(cert, literalBounds(literals))
	margin := math.Max(1e-9, cert.Tolerance*0.01)
	if ok && bound > cert.Tolerance+margin {
		sc.stats.Branching.LPConflictCertificateProofs++
		return true
	}
	return false
}

func (sc *searchContext) learnLPConflict(node mipNode, cert *lp.InfeasibilityCertificate) {
	if !sc.opts.UseLPConflictLearning || len(node.branchLiterals) == 0 || sc.ctx.Err() != nil {
		return
	}
	literals, ok := normalizeConflict(node.branchLiterals)
	if !ok {
		return
	}
	maxLiterals := sc.opts.LPConflictMaxLiterals
	if maxLiterals <= 0 {
		maxLiterals = defaultLPConflictMaxLiterals
	}
	if len(literals) > maxLiterals {
		return
	}
	maxProbes := sc.opts.LPConflictMinimizeProbes
	if maxProbes < 0 {
		maxProbes = 0
	}
	probes := 0
	if !nodeBoundsAreOnlyBranchLiterals(node) {
		if !sc.lpConflictCertificateProves(cert, literals) {
			if probes >= maxProbes {
				return
			}
			probes++
			if sc.solveLPConflictTrial(literals) != Infeasible {
				return
			}
		}
	}
	for i := 0; i < len(literals) && len(literals) > 1; {
		trial := make([]boundLiteral, 0, len(literals)-1)
		trial = append(trial, literals[:i]...)
		trial = append(trial, literals[i+1:]...)
		proved := sc.lpConflictCertificateProves(cert, trial)
		if !proved && probes < maxProbes {
			probes++
			proved = sc.solveLPConflictTrial(trial) == Infeasible
		}
		if proved {
			literals = trial
			continue
		}
		i++
	}
	if sc.lpConflicts.add(literals) {
		sc.stats.Branching.LPConflictsLearned++
		sc.stats.Branching.LPConflictLiterals += len(literals)
		debugf("lp conflict: learned %d-literal no-good from depth %d", len(literals), node.depth)
	}
}
