package qopt

import (
	"math"
	"os"

	"github.com/qizhanchan/qopt/lp"
)

// dualProofVerify (QOPT_DUALPROOF_VERIFY=1) turns every pool prune into a
// shadow decision: the node LP still runs and a prune whose LP does not come
// back infeasible is counted (and reported) instead of silently cutting a
// feasible subtree. Diagnostic harness only — it removes the prune's entire
// performance benefit.
var dualProofVerify = os.Getenv("QOPT_DUALPROOF_VERIFY") != ""

// Dual-proof conflict pool: SCIP's dual-ray infeasibility proofs
// (conflict_dualproofanalysis.c) adapted to the revised backend's certified
// aggregate Farkas exits. Each pooled proof is a model-space inequality
//
//	sum_k coefs[k] * x[inds[k]] >= rhs
//
// satisfied by every integer-feasible point of the work model (it is a
// nonnegative aggregation of the model's rows plus globally valid cut rows,
// with slack ranges and negligible terms folded through GLOBAL bounds), while
// the node that produced it provably violated it through its local bounds.
// Proofs are used exactly the way SCIP uses them by default: as propagating
// side constraints only — they are never added to any LP. A node whose
// post-propagation box makes the proof's maximum activity fall short of rhs
// is pruned as certified infeasible before its LP is even prepared, mirroring
// SCIP's cons_linear propagation cutoff on stored dualray constraints.
//
// Capacity and eviction follow SCIP's conflictstore defaults for dual-ray
// proofs: at most dualProofPoolSize entries (CONFLICTSTORE_DUALRAYSIZE = 100),
// evicting the entry whose last successful prune is oldest. The density
// filter follows conflict/maxvarsfac = 0.15: a proof touching more than 15%
// of the model's variables is not worth carrying (its activity interval
// rarely tightens enough to fire).

const (
	dualProofPoolSize   = 100
	dualProofMaxVarsFac = 0.15
	// dualProofMinMaxVars is the density-filter floor: proofs at or below
	// this many nonzeros are always kept regardless of maxvarsfac (SCIP's
	// conflict/minmaxvars intent — on small models 0.15*nvars rounds to
	// nothing and would reject every proof).
	dualProofMinMaxVars = 10
)

// DualProofStats reports dual-proof conflict pool activity for one solve.
type DualProofStats struct {
	Pooled           int // proofs accepted into the pool
	RejectedDense    int // proofs dropped by the maxvarsfac density filter
	RejectedInfinite int // proofs dropped because folding needed an infinite bound
	Evicted          int // pooled proofs displaced by newer ones
	Prunes           int // nodes pruned as certified-infeasible by a pooled proof
	// VerifyContradicted counts shadow prunes whose LP did NOT come back
	// infeasible under QOPT_DUALPROOF_VERIFY — any nonzero value is a
	// soundness bug in the proof translation or evaluation.
	VerifyContradicted int
}

type dualProof struct {
	inds    []int32
	coefs   []float64
	rhs     float64
	lastHit int64
	hits    int
}

type dualProofPool struct {
	feasTol float64
	// globalLower/globalUpper are the work model's root bounds, used to fold
	// negligible coefficients soundly and to discard globally redundant proofs.
	globalLower []float64
	globalUpper []float64
	proofs      []*dualProof
	evals       int64
	scratch     []float64 // dense model-space accumulator, len(model.vars)
	stats       DualProofStats
}

func newDualProofPool(m *Model, feasTol float64) *dualProofPool {
	lower, upper := effectiveBounds(m, nil)
	return &dualProofPool{
		feasTol:     feasTol,
		globalLower: lower,
		globalUpper: upper,
		scratch:     make([]float64, len(m.vars)),
	}
}

// addFromLP translates a certified lp.FarkasProof into model space and pools
// it. The lp proof is over the cache's scaled, lower-shifted y-columns
// (y_j = (x_var - nodeLower[var]) / colScale[j]), so the model-space form is
//
//	coef_var = sum_j proofCoef_j / colScale[j]   (var = colToVar[j])
//	rhs      = proofRHS + sum_var coef_var * nodeLower[var]
//
// which no longer references the node: the shift constants cancel against
// the shifted RHS the certificate was computed with. nodeLower must be the
// exact model-space lower bounds the node LP was prepared with.
func (p *dualProofPool) addFromLP(c *canonCache, fp *lp.FarkasProof, nodeLower []float64) {
	if p == nil || fp == nil || len(fp.Inds) == 0 {
		return
	}
	dense := p.scratch
	touched := make([]int32, 0, len(fp.Inds))
	maxAbs := 0.0
	for k, j := range fp.Inds {
		v := c.colToVar[j]
		if dense[v] == 0 {
			touched = append(touched, int32(v))
		}
		dense[v] += fp.Coefs[k] / c.colScale[j]
	}
	rhs := fp.RHS
	for _, v := range touched {
		rhs += dense[v] * nodeLower[v]
		if a := math.Abs(dense[v]); a > maxAbs {
			maxAbs = a
		}
	}
	clear := func() {
		for _, v := range touched {
			dense[v] = 0
		}
	}
	if maxAbs == 0 || math.IsNaN(rhs) || math.IsInf(rhs, 0) {
		clear()
		return
	}

	// Fold numerically negligible coefficients into rhs through the GLOBAL
	// bounds (dropping term c*x from the left side weakens rhs by that
	// term's global maximum, keeping the inequality valid at every node).
	// When the fold would need an infinite bound, one escape mirrors the
	// certificate exactly: a coefficient at or below the ratio test's
	// candidacy tolerance (feasTol in the scaled column space,
	// |coef|*colScale here) on an unbounded variable is treated as exactly
	// zero — the judgment certifyDualLeavingRow already took for this ray
	// (SCIP's SCIPsetIsDualfeasZero posture). Without it, a branching upper
	// on a globally unbounded variable at the certifying node leaks a noise
	// coefficient into the proof and the infinite fold rejects it wholesale.
	dropTol := 1e-9 * maxAbs
	inds := make([]int32, 0, len(touched))
	for _, v := range touched {
		coef := dense[v]
		if coef == 0 {
			continue
		}
		if math.Abs(coef) <= dropTol {
			worst := coef * p.globalUpper[v]
			if coef < 0 {
				worst = coef * p.globalLower[v]
			}
			if math.IsInf(worst, 0) || math.IsNaN(worst) {
				if col := c.varToCol[v]; col >= 0 && math.Abs(coef)*c.colScale[col] <= p.feasTol {
					continue // certificate-tolerance zero on an unbounded column
				}
				p.stats.RejectedInfinite++
				clear()
				return
			}
			rhs -= worst
			continue
		}
		inds = append(inds, v)
	}
	if len(inds) == 0 {
		clear()
		return
	}
	if len(inds) > dualProofMinMaxVars && float64(len(inds)) > dualProofMaxVarsFac*float64(len(dense)) {
		p.stats.RejectedDense++
		clear()
		return
	}

	proof := &dualProof{
		inds:    inds,
		coefs:   make([]float64, len(inds)),
		rhs:     rhs,
		lastHit: p.evals,
	}
	for k, v := range inds {
		proof.coefs[k] = dense[v]
	}
	clear()

	if len(p.proofs) < dualProofPoolSize {
		p.proofs = append(p.proofs, proof)
		p.stats.Pooled++
		return
	}
	// Evict the proof whose last prune is oldest (never-hit entries keep
	// their insertion stamp, so stale never-useful proofs cycle out first).
	worst := 0
	for i := 1; i < len(p.proofs); i++ {
		if p.proofs[i].lastHit < p.proofs[worst].lastHit ||
			(p.proofs[i].lastHit == p.proofs[worst].lastHit && p.proofs[i].hits < p.proofs[worst].hits) {
			worst = i
		}
	}
	p.proofs[worst] = proof
	p.stats.Pooled++
	p.stats.Evicted++
}

// pruneNode reports whether any pooled proof certifies the node box
// [lower, upper] infeasible: the box's maximum activity provably stays below
// the proof's rhs. The margin mirrors certifyDualLeavingRow's acceptance
// margin so a pooled proof is never trusted more aggressively than the
// certificate that created it.
func (p *dualProofPool) pruneNode(lower, upper []float64) bool {
	if p == nil || len(p.proofs) == 0 {
		return false
	}
	p.evals++
	for _, proof := range p.proofs {
		maxact := 0.0
		scale := math.Abs(proof.rhs)
		violated := true
		for k, v := range proof.inds {
			coef := proof.coefs[k]
			bound := upper[v]
			if coef < 0 {
				bound = lower[v]
			}
			if math.IsInf(bound, 0) || math.IsNaN(bound) {
				violated = false
				break
			}
			term := coef * bound
			maxact += term
			scale += math.Abs(term)
		}
		if !violated || math.IsNaN(maxact) {
			continue
		}
		margin := 100 * p.feasTol * (1 + scale)
		if maxact+margin < proof.rhs {
			proof.hits++
			proof.lastHit = p.evals
			p.stats.Prunes++
			return true
		}
	}
	return false
}
