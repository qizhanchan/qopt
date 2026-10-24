// Package lp is q-opt's LP engine: a bounded-variable revised simplex with a
// sparse LU basis factorization (dense-LU stability fallback), product-form
// eta updates between refactors, and dual-simplex re-optimization for
// warm starts.
//
// The package boundary is deliberately sparse and self-contained: a Problem
// carries the constraint matrix as sparse columns, the caller owns any
// presolve/scaling, and the returned Solution exposes the final Solver for
// basis-level consumers (Gomory cut generation, reduced-cost fixing) through
// read-only accessors instead of internal fields.
//
// Setting QOPT_DEBUG=1 in the environment enables stderr diagnostics.
package lp

import (
	"fmt"
	"math"
	"os"
	"strconv"
	"time"
)

// Sense is the row sense of a linear constraint.
type Sense int

const (
	LessOrEqual Sense = iota
	GreaterOrEqual
	Equal
)

// Status is the outcome of a solve. NodeLimit and GapLimit are never produced
// by this package; they are defined here so the MIP layer can share one status
// enum.
type Status int

const (
	Optimal Status = iota
	Infeasible
	Unbounded
	IterationLimit
	NodeLimit
	TimeLimit
	GapLimit
)

func (s Status) String() string {
	switch s {
	case Optimal:
		return "optimal"
	case Infeasible:
		return "infeasible"
	case Unbounded:
		return "unbounded"
	case IterationLimit:
		return "iteration_limit"
	case NodeLimit:
		return "node_limit"
	case TimeLimit:
		return "time_limit"
	case GapLimit:
		return "gap_limit"
	default:
		return "unknown"
	}
}

// Problem is a scaled LP in the caller's canonical form:
//
//	max c·x  s.t.  A x (sense) b,  0 ≤ x ≤ upper
//
// The objective is in MAXIMIZE sense (minimization callers negate upstream).
// Cols[j] is column j of A with entries in strictly increasing row order and
// no explicit zeros; len(Cols) == len(C) == number of structural columns, and
// len(B) == len(Senses) == number of rows. Upper holds per-column upper
// bounds (+Inf allowed); a short or nil Upper means +Inf for the missing
// columns. Rows with negative B are sign-normalized internally without
// mutating the caller's slices, so a Problem — including its columns — may be
// shared across many Solve calls (the cross-node reuse the MIP layer's
// canonicalize cache depends on).
type Problem struct {
	C           []float64
	Cols        []SparseVector
	Senses      []Sense
	B           []float64
	Upper       []float64
	RangeWidths []float64 // finite width marks L <= row <= B with L = B-width
}

// Options are the solver knobs. Zero values select the documented defaults.
type Options struct {
	FeasibilityTol float64
	OptimalityTol  float64
	// MaxIterations caps total simplex pivots for one Solve call.
	MaxIterations int
	// TimeLimit, measured from Start, bounds wall-clock time. Zero disables
	// the check. Start must be set whenever TimeLimit is.
	TimeLimit time.Duration
	Start     time.Time
	// MaxEtaUpdates is the number of product-form eta updates kept before a
	// full refactorization. Zero = default (64); negative = refactor every
	// pivot.
	MaxEtaUpdates int
	// EtaResidualFactor scales the relative basis-residual threshold that
	// forces a refactor: residual > factor × FeasibilityTol. Zero = default
	// (100).
	EtaResidualFactor float64
	// DenseLU pins the basis factorization to the dense LU. Default (false)
	// uses the sparse LU with a dense stability fallback.
	DenseLU bool
	// PersistFactor carries the LU + eta chain inside the WarmStart handle so
	// a child solve with the same basis matrix can skip its first refactor.
	PersistFactor bool
	// CertifyInfeasible enables a conservative tableau-row certificate when
	// warm dual re-optimization finds no eligible entering column. A certified
	// row proves that variable bounds cannot repair the leaving-row violation,
	// allowing the caller to skip the usual cold Phase-I verification.
	CertifyInfeasible bool
}

// Solution is the result of a Solve call.
type Solution struct {
	Status     Status
	Objective  float64
	Values     []float64 // structural-column values; nil unless Optimal
	Iterations int
	// Warm is a basis handle for re-solving after bound changes on the same
	// problem shape. Nil unless the solve reached Optimal.
	Warm *WarmStart
	// Solver is the final solver state, exposed for basis-level consumers
	// (cut generation, reduced-cost fixing). Nil unless the solve reached
	// Optimal. Read-only: use the exported accessor methods.
	Solver *Solver
	// Infeasibility is a Phase-I dual certificate when a cold solve proves
	// infeasibility. It is intentionally narrower than Solver: callers can
	// re-evaluate the same certificate after relaxing variable bounds without
	// mutating or depending on the final simplex basis.
	Infeasibility *InfeasibilityCertificate
	// InfeasibilityCertified is true only when the warm dual-simplex path
	// produced a tableau-row bound certificate. It is independent of the
	// Phase-I certificate above because the two proofs have different shapes.
	InfeasibilityCertified bool
	// FarkasProof is the aggregated proof row behind a certified infeasible
	// exit, exported for conflict reuse. Nil unless InfeasibilityCertified.
	FarkasProof *FarkasProof
	Stats       Stats
}

// FarkasProof is the structural-column form of the aggregate Farkas proof
// validated by certifyDualLeavingRow. Every point that satisfies the solved
// body's rows and slack ranges — regardless of the structural column bounds
// the certifying solve happened to use — satisfies
//
//	sum_k Coefs[k] * y[Inds[k]] >= RHS,
//
// while the certifying solve's own box provably violates it. Slack columns
// are folded into RHS through their global ranges (they do not change across
// bound-only re-solves), so a MIP caller can translate the inequality back to
// model space and reuse it as a valid conflict at any other node — SCIP's
// dual-ray proof shape (conflict_dualproofanalysis.c). The only tolerance
// judgment taken is the same one the acceptance test takes: an aggregated
// coefficient at or below the ratio test's candidacy tolerance on an
// unbounded column is treated as exactly zero.
type FarkasProof struct {
	Inds  []int32
	Coefs []float64
	RHS   float64
}

// InfeasibilityCertificate gives a lower bound on the Phase-I artificial
// objective for the standard-form row orientation recorded in Flip:
//
//	dual·b + AuxContribution + sum_j min(0, ReducedCosts[j]*upper[j]).
//
// A value above Tolerance proves the relaxed LP infeasible. ReducedCosts and
// upper refer only to structural columns; all unchanged auxiliary-column
// terms are folded into AuxContribution.
type InfeasibilityCertificate struct {
	Dual            []float64
	ReducedCosts    []float64
	AuxContribution float64
	Flip            []bool
	Tolerance       float64
}

// Stats aggregates numerical/diagnostic counters across solves.
type Stats struct {
	LPSolves              int
	Refactors             int
	EtaUpdates            int
	EtaRefactorFallbacks  int
	MaxEtaDepth           int
	MaxBasisResidual      float64
	MaxBasisRelResidual   float64
	MinPivot              float64
	MaxGrowth             float64
	PhaseIResidualRetries int
	// FactorReuses counts warm-start installs that reused the parent's
	// factorization instead of refactoring (Options.PersistFactor). High
	// reuse relative to Refactors is the throughput win.
	FactorReuses int
	// SparseDenseFallbacks counts refactors where the sparse LU was singular
	// or numerically unstable (basis residual above tolerance) and the dense
	// LU was used for that factorization instead.
	SparseDenseFallbacks int

	// Per-phase telemetry. Iteration counts split simplex pivots by which
	// loop performed them; the nanosecond sums are wall-clock per phase, so
	// (DenseFactors, DenseFactorNanos) exposes the dense-fallback burn that
	// dominated glass4/ic97_potential profiles without needing pprof.
	PhaseIIterations  int
	PhaseIIIterations int
	DualIterations    int
	SparseFactors     int
	DenseFactors      int
	// Warm-start event counters — NOT disjoint per-solve categories.
	// ColdStarts counts every cold Phase I/II run, including the fallback
	// after a failed warm attempt. WarmReopts counts handles that installed
	// and started a dual reoptimize (one that later fails ALSO counts in
	// WarmFailures and ColdStarts). WarmFlipRemaps counts handles translated
	// across a row-sign flip drift before install. WarmFailures counts
	// handles rejected by classification, failed installs, and abandoned
	// reopts.
	ColdStarts                 int
	WarmReopts                 int
	WarmFlipRemaps             int
	WarmFailures               int
	WarmRejected               int
	WarmInstallFailures        int
	WarmDualInfeasible         int
	WarmInfeasibleCandidates   int
	WarmInfeasibleCertificates int
	WarmInfeasibleChecks       int
	WarmInfeasibleInconclusive int
	WarmInfeasibleContradicted int
	WarmCertRejectBasis        int
	WarmCertRejectInfinite     int
	WarmCertRejectMargin       int
	WarmCertRejectInvalid      int
	WarmDualIterationCaps      int
	WarmArtificialFailure      int
	WarmColdInfeasible         int
	// Certified-chain verification cadence (see WarmStart.certChain).
	// WarmCertAccepts counts optima trusted on the certificate alone;
	// WarmCertVerifies counts certificate-eligible optima that hit the chain
	// limit and ran the explicit dual-feasibility check instead; and
	// WarmCertVerifyFails counts the checks that FAILED — each one is a
	// dual-infeasible vertex the unlimited-chain shortcut would have accepted
	// as the node's LP optimum. Failures split into bound corrections (the
	// vertex was returned with its objective weakened by the dual-infeasiblity
	// gap — see dualInfeasGap) and resumes (an infinite-range violation forced
	// the primal re-optimization); WarmCertBoundCorrectionMax is the largest
	// single correction applied.
	WarmCertAccepts            int
	WarmCertVerifies           int
	WarmCertVerifyFails        int
	WarmCertBoundCorrections   int
	WarmCertVerifyResumes      int
	WarmCertBoundCorrectionMax float64
	DualStalls                 int
	DualNoEntering             int
	DualIterationCaps          int
	// Focus-node session counters (Options via Session; zero when the
	// stateless path is used directly). SessionReopts counts in-place
	// reoptimize attempts, SessionFallbacks the attempts that had to re-run
	// through the stateless path, SessionAdoptions the stateless solves whose
	// solver became the new live state, and SessionInfeasibleRetries the
	// refactor-and-rerun escalations after an uncertified dual infeasibility.
	SessionReopts            int
	SessionFallbacks         int
	SessionAdoptions         int
	SessionInfeasibleRetries int
	// Factor nanos accrue inside whichever phase ran the refactor, so
	// Sparse/DenseFactorNanos are subsets of the phase nanos — summing all
	// five double-counts factor time. SparseFactorNanos includes failed
	// (singular) factor attempts; SparseFactors counts successes only.
	PhaseINanos       int64
	PhaseIINanos      int64
	DualNanos         int64
	SparseFactorNanos int64
	DenseFactorNanos  int64
}

// Add merges the counters from r into s (maxima for the high-water fields,
// sums for the event counts). A zero-LPSolves r is a no-op.
func (s *Stats) Add(r Stats) {
	if s == nil || r.LPSolves == 0 {
		return
	}
	s.LPSolves += r.LPSolves
	s.Refactors += r.Refactors
	s.EtaUpdates += r.EtaUpdates
	s.EtaRefactorFallbacks += r.EtaRefactorFallbacks
	if r.MaxEtaDepth > s.MaxEtaDepth {
		s.MaxEtaDepth = r.MaxEtaDepth
	}
	if r.MaxBasisResidual > s.MaxBasisResidual {
		s.MaxBasisResidual = r.MaxBasisResidual
	}
	if r.MaxBasisRelResidual > s.MaxBasisRelResidual {
		s.MaxBasisRelResidual = r.MaxBasisRelResidual
	}
	if s.MinPivot == 0 || (r.MinPivot > 0 && r.MinPivot < s.MinPivot) {
		s.MinPivot = r.MinPivot
	}
	if r.MaxGrowth > s.MaxGrowth {
		s.MaxGrowth = r.MaxGrowth
	}
	s.PhaseIResidualRetries += r.PhaseIResidualRetries
	s.FactorReuses += r.FactorReuses
	s.SparseDenseFallbacks += r.SparseDenseFallbacks
	s.PhaseIIterations += r.PhaseIIterations
	s.PhaseIIIterations += r.PhaseIIIterations
	s.DualIterations += r.DualIterations
	s.SparseFactors += r.SparseFactors
	s.DenseFactors += r.DenseFactors
	s.PhaseINanos += r.PhaseINanos
	s.PhaseIINanos += r.PhaseIINanos
	s.DualNanos += r.DualNanos
	s.SparseFactorNanos += r.SparseFactorNanos
	s.DenseFactorNanos += r.DenseFactorNanos
	s.ColdStarts += r.ColdStarts
	s.WarmReopts += r.WarmReopts
	s.WarmFlipRemaps += r.WarmFlipRemaps
	s.WarmFailures += r.WarmFailures
	s.WarmRejected += r.WarmRejected
	s.WarmInstallFailures += r.WarmInstallFailures
	s.WarmDualInfeasible += r.WarmDualInfeasible
	s.WarmInfeasibleCandidates += r.WarmInfeasibleCandidates
	s.WarmInfeasibleCertificates += r.WarmInfeasibleCertificates
	s.WarmInfeasibleChecks += r.WarmInfeasibleChecks
	s.WarmInfeasibleInconclusive += r.WarmInfeasibleInconclusive
	s.WarmInfeasibleContradicted += r.WarmInfeasibleContradicted
	s.WarmCertRejectBasis += r.WarmCertRejectBasis
	s.WarmCertRejectInfinite += r.WarmCertRejectInfinite
	s.WarmCertRejectMargin += r.WarmCertRejectMargin
	s.WarmCertRejectInvalid += r.WarmCertRejectInvalid
	s.WarmDualIterationCaps += r.WarmDualIterationCaps
	s.WarmArtificialFailure += r.WarmArtificialFailure
	s.WarmColdInfeasible += r.WarmColdInfeasible
	s.WarmCertAccepts += r.WarmCertAccepts
	s.WarmCertVerifies += r.WarmCertVerifies
	s.WarmCertVerifyFails += r.WarmCertVerifyFails
	s.WarmCertBoundCorrections += r.WarmCertBoundCorrections
	s.WarmCertVerifyResumes += r.WarmCertVerifyResumes
	if r.WarmCertBoundCorrectionMax > s.WarmCertBoundCorrectionMax {
		s.WarmCertBoundCorrectionMax = r.WarmCertBoundCorrectionMax
	}
	s.DualStalls += r.DualStalls
	s.DualNoEntering += r.DualNoEntering
	s.DualIterationCaps += r.DualIterationCaps
	s.SessionReopts += r.SessionReopts
	s.SessionFallbacks += r.SessionFallbacks
	s.SessionAdoptions += r.SessionAdoptions
	s.SessionInfeasibleRetries += r.SessionInfeasibleRetries
}

func debugEnabled() bool { return os.Getenv("QOPT_DEBUG") != "" }

var infeasibilityCertificateShadow = os.Getenv("QOPT_CERT_SHADOW") != ""

// certChainLimit bounds how many consecutive solves may accept a warm
// reoptimize Optimal on the classifyWarm certificate alone before an explicit
// dual-feasibility check (one BTRAN + one pricing pass) is forced. The
// certificate's induction — "certified handles stay dual-feasible" — holds
// only up to the dual ratio test's per-pivot tolerance, and the drift
// compounds across a chain of certified reuses: on pk1 an unbounded chain
// accepted a dual-infeasible vertex whose overstated bound pruned the optimal
// subtree and falsely "proved" 12 where the optimum is 11. The verification
// resets the chain, so the default cadence pays roughly one pricing pass per
// certChainLimit certified accepts. QOPT_CERT_CHAIN overrides for A/B runs:
// 0 verifies every accept, a negative value restores the legacy unlimited
// shortcut.
const defaultCertChainLimit = 8

// certCorrectionRelTol caps the verify-failure bound correction relative to
// the vertex objective (see certCorrectionCap). QOPT_CERT_CORRTOL overrides
// for A/B tuning.
var certCorrectionRelTol = func() float64 {
	env := os.Getenv("QOPT_CERT_CORRTOL")
	if env == "" {
		return 1e-4
	}
	v, err := strconv.ParseFloat(env, 64)
	if err != nil {
		return 1e-4
	}
	return v
}()

var certChainLimit = func() int {
	env := os.Getenv("QOPT_CERT_CHAIN")
	if env == "" {
		return defaultCertChainLimit
	}
	v, err := strconv.Atoi(env)
	if err != nil {
		return defaultCertChainLimit
	}
	return v
}()

// Experimental opt-in gates for the Harris two-pass ratio tests
// (QOPT_HARRIS = "primal", "dual", or "both"). Measured 2026-07-10 on the
// MIPLIB sub-suite and left OFF by default: max-|pivot| tie-breaking lands
// the simplex on different degenerate vertices, and on ic97_potential (both
// variants) and glass4 (primal) the per-node rounding heuristic stops
// producing ANY incumbent — a milestone-1 acceptance violation — while the
// only win was neos-3754480-nidda under the dual variant (obj 14516→14147,
// +18% nodes). Structural-bound checks show the Harris points are NOT
// infeasible; the effect is pure vertex-selection/search-trajectory
// sensitivity. Retest only together with a rounding-friendly cleanup pass.
var harrisEnv = os.Getenv("QOPT_HARRIS")

func harrisPrimal() bool { return harrisEnv == "both" || harrisEnv == "primal" }
func harrisDual() bool   { return harrisEnv == "both" || harrisEnv == "dual" }

// etaCostFactor is the K in the cost-based refactor rule (refactor once the
// eta chain's stored nnz exceeds K × the factor's nnz). QOPT_ETAK overrides
// for A/B tuning; ≤0 disables the rule (fixed 64-eta cap only) and an
// unparseable value falls back to the default K=1.
var etaK = func() float64 {
	env := os.Getenv("QOPT_ETAK")
	if env == "" {
		return 1.0
	}
	k, err := strconv.ParseFloat(env, 64)
	if err != nil {
		return 1.0
	}
	return k
}()

func etaCostFactor() float64 { return etaK }

func debugf(format string, args ...any) {
	if debugEnabled() {
		fmt.Fprintf(os.Stderr, "[qopt] "+format+"\n", args...)
	}
}

func flipSense(s Sense) Sense {
	switch s {
	case LessOrEqual:
		return GreaterOrEqual
	case GreaterOrEqual:
		return LessOrEqual
	default:
		return Equal
	}
}

func cleanZero(v, tol float64) float64 {
	if math.Abs(v) <= tol {
		return 0
	}
	return v
}
