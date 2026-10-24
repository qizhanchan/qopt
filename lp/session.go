package lp

import (
	"context"
	"math"
	"os"
	"strconv"
	"time"
)

// sessionInstallOnly (QOPT_SESSION_INSTALL_ONLY=1) disables the no-refactor
// fast paths so every session reoptimize installs the caller's handle from a
// fresh refactor — an A/B bisection aid for isolating factor-persistence
// effects from the arena/setup savings.
var sessionInstallOnly = os.Getenv("QOPT_SESSION_INSTALL_ONLY") != ""

// sessionVerify (QOPT_SESSION_VERIFY=1) cross-checks every session-accepted
// result against a fresh stateless cold solve and panics on divergence — a
// debugging harness for locating unsound in-place acceptances.
var sessionVerify = os.Getenv("QOPT_SESSION_VERIFY") != ""

// sessionCertChainLimit is the session's own trusted-accept cadence, distinct
// from the stateless certChainLimit: 0 by default, i.e. every in-place accept
// runs the explicit dual-feasibility pass (failures still repaired by the
// capped bound correction — the resume is the past-the-cap escape only). The
// stateless K=8 window is measured-safe for the stateless trajectory, whose
// cold rebuilds at fallback points act as drift firebreaks; the session walks
// hundreds of thousands of consecutive in-place solves with no such reset,
// and under the same K=8 window a single overstated bound inside the window
// re-created pk1's false proof (claimed optimal 12 vs true 11, matched 5m
// suite, July 21 2026). QOPT_SESSION_CERT_CHAIN overrides for A/B tuning;
// negative mirrors the stateless legacy-unlimited semantics.
var sessionCertChainLimit = func() int {
	env := os.Getenv("QOPT_SESSION_CERT_CHAIN")
	if env == "" {
		return 0
	}
	v, err := strconv.Atoi(env)
	if err != nil {
		return 0
	}
	return v
}()

// Session is the focus-node LP: a persistent, already-assembled solver (basis,
// bounds, factorization, eta chain) that follows the B&B tree walk, ported
// from SCIP's SCIPlpSolveAndEval / focus-node switching. Where the stateless
// SolveRows re-runs classify + build + warm install (one refactor) for every
// node, a session solve on an unchanged problem shape rewrites the RHS and
// the structural upper bounds in place and dual-reoptimizes — no allocation
// and no warm-handle translation, and on a pure bound tightening (the
// dive/plunge pattern) no install refactor either: the live factorization and
// eta chain carry straight across the node boundary.
//
// The starting basis is chosen per node, because q-opt's traversal is not
// SCIP's plunge-dominated walk: best-estimate jumps and reliability probes
// interleave bound relaxations that make "continue from whatever was solved
// last" a poor warm start (measured: >50% in-place failures and a large
// primal-resume burn on neos17/timtab1). A solve that only tightens the
// session's current bounds continues from the current basis; anything else
// installs the caller's warm handle (the node's parent basis — the same
// start the stateless path would use) into the live solver, skipping the
// refactor when that basis is already the one in the factorization.
//
// The session never replaces the stateless path; it fronts it. Any shape
// change and any in-place attempt the session cannot finish soundly falls
// back to Incremental.SolveRows (cold when a dual attempt already ran), and
// the solver that fallback produces is adopted as the new session state.
// Accepted optima run an explicit dual-feasibility pass every solve
// (sessionCertChainLimit, default 0 — see its comment for why the stateless
// K=8 trust window is not safe on the session's trajectory), but a failed
// pass is repaired by the stateless path's capped Neumaier–Shcherbina bound
// correction (see SolveRows in incremental.go) instead of the full primal
// resume the session's original verify-every-accept posture paid, which
// consumed glass4's entire node budget. Cold Phase I remains the
// infeasibility reference.
//
// A Session is single-goroutine, like the Incremental it wraps.
type Session struct {
	inc    *Incremental
	states []*sessionState // MRU; small fixed cap
}

// sessionStateCap bounds the number of live solvers a session keeps. The MIP
// layer solves at most a few distinct shapes per tree walk through one cache
// (the node LP's RowSet plus occasional bare-body probe solves); anything
// beyond that is transient and not worth a resident factorization.
const sessionStateCap = 3

// sessionFailStreakLimit drops a live state after this many consecutive
// failed in-place attempts, so a numerically hopeless basis cannot tax every
// node with a wasted dual run before its stateless fallback.
const sessionFailStreakLimit = 2

// sessionState is one live solver keyed by the RowSet identity it was
// assembled on.
type sessionState struct {
	s   *Solver
	ext *RowSet // == s.form.ext; nil for the base body
	// phaseIICost caches the min-sense Phase II cost vector for s's column
	// layout (constant across the tree: the objective never changes).
	phaseIICost []float64
	// dualFeasible records whether the solver was left on a basis the session
	// may continue from in place: verified dual-feasible, dual-feasible by
	// the bounded chain induction (a trusted accept), or corrected (known
	// dual-infeasible but with certChain at the limit, so every further
	// accept re-runs the explicit check — the same posture the stateless path
	// takes when it hands a corrected warm handle to descendants). Soundness
	// rides on the accept-time verification cadence, not on this flag.
	dualFeasible bool
	// certChain mirrors WarmStart.certChain for the live basis: consecutive
	// accepts trusted without the explicit dual-feasibility pass. It follows
	// the basis lineage — installing a caller handle adopts the handle's
	// depth; in-place continues keep the session's own.
	certChain  int
	failStreak int

	// lastPath describes the acceptance route of the most recent reoptDone,
	// for the QOPT_SESSION_VERIFY diagnostics only.
	lastPath string
}

// NewSession wraps inc with focus-node state. The zero session simply
// delegates every solve until the first adoptable Optimal.
func NewSession(inc *Incremental) *Session {
	return &Session{inc: inc}
}

// SolveRows is the session-aware equivalent of Incremental.SolveRows. ext
// identity (pointer) selects the live state; on a match the solve runs in
// place, otherwise — and on any in-place failure — it delegates to the
// stateless path and adopts the resulting solver.
func (ss *Session) SolveRows(ctx context.Context, opts Options, ext *RowSet, b, upper []float64, warm *WarmStart) Solution {
	var attempted Stats
	attemptedIters := 0
	if st := ss.lookup(ext); st != nil {
		sol, outcome := st.reoptimize(ss.inc, ctx, opts, ext, b, upper, warm)
		if outcome == reoptDone {
			st.failStreak = 0
			ss.touch(st)
			if sol.Solver != nil {
				sol.Solver = sol.Solver.detach()
			}
			if sessionVerify {
				ss.verifyAgainstStateless(ctx, opts, ext, b, upper, warm, sol, st.lastPath)
			}
			return sol
		}
		if outcome == reoptAttempted {
			attempted = st.s.stats
			attemptedIters = st.s.iterations
			// The session already ran the warm dual attempt this handle
			// stands for; repeating it through the stateless path before its
			// cold rebuild would double the failed work (measured: tripled
			// no-entering exits on timtab1). Go straight to the cold
			// reference, the same escalation SolveRows itself performs after
			// a failed warm attempt.
			warm = nil
			st.failStreak++
			if st.failStreak >= sessionFailStreakLimit {
				ss.drop(st)
			}
		}
	}

	sol := ss.inc.SolveRows(ctx, opts, ext, b, upper, warm)
	if attempted.LPSolves > 0 {
		// The failed in-place attempt burned real pivots and factor work;
		// account for it the same way SolveRows' internal cold rebuild
		// accounts for its abandoned warm attempt (LPSolves counts engine
		// attempts there too).
		sol.Stats.Add(attempted)
		sol.Stats.SessionFallbacks++
		sol.Iterations += attemptedIters
	}
	if sol.Status == Optimal && sol.Solver != nil {
		live := sol.Solver.detach()
		chain := 0
		if sol.Warm != nil {
			// The stateless accept may itself have ridden the certificate;
			// the adopted basis inherits that lineage depth.
			chain = sol.Warm.certChain
		}
		ss.adopt(live, chain)
		sol.Stats.SessionAdoptions++
	}
	return sol
}

// verifyAgainstStateless is the QOPT_SESSION_VERIFY harness. The session's
// contract is drop-in parity with Incremental.SolveRows, so the reference is
// the stateless solve of the same request WITH the same warm handle — the
// call the session fronted. Warm-accepted vertices are only optimal to
// tolerance (and trusted/corrected accepts deliberately carry the certified
// cadence's bounded drift); a cold re-solve is therefore NOT a sound
// equality reference for them — it is one only when the session itself went
// through the stateless path, which SolveRows already covers.
//
// Run the harness with QOPT_CERT_CHAIN=0. The session verifies every accept
// (sessionCertChainLimit 0) while the stateless default trusts K=8 chains,
// so under default knobs the reference itself can return an unverified
// drifted vertex that the session's per-accept check catches and repairs —
// a mismatch that indicts the reference, not the session. Forcing the
// stateless path to verify everywhere makes the two sides comparable.
func (ss *Session) verifyAgainstStateless(ctx context.Context, opts Options, ext *RowSet, b, upper []float64, warm *WarmStart, got Solution, path string) {
	if got.Status == TimeLimit {
		return
	}
	ref := ss.inc.SolveRows(ctx, opts, ext, b, upper, warm)
	// A reference that hits a limit (pk1-class degenerate cycling) is
	// inconclusive, not a divergence.
	if ref.Status != Optimal && ref.Status != Infeasible {
		return
	}
	if ref.Status != got.Status {
		panic("session verify: status mismatch: session=" + got.Status.String() +
			" stateless=" + ref.Status.String() + " path=" + path)
	}
	if got.Status == Optimal {
		tol := 1e-6 * math.Max(1, math.Abs(ref.Objective))
		if math.Abs(got.Objective-ref.Objective) > tol {
			panic("session verify: objective mismatch: session=" +
				strconvFormat(got.Objective) + " stateless=" + strconvFormat(ref.Objective) + " path=" + path)
		}
	}
}

func strconvFormat(v float64) string {
	return strconv.FormatFloat(v, 'g', 17, 64)
}

// lookup returns the live state assembled on ext, or nil.
func (ss *Session) lookup(ext *RowSet) *sessionState {
	for _, st := range ss.states {
		if st.s.form != nil && st.s.form.ext == ext {
			return st
		}
	}
	return nil
}

// touch moves st to the MRU front.
func (ss *Session) touch(st *sessionState) {
	for i, cur := range ss.states {
		if cur == st {
			if i != 0 {
				copy(ss.states[1:i+1], ss.states[:i])
				ss.states[0] = st
			}
			return
		}
	}
}

// drop removes st.
func (ss *Session) drop(st *sessionState) {
	for i, cur := range ss.states {
		if cur == st {
			ss.states = append(ss.states[:i], ss.states[i+1:]...)
			return
		}
	}
}

// adopt installs a detached solver as the live state for its own RowSet
// identity, replacing a previous state on the same identity and evicting the
// LRU state beyond the cap. certChain is the accept-lineage depth of the
// adopted basis (the stateless solve's outgoing warm-handle chain).
func (ss *Session) adopt(s *Solver, certChain int) {
	if s == nil || s.form == nil {
		return
	}
	st := &sessionState{
		s:            s,
		ext:          s.form.ext,
		phaseIICost:  makePhaseIICost(s),
		dualFeasible: true,
		certChain:    certChain,
	}
	if old := ss.lookup(s.form.ext); old != nil {
		ss.drop(old)
	}
	if len(ss.states) >= sessionStateCap {
		ss.states = ss.states[:sessionStateCap-1]
	}
	ss.states = append([]*sessionState{st}, ss.states...)
}

// detach returns a read-consistent copy of the solver for callers that hold
// basis-level state past this solve (cut generation, reduced-cost fixing,
// session adoption). The mutable per-solve arrays are copied; the factor, the
// compiled form, and the scratch buffers are shared. Sharing is safe
// single-threaded: the factor and eta columns are read-only after
// construction (further pivots only append to the eta slice or replace the
// factor pointer, never rewrite either), and scratch is transient within one
// accessor call.
func (s *Solver) detach() *Solver {
	c := *s
	c.basicCols = append([]int(nil), s.basicCols...)
	c.status = append([]ColStatus(nil), s.status...)
	c.xBasic = append([]float64(nil), s.xBasic...)
	c.upper = append([]float64(nil), s.upper...)
	c.bNorm = append([]float64(nil), s.bNorm...)
	c.etaUpdates = s.etaUpdates[:len(s.etaUpdates):len(s.etaUpdates)]
	return &c
}

// handleCompatible reports whether warm can be installed verbatim into the
// live solver: identical standard-form layout (same row-sign flips, so the
// aux columns mean the same thing) and content-identical ext rows. This is
// classifyWarm's exact-match class; drifted flips and prefix extensions stay
// on the stateless path.
func (st *sessionState) handleCompatible(warm *WarmStart, ext *RowSet) bool {
	s := st.s
	return warm != nil &&
		len(warm.basicCols) == s.m && len(warm.nonBasicStatus) == s.nCols &&
		boolSliceEqual(warm.flip, s.form.flip) &&
		rowSetSensesEqual(warm.extSenses, ext, len(warm.extSenses)) &&
		rowSetPrefixEqual(warm.extSenses, warm.extRows, ext, len(warm.extRows))
}

// basisEquals reports whether warm names exactly the live solver's current
// basis (same basic column per row and same non-basic statuses). Installing
// such a handle is a no-op for the factorization, so the refactor is skipped
// and the live factor + eta chain carries across.
func (st *sessionState) basisEquals(warm *WarmStart) bool {
	s := st.s
	for i, bc := range warm.basicCols {
		if s.basicCols[i] != bc {
			return false
		}
	}
	for j, stat := range warm.nonBasicStatus {
		if s.status[j] != stat {
			return false
		}
	}
	return true
}

// reoptOutcome describes how a session reoptimize ended: finished with a
// returnable Solution, attempted a dual run but could not finish soundly
// (the caller escalates to a cold stateless solve), or skipped without
// touching the solver (the caller runs the stateless path with the original
// warm handle).
type reoptOutcome int

const (
	reoptDone reoptOutcome = iota
	reoptAttempted
	reoptSkipped
)

// reoptimize runs one node solve in place on the live solver. The starting
// basis is picked per node: a compatible caller handle (the node's parent
// basis) is installed — skipping the refactor when it already IS the current
// basis — and only a handle-free pure tightening continues from the current
// basis. On reoptAttempted/reoptSkipped the live solver is left structurally
// consistent so a later in-place attempt remains legal.
func (st *sessionState) reoptimize(inc *Incremental, ctx context.Context, opts Options, ext *RowSet, b, upper []float64, warm *WarmStart) (Solution, reoptOutcome) {
	s := st.s
	if len(b) != s.m {
		return Solution{}, reoptSkipped
	}

	// Start-basis selection, before any mutation. The caller's handle (the
	// node's parent basis) has priority: measured on neos17/timtab1, starting
	// from "whatever the session solved last" — even restricted to pure bound
	// tightenings — costs more dual iterations and stalls than it saves in
	// setup, because probes and heuristic dives on the same cache drag the
	// live basis far from the next node's optimum. Continue-in-place without
	// a handle only when the bounds monotonically tightened.
	installHandle := false
	// certEligible mirrors the stateless certEligible premise: the accept's
	// dual feasibility is implied by induction only when the start basis is
	// dual-feasible by lineage AND no structural bound was relaxed (a
	// relaxation can make the start dual-infeasible, so the dual run's
	// "Optimal" exit proves nothing — exactly classifyWarm's uncertified
	// class). Harris dual pricing voids the induction the same way it does
	// on the stateless path.
	certEligible := false
	switch {
	case st.handleCompatible(warm, ext):
		// In-place continue when the handle IS the current basis (the
		// plunge/dive pattern: the parent was the previous solve on this
		// state); otherwise install it, which matches the stateless path's
		// start exactly. Either way the chain lineage is the handle's: the
		// basis about to be reoptimized is the one the handle's accept
		// history stands behind.
		installHandle = sessionInstallOnly || !st.basisEquals(warm)
		st.certChain = warm.certChain
		certEligible = warmUppersNotRelaxed(warm, upper, s.n) && !harrisDual()
	case warm == nil && !sessionInstallOnly && st.dualFeasible && st.tighteningOnly(upper):
		certEligible = !harrisDual()
	default:
		return Solution{}, reoptSkipped
	}

	s.ctx = ctx
	s.start = opts.Start
	// The certificate attempt on a "no entering column" exit is integral to
	// the focus-node design: it is what converts a warm dual infeasibility
	// signal into a proof instead of a cold Phase I re-derivation.
	opts.CertifyInfeasible = true
	s.opts = opts
	s.iterations = 0
	s.stats = Stats{LPSolves: 1, SessionReopts: 1}
	s.forceDenseFactor = false
	s.denseRescueFutile = inc.denseRescueDisabled()

	inc.orientRHS(b, s.bNorm, s.form.flip)

	switch {
	case installHandle:
		s.setStructuralUppers(upper)
		if err := s.applyWarmStart(warm, s.bNorm); err != nil {
			s.stats.WarmFailures++
			s.stats.WarmInstallFailures++
			return Solution{}, reoptAttempted
		}
		s.stats.WarmReopts++
	default:
		// No-refactor start: the handle IS the current basis, or a handle-free
		// pure tightening continues from it. Re-anchor to the new bounds,
		// reuse the live factorization — the refactor the session exists to
		// skip — but guard it with the same relative basis-residual rule the
		// pivot path applies: past tolerance, refactor before trusting x_B.
		if warm != nil {
			s.stats.WarmReopts++
		}
		s.setStructuralUppers(upper)
		s.clampNonBasicStatuses()
		s.recomputeBasicValues()
		if _, relResid, _ := s.basisResidualInf(); relResid > s.etaResidualTolerance() {
			if err := s.refactor(); err != nil {
				return Solution{}, reoptAttempted
			}
		}
	}

	dualCap := opts.MaxIterations / 4
	if dualCap < 2000 {
		dualCap = 2000
	}
	if dualCap > opts.MaxIterations {
		dualCap = opts.MaxIterations
	}

	st.dualFeasible = false // re-established only by a verified exit below
	status := s.runDualSimplexCapped(st.phaseIICost, dualCap)
	// No refactor-and-retry escalation on an uncertified "no entering column"
	// exit: measured on neos17/timtab1, the retried dual almost always
	// repeats the same exit (0 and 15 certificates against 92 and 511
	// retries), so the extra refactor + dual run per infeasible node just
	// doubled the failed work before the cold Phase I reference.

	switch status {
	case Optimal:
		if _, ok := s.checkArtificialResidual(); !ok {
			s.stats.WarmArtificialFailure++
			debugf("session: dual reoptimize Optimal but artificials drifted — stateless fallback")
			return Solution{}, reoptAttempted
		}
		// Accepted optima ride the same certified-chain cadence as the
		// stateless path. "Dual simplex preserves dual feasibility" is the
		// same tolerance-bounded induction as the classifyWarm certificate —
		// it was its unbounded form that cost pk1 the five-minute proof
		// (claimed optimal 12 vs true 11) — so an accept is trusted only
		// while the chain is under certChainLimit; at the limit the explicit
		// pass runs, and a failure is repaired by the capped bound
		// correction. The session's earlier verify-every-accept posture paid
		// a full primal resume per failure, which consumed glass4's entire
		// node budget; the resume is now the past-the-cap escape only.
		if certEligible && (sessionCertChainLimit < 0 || st.certChain < sessionCertChainLimit) {
			s.stats.WarmCertAccepts++
			st.certChain++
			st.dualFeasible = true
			st.lastPath = "trusted/install=" + strconv.FormatBool(installHandle)
			sol := extractSolution(s, inc.c, opts)
			if sol.Warm != nil {
				sol.Warm.certChain = st.certChain
			}
			return sol, reoptDone
		}
		if certEligible {
			// Chain-limit verification with the hybrid repair, verbatim from
			// the stateless path: a failure within the cap becomes a bound
			// correction; only past the cap (or on an infinite-range
			// violation) does the solve pay the primal resume below.
			s.stats.WarmCertVerifies++
			gap, finite := s.dualInfeasGap(st.phaseIICost)
			if gap == 0 {
				st.certChain = 0
				st.dualFeasible = true
				st.lastPath = "verified/install=" + strconv.FormatBool(installHandle)
				return extractSolution(s, inc.c, opts), reoptDone
			}
			s.stats.WarmCertVerifyFails++
			if finite && gap <= certCorrectionCap(s, inc.c) {
				s.stats.WarmCertBoundCorrections++
				if gap > s.stats.WarmCertBoundCorrectionMax {
					s.stats.WarmCertBoundCorrectionMax = gap
				}
				st.certChain++
				st.dualFeasible = true
				st.lastPath = "corrected/install=" + strconv.FormatBool(installHandle)
				sol := extractSolution(s, inc.c, opts)
				// Same sign convention as the stateless corrected accept:
				// phaseIICost is the negated (min-sense) objective, so the
				// min-sense correction raises the max-sense bound.
				sol.Objective += gap
				if sol.Warm != nil {
					sol.Warm.certChain = st.certChain
				}
				return sol, reoptDone
			}
			s.stats.WarmCertVerifyResumes++
		} else if s.dualFeasibleAt(st.phaseIICost) {
			// Uncertified accept (bound relaxation or Harris): the explicit
			// pass vouches for the basis, the same as the stateless path.
			st.certChain = 0
			st.dualFeasible = true
			st.lastPath = "verified/install=" + strconv.FormatBool(installHandle)
			return extractSolution(s, inc.c, opts), reoptDone
		}
		// Primal-feasible but dual-infeasible with no cheap repair: resume
		// the primal Phase II in place for the true bound, which also
		// repairs the basis for later solves on this state.
		debugf("session: dual reoptimize Optimal but basis is dual-infeasible — resuming primal Phase II")
		for _, j := range s.artificials {
			s.upper[j] = 0
			if s.status[j] != ColBasic {
				s.status[j] = ColArtificial
			}
		}
		t0 := time.Now()
		st2 := s.runSimplex(st.phaseIICost, false)
		s.stats.PhaseIINanos += time.Since(t0).Nanoseconds()
		switch st2 {
		case Optimal:
			if _, ok := s.checkArtificialResidual(); ok {
				st.certChain = 0
				st.dualFeasible = true
				st.lastPath = "primal-resume/install=" + strconv.FormatBool(installHandle)
				return extractSolution(s, inc.c, opts), reoptDone
			}
			debugf("session: primal resume Optimal but artificials drifted — stateless fallback")
			return Solution{}, reoptAttempted
		case TimeLimit:
			return statusSolution(s, TimeLimit), reoptDone
		case Unbounded:
			return statusSolution(s, Unbounded), reoptDone
		}
		return Solution{}, reoptAttempted
	case TimeLimit:
		return statusSolution(s, TimeLimit), reoptDone
	case Infeasible:
		if s.dualInfeasibilityCertified {
			s.stats.WarmInfeasibleCandidates++
			if !infeasibilityCertificateShadow {
				s.stats.WarmDualInfeasible++
				s.stats.WarmInfeasibleCertificates++
				// The basis is dual-feasible (the dual simplex maintained it
				// through every pivot) — only primal repair failed, with a
				// proof. The state remains a sound reopt start.
				st.dualFeasible = true
				sol := statusSolution(s, Infeasible)
				sol.InfeasibilityCertified = true
				sol.FarkasProof = s.farkasProof
				return sol, reoptDone
			}
		}
		s.stats.WarmDualInfeasible++
		return Solution{}, reoptAttempted
	default:
		// IterationLimit: dual stall or cap. The stateless fallback re-proves
		// from a cold start, as the warm path always has.
		s.stats.WarmDualIterationCaps++
		return Solution{}, reoptAttempted
	}
}

// tighteningOnly reports whether every structural upper bound in the request
// is at most the live solver's current one.
func (st *sessionState) tighteningOnly(upper []float64) bool {
	s := st.s
	for j := 0; j < s.n; j++ {
		u := math.Inf(1)
		if j < len(upper) {
			u = upper[j]
		}
		if u > s.upper[j] {
			return false
		}
	}
	return true
}

// setStructuralUppers rewrites the structural upper bounds in place. Only
// structural columns change between nodes: slack/surplus uppers are +Inf for
// life, range-slack widths are constant, and artificial uppers keep whatever
// pinning Phase I left.
func (s *Solver) setStructuralUppers(upper []float64) {
	for j := 0; j < s.n; j++ {
		u := math.Inf(1)
		if j < len(upper) {
			u = upper[j]
		}
		s.upper[j] = u
	}
}

// clampNonBasicStatuses re-anchors structural non-basic statuses to the
// current bounds, mirroring applyWarmStart's clamp for the in-place paths
// that skip the install copy.
func (s *Solver) clampNonBasicStatuses() {
	for j := 0; j < s.n; j++ {
		if s.status[j] != ColAtUpper {
			continue
		}
		if math.IsInf(s.upper[j], 1) || s.upper[j] <= s.lower[j] {
			s.status[j] = ColAtLower
		}
	}
}
