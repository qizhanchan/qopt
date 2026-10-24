package qopt

import (
	"context"
	"time"

	"github.com/qizhanchan/qopt/lp"
)

// revisedBackend adapts the lp package's bounded-variable revised simplex to
// the lpBackend interface. The engine itself (solver, dual reoptimize, LU
// factors, warm-start handles) lives in the lp subpackage; this file only
// converts between the qopt-side request/response shapes and lp's.

type revisedBackend struct{}

func (revisedBackend) Name() string { return "revised" }

func (revisedBackend) Solve(ctx context.Context, req lpRequest, opts Options, start time.Time) lpResponse {
	inc := lp.NewIncremental(lp.Problem{
		C:      req.c,
		Cols:   denseToCols(req.a, len(req.c)),
		Senses: req.senses,
	})
	return solveRevisedOn(inc, ctx, opts, start, req.b, req.upper, req.warm)
}

// solveRevisedOn runs one solve on a loaded lp.Incremental and adapts the
// result to the lpResponse shape. It is the shared entry for the generic
// backend path above (which loads a fresh body per request) and the
// canonCache fast path (which loads the body once per B&B tree and re-solves
// it at every node with only the RHS/upper bounds changing).
func solveRevisedOn(inc *lp.Incremental, ctx context.Context, opts Options, start time.Time, b, upper []float64, warm *lpWarmStart) lpResponse {
	return solveRevisedOnRows(inc, ctx, opts, start, nil, b, upper, warm)
}

// solveRevisedOnRows is solveRevisedOn with an optional appended RowSet (the
// canonCache cut-row path).
func solveRevisedOnRows(inc *lp.Incremental, ctx context.Context, opts Options, start time.Time, ext *lp.RowSet, b, upper []float64, warm *lpWarmStart) lpResponse {
	return solveRevisedSessionRows(nil, inc, ctx, opts, start, ext, b, upper, warm)
}

// solveRevisedSessionRows routes one node solve through the focus-node
// session when one is supplied (Options.RevisedFocusSession), falling back to
// the stateless Incremental path when sess is nil.
func solveRevisedSessionRows(sess *lp.Session, inc *lp.Incremental, ctx context.Context, opts Options, start time.Time, ext *lp.RowSet, b, upper []float64, warm *lpWarmStart) lpResponse {
	var w *lp.WarmStart
	if warm != nil && warm.backend == LPBackendRevised {
		w = warm.revised
	}
	var sol lp.Solution
	if sess != nil {
		sol = sess.SolveRows(ctx, lpOptions(opts, start), ext, b, upper, w)
	} else {
		sol = inc.SolveRows(ctx, lpOptions(opts, start), ext, b, upper, w)
	}
	resp := lpResponse{
		status:                 sol.Status,
		objective:              sol.Objective,
		values:                 sol.Values,
		iterations:             sol.Iterations,
		infeasibility:          sol.Infeasibility,
		infeasibilityCertified: sol.InfeasibilityCertified,
		farkasProof:            sol.FarkasProof,
		revised:                sol.Stats,
	}
	if sol.Warm != nil {
		resp.warm = &lpWarmStart{backend: LPBackendRevised, revised: sol.Warm}
	}
	if sol.Solver != nil {
		rowSenses := sol.Solver.RowSenses()
		normSenses := make([]Sense, len(rowSenses))
		for i, sense := range rowSenses {
			normSenses[i] = Sense(sense)
		}
		resp.revisedState = &revisedGomoryState{
			solver:     sol.Solver,
			normSenses: normSenses,
			flippedRow: sol.Solver.RowFlips(),
		}
	}
	return resp
}

// lpOptions maps the solver-wide Options onto the lp engine's knob subset.
func lpOptions(opts Options, start time.Time) lp.Options {
	return lp.Options{
		FeasibilityTol:    opts.FeasibilityTol,
		OptimalityTol:     opts.OptimalityTol,
		MaxIterations:     opts.MaxIterations,
		TimeLimit:         opts.TimeLimit,
		Start:             start,
		MaxEtaUpdates:     opts.RevisedMaxEtaUpdates,
		EtaResidualFactor: opts.RevisedEtaResidualFactor,
		DenseLU:           opts.RevisedDenseLU,
		PersistFactor:     opts.RevisedPersistFactor,
		CertifyInfeasible: !opts.RevisedNoCertifyInfeasible,
	}
}

// denseToCols converts a dense row-major matrix into lp's sparse columns
// (entries in increasing row order, zeros dropped).
func denseToCols(a [][]float64, n int) []lp.SparseVector {
	m := len(a)
	cols := make([]lp.SparseVector, n)
	for j := 0; j < n; j++ {
		for i := 0; i < m; i++ {
			if v := a[i][j]; v != 0 {
				cols[j].Append(i, v)
			}
		}
	}
	return cols
}
