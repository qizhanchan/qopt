package qopt

import (
	"context"
	"time"

	"github.com/qizhanchan/qopt/lp"
)

// lpRequest is the input to an lpBackend. The matrix and vectors are expected
// to be already scaled by the caller (e.g. via ruizScale). The objective
// vector c is in maximization sense — Minimize models have already been
// negated upstream.
//
// upper holds canonical-y-space upper bounds (= original upper − lower; +Inf
// for unbounded). Populated when the caller wants the backend to handle
// bounds natively instead of via synthetic UB rows. The tableau backend
// ignores it because canonicalize still adds UB rows for cut-generation
// compatibility; the revised backend consumes it directly.
//
// warm is an optional warm-start handle from a previous solve on the same
// or a closely related LP. Backends that do not support warm starts ignore
// it. Cross-backend handles are silently rejected.
type lpRequest struct {
	c      []float64
	a      [][]float64
	senses []Sense
	b      []float64
	upper  []float64
	warm   *lpWarmStart
}

// lpResponse is the result returned by an lpBackend.
//
// values are in the same y-space the request used (caller unscales and adds
// lower bounds back). warm is populated when the backend supports warm
// starts and the solve produced a basis worth carrying forward. state is
// populated only by the tableau backend, for Gomory cut generation; other
// backends leave it nil and cuts are skipped. revisedState is the revised
// backend's analog, populated on every successful revised solve — cut
// generation (revisedGomoryCuts) reads it directly off the final basis
// instead of re-solving on the tableau backend.
type lpResponse struct {
	status                    Status
	objective                 float64
	values                    []float64
	iterations                int
	warm                      *lpWarmStart
	state                     *simplexState
	revisedState              *revisedGomoryState
	infeasibility             *lp.InfeasibilityCertificate
	infeasibilityCertified    bool
	farkasProof               *lp.FarkasProof
	revised                   RevisedStats
	propagationReductions     int
	objectiveCutoffReductions int
	objectiveCutoffPruned     bool
}

// lpWarmStart is the basis handle backends pass forward between solves. The
// `backend` field stamps which backend produced it so foreign backends know
// to discard it. Concrete payload lives in the per-backend sub-struct.
type lpWarmStart struct {
	backend LPBackend
	revised *lp.WarmStart
}

// lpBackend is implemented by each LP backend.
type lpBackend interface {
	Name() string
	Solve(ctx context.Context, req lpRequest, opts Options, start time.Time) lpResponse
}

func pickLPBackend(opts Options) lpBackend {
	switch opts.LPBackend {
	case LPBackendRevised:
		return revisedBackend{}
	default:
		return tableauBackend{}
	}
}

// tableauBackend wraps the legacy two-phase dense simplex behind the
// lpBackend interface. It ignores lpRequest.upper (UB rows are added by
// canonicalize for this backend) and does not produce warm-start handles.
type tableauBackend struct{}

func (tableauBackend) Name() string { return "tableau" }

func (tableauBackend) Solve(ctx context.Context, req lpRequest, opts Options, start time.Time) lpResponse {
	sres := solveSimplex(ctx, simplexProblem{
		c:      req.c,
		a:      req.a,
		senses: req.senses,
		b:      req.b,
	}, opts, start)
	return lpResponse{
		status:     sres.status,
		objective:  sres.objective,
		values:     sres.values,
		iterations: sres.iterations,
		state:      sres.state,
	}
}
