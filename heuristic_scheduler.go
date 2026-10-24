package qopt

import (
	"context"
	"time"
)

const (
	defaultHeuristicTimeFraction = 0.15
	maxHeuristicWallTime         = 30 * time.Second
	noLimitHeuristicWallTime     = 5 * time.Second
	defaultHeuristicLPSolves     = 180
	maxHeuristicCallTime         = 2 * time.Second
	maxNeighborhoodWallTime      = 3 * time.Second
)

// heuristicScheduler gives every primal heuristic one shared wall-clock and
// LP-solve budget. Cheap repair attempts consume no LP quota; pump/diving do.
// The scheduler is per Solve and deterministic except for the wall-clock stop.
type heuristicScheduler struct {
	ctx        context.Context
	wallBudget time.Duration
	wallUsed   time.Duration
	lpBudget   int
	lpUsed     int
	// Neighborhood sub-MIPs have highly instance-dependent LP costs, so their
	// portfolio is governed by wall time rather than the root-heuristic LP
	// quota. Their time still counts against wallUsed.
	neighborhoodWallUsed time.Duration
	stats                *SolverStats
	disabled             bool
}

func (h *heuristicScheduler) beginNeighborhood(name string) (context.Context, func(bool, int), bool) {
	if h == nil || h.disabled || h.ctx.Err() != nil || h.wallUsed >= h.wallBudget ||
		h.neighborhoodWallUsed >= maxNeighborhoodWallTime {
		return nil, nil, false
	}
	started := time.Now()
	remaining := h.wallBudget - h.wallUsed
	neighborhoodRemaining := maxNeighborhoodWallTime - h.neighborhoodWallUsed
	if remaining > neighborhoodRemaining {
		remaining = neighborhoodRemaining
	}
	if remaining > maxHeuristicCallTime {
		remaining = maxHeuristicCallTime
	}
	ctx, cancel := context.WithTimeout(h.ctx, remaining)
	if h.stats != nil {
		h.stats.recordHeuristic(name, false, 0, 0)
	}
	finish := func(success bool, lpSolves int) {
		cancel()
		elapsed := time.Since(started)
		h.lpUsed += lpSolves
		h.wallUsed += elapsed
		h.neighborhoodWallUsed += elapsed
		if h.stats != nil {
			h.stats.recordHeuristic(name, success, lpSolves, elapsed)
		}
	}
	return ctx, finish, true
}

func newHeuristicScheduler(ctx context.Context, opts Options, stats *SolverStats) *heuristicScheduler {
	budget := noLimitHeuristicWallTime
	if opts.TimeLimit > 0 {
		budget = time.Duration(defaultHeuristicTimeFraction * float64(opts.TimeLimit))
		if budget > maxHeuristicWallTime {
			budget = maxHeuristicWallTime
		}
	}
	if budget <= 0 {
		budget = time.Millisecond
	}
	return &heuristicScheduler{
		ctx:        ctx,
		wallBudget: budget,
		lpBudget:   defaultHeuristicLPSolves,
		stats:      stats,
		disabled:   opts.DisablePrimalHeuristics,
	}
}

func (h *heuristicScheduler) begin(name string, requestedLP int) (context.Context, func(bool, int), bool) {
	if h == nil || h.disabled || h.ctx.Err() != nil || h.wallUsed >= h.wallBudget || h.lpUsed+requestedLP > h.lpBudget {
		return nil, nil, false
	}
	started := time.Now()
	remaining := h.wallBudget - h.wallUsed
	if remaining > maxHeuristicCallTime {
		remaining = maxHeuristicCallTime
	}
	ctx, cancel := context.WithTimeout(h.ctx, remaining)
	if h.stats != nil {
		h.stats.recordHeuristic(name, false, 0, 0)
	}
	finish := func(success bool, lpSolves int) {
		cancel()
		h.lpUsed += lpSolves
		h.wallUsed += time.Since(started)
		if h.stats != nil {
			h.stats.recordHeuristic(name, success, lpSolves, time.Since(started))
		}
	}
	return ctx, finish, true
}
