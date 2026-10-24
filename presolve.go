package qopt

// tightenBounds runs deterministic activity-based bound propagation over an
// event queue. Immutable row/column adjacency is cached on Model; each call
// allocates only the node's mutable bounds, queue flags, and reason records.
//
// lower/upper are mutated in place. It returns infeasible=true if propagation
// proves a row or variable domain infeasible. maxPasses is a work cap measured
// in full-row-sweep equivalents, preserving the historical API.
func tightenBounds(m *Model, lower, upper []float64, maxPasses int) bool {
	if maxPasses <= 0 || len(m.constraints) == 0 {
		return false
	}
	d := newDomainState(m, lower, upper, 1e-9)
	return !d.propagate(maxPasses)
}
