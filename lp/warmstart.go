package lp

// WarmStart carries the basis representation between solves on the same
// canonical LP body. Variable indexing follows the problem's logical column
// space: structural x_j for j in [0, n), then per-row slack/surplus/
// artificial columns appended in row order (one slack for each LE row, one
// surplus + one artificial for each GE row, one artificial for each EQ row,
// after the b ≥ 0 sign normalization). basicCols[i] gives the column index
// basic in row i; nonBasicStatus[j] gives the status of column j.
//
// A handle is valid for a problem with the same standard-form shape and
// auxiliary-column meanings. Bounds/RHS may change; extra-row coefficients
// may also change when their senses and positions stay fixed. Solve verifies
// compatibility, refactors against a changed matrix, and silently cold-starts
// on an unusable handle.
type WarmStart struct {
	basicCols      []int
	nonBasicStatus []ColStatus
	// flip is the row-sign flip pattern of the standard form this handle was
	// produced on. When the new solve's pattern differs, the aux-column
	// layout shifts, so the handle's indices name different columns; a raw
	// install would land on a semantically wrong (often exactly singular)
	// basis. classifyWarm translates such handles into the new layout via
	// remapWarmFlips — the two forms differ only by a row-sign diagonal, so
	// the basis maps exactly — and cold-starts when translation is impossible.
	flip []bool
	// extSenses/extRows record the RowSet the producing solve ran with
	// (references into it — RowSets are immutable). classifyWarm uses them
	// to certify same-content re-solves and to extend the handle when the
	// current RowSet appends rows to this prefix.
	extSenses []Sense
	extRows   []SparseVector
	// upper is the upper bound for each column at the time the warm-start
	// was produced. Solve compares it against the new solve's bounds and
	// rejects the handle if any structural bound was RELAXED — dual
	// re-optimization is only sound for tightenings (see
	// warmUppersNotRelaxed).
	upper []float64
	// flipRemapped marks a handle synthesized by remapWarmFlips: the basis was
	// translated across a row-sign flip drift. Used for telemetry only.
	flipRemapped bool
	// certChain counts the consecutive certificate-trusted warm accepts behind
	// this handle's basis — solves whose Optimal was accepted on the
	// classifyWarm certificate alone, without an explicit dual-feasibility
	// check. Each certified reuse lets the dual ratio test's tolerance drift
	// compound (measured on pk1: a long chain accepted a dual-infeasible
	// vertex, overstating the node bound and falsely "proving" 12 where the
	// optimum is 11), so SolveRows re-verifies once the chain reaches
	// certChainLimit and resets it on every explicitly verified accept.
	certChain int
	// lu and etas carry the factorization of the basis B (base factor plus
	// the product-form eta chain) when Options.PersistFactor is set. A child
	// B&B node has the same basis matrix as its parent, so lu+etas is exactly
	// B^{-1} for the child too and can be reused without a refactor. lu is
	// shared (read-only after construction); etas is a copy so siblings don't
	// alias. nil when persistence is off.
	lu   basisFactor
	etas []etaUpdate
}

// ColStatus tags a column's location in the simplex: at its lower bound, at
// its upper bound, basic, or an artificial that has been driven out (kept on
// the column list but not eligible to re-enter).
type ColStatus int8

const (
	ColAtLower ColStatus = iota
	ColAtUpper
	ColBasic
	ColArtificial
)
