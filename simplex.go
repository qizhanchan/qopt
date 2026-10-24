package qopt

import (
	"context"
	"fmt"
	"math"
	"os"
	"time"
)

// debugEnabled toggles diagnostic output via the QOPT_DEBUG environment
// variable. Kept lazy so it doesn't affect the hot path when unset.
func debugEnabled() bool { return os.Getenv("QOPT_DEBUG") != "" }

func debugf(format string, args ...any) {
	if debugEnabled() {
		fmt.Fprintf(os.Stderr, "[qopt] "+format+"\n", args...)
	}
}

type simplexResult struct {
	status     Status
	objective  float64
	values     []float64
	iterations int
	// state exposes the internal tableau so cut generation can read the
	// optimal basis. Populated only when solveLPModel is called via
	// solveLPModelWithState; left nil otherwise.
	state *simplexState
}

// simplexState captures everything cut generation needs about the simplex's
// post-optimization tableau: the dense tableau itself, the basis assignment
// (basis[i] gives the simplex column index of the variable basic in row i),
// per-column metadata so we can map non-basic columns back to original
// variables, and the scaling factors used during the solve.
type simplexState struct {
	tab          [][]float64
	basis        []int
	colKind      []simplexColKind // length totalVars
	totalVars    int
	n            int         // number of original (canonical y_*) vars; cols [0,n) are these
	rhs          []float64   // post-normalize per-row RHS used to build the tableau
	flippedRow   []bool      // row i had its sign flipped during normalization
	rowOrigSense []Sense     // original sense before sign flip
	rowKinds     []lpRowKind // per-row: user constraint index vs synthetic UB row
	rowCoeffs    [][]float64 // unscaled canonical row coefficients in y-space
	rowRHS       []float64   // unscaled canonical row RHS after lower-bound shifting
	scales       scaleFactors
	upper        []float64 // canonical upper bound per variable (UB − lower)
	colToVar     []int     // canonical y column -> original Model var index
	varToCol     []int     // original Model var index -> canonical y column, or -1 if fixed
	modelLower   []float64 // original-space bounds after presolve tightening
	modelUpper   []float64
	fixed        []bool
	fixedVal     []float64
}

// simplexColKind tags simplex column j with the role it played in the LP
// setup. Cut generation needs this to translate a tableau row back into
// original-variable coefficients.
type simplexColKind struct {
	kind colKind
	row  int // for slack / surplus / artificial: which original constraint row
}

type colKind int

const (
	colY colKind = iota // original (canonical) variable
	colSlack
	colSurplus
	colArtificial
)

type simplexProblem struct {
	c      []float64
	a      [][]float64
	senses []Sense
	b      []float64
}

func solveLPModel(ctx context.Context, m *Model, bounds []boundConstraint, opts Options) (Result, error) {
	res, _, err := solveLPModelWithState(ctx, m, bounds, opts)
	return res, err
}

// solveLPModelWithState is the LP entry point that returns the simplex's
// internal state alongside the solution. Used by Gomory cut generation, which
// reads the optimal tableau and basis to derive valid inequalities.
func solveLPModelWithState(ctx context.Context, m *Model, bounds []boundConstraint, opts Options) (Result, *simplexState, error) {
	res, resp, _, err := solveLPModelWithBackend(ctx, m, bounds, opts, nil)
	return res, resp.state, err
}

// solveLPModelWithBackend is the full LP entry point. It runs canonicalize +
// scaling once, dispatches to the configured backend, and unscales the
// solution back to the user variable space. The returned lpResponse is the
// raw backend output (state for cuts, warm for next-node reuse). lpProblem
// is returned so MIP can pass it back to a sibling solve when reusing the
// model body across bound changes.
//
// `warm` is an optional warm-start handle. It is forwarded verbatim to the
// backend; backends discard handles that don't match their own backend type.
func solveLPModelWithBackend(ctx context.Context, m *Model, bounds []boundConstraint, opts Options, warm *lpWarmStart) (Result, lpResponse, lpProblem, error) {
	lp := canonicalizeWithOptions(m, bounds, warm == nil, equalityAggregationEnabled(opts))
	backend := pickLPBackend(opts)
	if lp.hasWarmStartBreakingReductions() {
		warm = nil
	}

	// The revised backend handles upper bounds natively as per-variable
	// bounds. Drop the synthetic UB rows from the matrix so that the
	// standard form (number of rows, slack/artificial column count) stays
	// stable across B&B nodes — which is the precondition for warm-start
	// basis re-use. The tightening-infeasibility row (idx == −1) and all
	// real user constraints are preserved.
	if opts.LPBackend == LPBackendRevised {
		lp = stripUBRows(lp)
	}

	cMax := make([]float64, len(lp.c))
	copy(cMax, lp.c)
	if lp.direction == Minimize {
		for i := range cMax {
			cMax[i] = -cMax[i]
		}
	}

	// Equilibrate the constraint matrix so feasibility tolerances are
	// meaningful regardless of the user's data scale. lp.a / lp.b / cMax are
	// freshly allocated by canonicalize on every B&B call, so mutating them
	// in place is safe. Identity scale → no-op when m==0.
	factors := ruizScale(lp.a, lp.b, cMax, 8)

	// Canonical y-space upper bounds = original upper − lower. Passed through
	// to backends that support bounded-variable form natively.
	ub := make([]float64, len(lp.upper))
	for i := range lp.upper {
		ub[i] = lp.upper[i] - lp.lower[i]
		// Scaling shrinks the y-axis by colScale[i]; the bound shrinks the
		// same way so y_i ≤ ub_canonical / colScale_i.
		if !math.IsInf(ub[i], 1) {
			ub[i] /= factors.colScale[i]
		}
	}

	start := time.Now()
	resp := backend.Solve(ctx, lpRequest{
		c:      cMax,
		a:      lp.a,
		senses: lp.senses,
		b:      lp.b,
		upper:  ub,
		warm:   warm,
	}, opts, start)
	if resp.state != nil {
		resp.state.scales = factors
		resp.state.rowKinds = lp.rowKinds
		resp.state.rowCoeffs = unscaleCanonicalRows(lp.a, factors)
		resp.state.rowRHS = unscaleCanonicalRHS(lp.b, factors)
		resp.state.upper = ub
		resp.state.colToVar = lp.colToVar
		resp.state.varToCol = lp.varToCol
		resp.state.modelLower = lp.origLower
		resp.state.modelUpper = lp.origUpper
		resp.state.fixed = lp.fixed
		resp.state.fixedVal = lp.fixedVal
	}
	if resp.revisedState != nil {
		resp.revisedState.scales = factors
		resp.revisedState.rowKinds = lp.rowKinds
		resp.revisedState.colToVar = lp.colToVar
		resp.revisedState.varToCol = lp.varToCol
		resp.revisedState.modelLower = lp.origLower
		resp.revisedState.modelUpper = lp.origUpper
		resp.revisedState.fixed = lp.fixed
		resp.revisedState.fixedVal = lp.fixedVal
	}
	res := Result{
		Status:     resp.status,
		Iterations: resp.iterations,
		Stats:      &SolverStats{Presolve: lp.presolveStats(m)},
	}
	if resp.revised.LPSolves > 0 {
		res.Stats.addRevised(resp.revised)
	}
	if resp.status != Optimal {
		return res, resp, lp, nil
	}

	res.Values = make([]float64, len(m.vars))
	for i := range res.Values {
		if i < len(lp.fixed) && lp.fixed[i] {
			res.Values[i] = lp.fixedVal[i]
		}
	}
	for i, y := range resp.values {
		// Unscale y back to the original variable space, then add the lower
		// bound that canonicalize folded in.
		if i >= len(lp.colToVar) {
			continue
		}
		orig := lp.colToVar[i]
		res.Values[orig] = y*factors.colScale[i] + lp.lower[i]
	}
	for _, recovery := range lp.recoveries {
		value := recovery.constant
		for _, term := range recovery.terms {
			value += term.coef * res.Values[term.v]
		}
		res.Values[recovery.variable] = value
	}
	if lp.direction == Minimize {
		res.Objective = -resp.objective + lp.offset
	} else {
		res.Objective = resp.objective + lp.offset
	}
	return res, resp, lp, nil
}

func unscaleCanonicalRows(rows [][]float64, factors scaleFactors) [][]float64 {
	out := make([][]float64, len(rows))
	for i, row := range rows {
		out[i] = make([]float64, len(row))
		for j, coef := range row {
			out[i][j] = coef / (factors.rowScale[i] * factors.colScale[j])
		}
	}
	return out
}

func unscaleCanonicalRHS(rhs []float64, factors scaleFactors) []float64 {
	out := make([]float64, len(rhs))
	for i, value := range rhs {
		out[i] = value / factors.rowScale[i]
	}
	return out
}

func solveSimplex(ctx context.Context, p simplexProblem, opts Options, start time.Time) simplexResult {
	n := len(p.c)
	m := len(p.a)
	if m == 0 {
		for _, c := range p.c {
			if c > opts.OptimalityTol {
				return simplexResult{status: Unbounded}
			}
		}
		return simplexResult{status: Optimal, values: make([]float64, n)}
	}

	rows := make([][]float64, m)
	senses := make([]Sense, m)
	rhs := make([]float64, m)
	flippedRow := make([]bool, m)
	rowOrigSense := make([]Sense, m)
	for i := 0; i < m; i++ {
		rows[i] = make([]float64, n)
		copy(rows[i], p.a[i])
		senses[i] = p.senses[i]
		rowOrigSense[i] = p.senses[i]
		rhs[i] = p.b[i]
		if rhs[i] < 0 {
			rhs[i] = -rhs[i]
			for j := range rows[i] {
				rows[i][j] = -rows[i][j]
			}
			senses[i] = flipSense(senses[i])
			flippedRow[i] = true
		}
	}

	totalVars := n
	basis := make([]int, m)
	artificial := map[int]bool{}
	// colKind[j] is appended in lockstep with totalVars so cut generation can
	// later identify whether column j is an original y_*, a slack, a surplus,
	// or an artificial — and which constraint row each aux column belongs to.
	colKind := make([]simplexColKind, n, n+3*m)
	for j := 0; j < n; j++ {
		colKind[j] = simplexColKind{kind: colY}
	}
	for i := 0; i < m; i++ {
		switch senses[i] {
		case LessOrEqual:
			basis[i] = totalVars
			colKind = append(colKind, simplexColKind{kind: colSlack, row: i})
			totalVars++
		case GreaterOrEqual:
			colKind = append(colKind, simplexColKind{kind: colSurplus, row: i})
			totalVars++ // surplus
			basis[i] = totalVars
			artificial[totalVars] = true
			colKind = append(colKind, simplexColKind{kind: colArtificial, row: i})
			totalVars++
		case Equal:
			basis[i] = totalVars
			artificial[totalVars] = true
			colKind = append(colKind, simplexColKind{kind: colArtificial, row: i})
			totalVars++
		}
	}

	tab := make([][]float64, m+1)
	for i := range tab {
		tab[i] = make([]float64, totalVars+1)
	}
	nextAux := n
	for i := 0; i < m; i++ {
		copy(tab[i][:n], rows[i])
		switch senses[i] {
		case LessOrEqual:
			tab[i][nextAux] = 1
			nextAux++
		case GreaterOrEqual:
			tab[i][nextAux] = -1
			nextAux++
			tab[i][nextAux] = 1
			nextAux++
		case Equal:
			tab[i][nextAux] = 1
			nextAux++
		}
		tab[i][totalVars] = rhs[i]
	}

	// Two-phase simplex. Phase I minimizes the sum of artificial-variable
	// magnitudes to find a feasible basis; Phase II then optimizes the
	// original objective. We chose two-phase over Big-M after empirical
	// comparison on the MIPLIB sub-suite: Big-M did not rescue the two
	// stuck-Phase-I instances (glass4, neos-3046615-murg — see bench/) and
	// the M penalty's numerical noise mildly degraded incumbents on others
	// like assign1-5-8 and mad. The remaining false-infeasibility cases are
	// understood as accumulated float error in the dense tableau and would
	// need basis refactorization to recover from.
	iterations := 0
	if len(artificial) > 0 {
		phaseCost := make([]float64, totalVars)
		for idx := range artificial {
			phaseCost[idx] = -1
		}
		setObjective(tab, basis, phaseCost)
		st, _ := simplexOptimize(ctx, tab, basis, artificial, opts, start, &iterations)
		if st != Optimal {
			return simplexResult{status: st, iterations: iterations}
		}
		if tab[m][totalVars] < -opts.FeasibilityTol {
			if debugEnabled() {
				directSum := 0.0
				for i, bi := range basis {
					if artificial[bi] {
						directSum += math.Abs(tab[i][totalVars])
					}
				}
				debugf("Phase I terminated: trackedSum=%g  directSum=%g  iters=%d  tol=%g",
					-tab[m][totalVars], directSum, iterations, opts.FeasibilityTol)
			}
			return simplexResult{status: Infeasible, iterations: iterations}
		}
		for i, b := range basis {
			if artificial[b] && math.Abs(tab[i][totalVars]) > opts.FeasibilityTol {
				return simplexResult{status: Infeasible, iterations: iterations}
			}
		}
	}

	cost := make([]float64, totalVars)
	copy(cost, p.c)
	setObjective(tab, basis, cost)
	st, _ := simplexOptimize(ctx, tab, basis, artificial, opts, start, &iterations)
	if st != Optimal {
		return simplexResult{status: st, iterations: iterations}
	}

	values := make([]float64, n)
	for i, b := range basis {
		if b < n {
			values[b] = cleanZero(tab[i][totalVars], opts.FeasibilityTol)
		}
	}
	return simplexResult{
		status:     Optimal,
		objective:  cleanZero(tab[m][totalVars], opts.FeasibilityTol),
		values:     values,
		iterations: iterations,
		state: &simplexState{
			tab:          tab,
			basis:        basis,
			colKind:      colKind,
			totalVars:    totalVars,
			n:            n,
			rhs:          rhs,
			flippedRow:   flippedRow,
			rowOrigSense: rowOrigSense,
		},
	}
}

func setObjective(tab [][]float64, basis []int, cost []float64) {
	obj := tab[len(tab)-1]
	for j := range obj {
		obj[j] = 0
	}
	for j, c := range cost {
		obj[j] = -c
	}
	rhs := len(obj) - 1
	for i, b := range basis {
		cb := 0.0
		if b >= 0 && b < len(cost) {
			cb = cost[b]
		}
		if cb == 0 {
			continue
		}
		for j := 0; j <= rhs; j++ {
			obj[j] += cb * tab[i][j]
		}
	}
}

func simplexOptimize(ctx context.Context, tab [][]float64, basis []int, artificial map[int]bool, opts Options, start time.Time, iterations *int) (Status, int) {
	m := len(tab) - 1
	rhs := len(tab[0]) - 1
	local := 0
	for {
		if err := ctx.Err(); err != nil {
			return TimeLimit, local
		}
		if opts.TimeLimit > 0 && time.Since(start) >= opts.TimeLimit {
			return TimeLimit, local
		}
		if *iterations >= opts.MaxIterations {
			return IterationLimit, local
		}

		enter := -1
		for j := 0; j < rhs; j++ {
			if artificial[j] {
				continue
			}
			if tab[m][j] < -opts.OptimalityTol {
				enter = j
				break
			}
		}
		if enter == -1 {
			return Optimal, local
		}

		leave := -1
		bestRatio := math.Inf(1)
		for i := 0; i < m; i++ {
			a := tab[i][enter]
			if a <= opts.FeasibilityTol {
				continue
			}
			ratio := tab[i][rhs] / a
			if ratio < bestRatio-opts.FeasibilityTol || (math.Abs(ratio-bestRatio) <= opts.FeasibilityTol && (leave == -1 || basis[i] < basis[leave])) {
				bestRatio = ratio
				leave = i
			}
		}
		if leave == -1 {
			return Unbounded, local
		}
		pivot(tab, leave, enter)
		basis[leave] = enter
		*iterations++
		local++
	}
}

func pivot(tab [][]float64, row, col int) {
	p := tab[row][col]
	for j := range tab[row] {
		tab[row][j] /= p
	}
	for i := range tab {
		if i == row {
			continue
		}
		f := tab[i][col]
		if f == 0 {
			continue
		}
		for j := range tab[i] {
			tab[i][j] -= f * tab[row][j]
		}
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
