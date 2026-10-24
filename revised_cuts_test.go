package qopt

import (
	"context"
	"math"
	"testing"
	"time"
)

// enumerateIntegerFeasible brute-forces every integer point within [0,bound]^n
// (bound per variable) that satisfies m's original constraints, for small
// test-only models. Used to check that a cut is a genuinely valid inequality
// — true for every integer-feasible point of the original model — rather
// than merely "not violated by the current LP vertex".
func enumerateIntegerFeasible(t *testing.T, m *Model, bounds []int) [][]float64 {
	t.Helper()
	n := len(m.vars)
	if len(bounds) != n {
		t.Fatalf("bounds length %d != vars %d", len(bounds), n)
	}
	var out [][]float64
	point := make([]float64, n)
	var rec func(i int)
	rec = func(i int) {
		if i == n {
			for _, c := range m.constraints {
				lhs := 0.0
				for v, coef := range c.Coeffs {
					lhs += coef * point[v]
				}
				switch c.Sense {
				case LessOrEqual:
					if lhs > c.RHS+1e-9 {
						return
					}
				case GreaterOrEqual:
					if lhs < c.RHS-1e-9 {
						return
					}
				case Equal:
					if math.Abs(lhs-c.RHS) > 1e-9 {
						return
					}
				}
			}
			out = append(out, append([]float64(nil), point...))
			return
		}
		for v := 0; v <= bounds[i]; v++ {
			point[i] = float64(v)
			rec(i + 1)
		}
	}
	rec(0)
	if len(out) == 0 {
		t.Fatalf("no integer-feasible points found within bounds %v — test model is misconfigured", bounds)
	}
	return out
}

func assertCutsValidOnGrid(t *testing.T, label string, cuts []Constraint, feasible [][]float64) {
	t.Helper()
	for _, cut := range cuts {
		for _, pt := range feasible {
			lhs := 0.0
			for v, coef := range cut.Coeffs {
				lhs += coef * pt[int(v)]
			}
			ok := false
			switch cut.Sense {
			case LessOrEqual:
				ok = lhs <= cut.RHS+1e-6
			case GreaterOrEqual:
				ok = lhs >= cut.RHS-1e-6
			case Equal:
				ok = math.Abs(lhs-cut.RHS) <= 1e-6
			}
			if !ok {
				t.Fatalf("%s: cut %+v (coeffs=%v sense=%v rhs=%v) excludes feasible integer point %v (lhs=%v)",
					label, cut.Name, cut.Coeffs, cut.Sense, cut.RHS, pt, lhs)
			}
		}
	}
}

// rootRevisedCuts solves m's LP relaxation on the revised backend and returns
// the native GMI cuts derived from its final basis.
func rootRevisedCuts(t *testing.T, m *Model, opts Options) []Constraint {
	t.Helper()
	opts.LPBackend = LPBackendRevised
	opts = normalizeOptions(opts)
	res, resp, _, err := solveLPModelWithBackend(context.Background(), m, nil, opts, nil)
	if err != nil {
		t.Fatalf("solve: %v", err)
	}
	if res.Status != Optimal {
		t.Fatalf("root LP status = %s, want Optimal", res.Status)
	}
	if resp.revisedState == nil {
		t.Fatalf("revisedState is nil — plumbing regression")
	}
	modelLower, modelUpper := effectiveBounds(m, nil)
	return revisedGomoryCuts(m, modelLower, modelUpper, resp.revisedState, opts)
}

// TestRevisedGomoryCuts_ValidOnBruteForceGrid is the core correctness check:
// every cut the native revised-LU generator emits must hold for every
// integer-feasible point of the original model, not just be violated at the
// current fractional LP vertex. An invalid GMI cut would silently prove a
// wrong (too-good) optimum, so this enumerates the full integer grid rather
// than trusting algebra alone.
func TestRevisedGomoryCuts_ValidOnBruteForceGrid(t *testing.T) {
	cases := []struct {
		name   string
		build  func() *Model
		bounds []int
	}{
		{
			name: "classic_6x4y",
			build: func() *Model {
				m := NewModel(Maximize)
				x := m.AddVar(VarSpec{Name: "x", Type: Integer})
				y := m.AddVar(VarSpec{Name: "y", Type: Integer})
				must(t, m.SetObjective(map[Var]float64{x: 5, y: 4}))
				must(t, m.AddConstraint(map[Var]float64{x: 6, y: 4}, LessOrEqual, 24))
				must(t, m.AddConstraint(map[Var]float64{x: 1, y: 2}, LessOrEqual, 6))
				return m
			},
			bounds: []int{4, 3},
		},
		{
			name: "ge_row",
			build: func() *Model {
				m := NewModel(Minimize)
				x := m.AddVar(VarSpec{Name: "x", Type: Integer, Upper: Float64(10)})
				y := m.AddVar(VarSpec{Name: "y", Type: Integer, Upper: Float64(10)})
				must(t, m.SetObjective(map[Var]float64{x: 2, y: 3}))
				must(t, m.AddConstraint(map[Var]float64{x: 3, y: 5}, GreaterOrEqual, 11))
				return m
			},
			bounds: []int{10, 10},
		},
		{
			// Forces a nonbasic-at-upper structural column at the LP optimum:
			// x is capped by its OWN VarSpec upper bound (5), not by a row —
			// the case gomoryCuts never has to handle because the tableau
			// backend always represents upper bounds as extra rows. y's row
			// is simultaneously left fractional (7.5), so this single case
			// exercises the bounded-variable substitution and a standard
			// basic-row GMI term together.
			name: "nonbasic_at_upper_plus_fractional_row",
			build: func() *Model {
				m := NewModel(Maximize)
				x := m.AddVar(VarSpec{Name: "x", Type: Integer, Upper: Float64(5)})
				y := m.AddVar(VarSpec{Name: "y", Type: Integer, Upper: Float64(100)})
				must(t, m.SetObjective(map[Var]float64{x: 3, y: 2}))
				must(t, m.AddConstraint(map[Var]float64{x: 1, y: 2}, LessOrEqual, 20))
				return m
			},
			bounds: []int{5, 10},
		},
		{
			// Mixes an LE row, a GE row, and an EQ row so the aux column
			// layout (slack / surplus+artificial / artificial, in row order)
			// gets exercised across all three sense kinds in one basis, with
			// x again pinned at its own upper bound.
			name: "mixed_le_ge_eq_rows",
			build: func() *Model {
				m := NewModel(Minimize)
				x := m.AddVar(VarSpec{Name: "x", Type: Integer, Upper: Float64(4)})
				y := m.AddVar(VarSpec{Name: "y", Type: Integer, Upper: Float64(20)})
				z := m.AddVar(VarSpec{Name: "z", Type: Integer, Upper: Float64(20)})
				must(t, m.SetObjective(map[Var]float64{x: -3, y: 1, z: 1}))
				must(t, m.AddConstraint(map[Var]float64{x: 1, y: 1}, LessOrEqual, 12))
				must(t, m.AddConstraint(map[Var]float64{y: 1, z: 2}, GreaterOrEqual, 9))
				must(t, m.AddConstraint(map[Var]float64{x: 2, z: 1}, Equal, 11))
				return m
			},
			bounds: []int{4, 20, 20},
		},
		{
			name: "three_var_knapsack",
			build: func() *Model {
				m := NewModel(Maximize)
				x := m.AddVar(VarSpec{Name: "x", Type: Integer, Upper: Float64(6)})
				y := m.AddVar(VarSpec{Name: "y", Type: Integer, Upper: Float64(6)})
				z := m.AddVar(VarSpec{Name: "z", Type: Integer, Upper: Float64(6)})
				must(t, m.SetObjective(map[Var]float64{x: 7, y: 5, z: 3}))
				must(t, m.AddConstraint(map[Var]float64{x: 4, y: 3, z: 2}, LessOrEqual, 17))
				return m
			},
			bounds: []int{6, 6, 6},
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			m := tc.build()
			cuts := rootRevisedCuts(t, m, Options{})
			feasible := enumerateIntegerFeasible(t, tc.build(), tc.bounds)
			if len(cuts) == 0 {
				t.Logf("%s: no cuts emitted (root LP may already be near-integral)", tc.name)
			}
			assertCutsValidOnGrid(t, tc.name, cuts, feasible)
		})
	}
}

// TestRevisedGomoryCuts_AgreesWithTableauOnOptimum cross-checks the native
// revised-backend GMI path against the well-tested tableau gomoryCuts:
// solving the same MIP to completion with per-node cuts enabled on each
// backend must reach the same optimum, since valid cuts (from either
// generator) never change the true answer — a divergence means one side
// generated an invalid (optimum-excluding) cut.
func TestRevisedGomoryCuts_AgreesWithTableauOnOptimum(t *testing.T) {
	build := func() *Model {
		m := NewModel(Maximize)
		x := m.AddVar(VarSpec{Name: "x", Type: Integer, Upper: Float64(5)})
		y := m.AddVar(VarSpec{Name: "y", Type: Integer, Upper: Float64(100)})
		must(t, m.SetObjective(map[Var]float64{x: 3, y: 2}))
		must(t, m.AddConstraint(map[Var]float64{x: 1, y: 2}, LessOrEqual, 20))
		return m
	}
	tab, errTab := build().Solve(context.Background(), Options{LPBackend: LPBackendTableau, MaxCutDepth: 6})
	rev, errRev := build().Solve(context.Background(), Options{LPBackend: LPBackendRevised, MaxCutDepth: 6})
	if errTab != nil || errRev != nil {
		t.Fatalf("solve errors: tab=%v rev=%v", errTab, errRev)
	}
	if tab.Status != Optimal || rev.Status != Optimal {
		t.Fatalf("non-optimal: tableau=%s revised=%s", tab.Status, rev.Status)
	}
	if math.Abs(tab.Objective-rev.Objective) > 1e-6 {
		t.Fatalf("tableau and revised per-node cuts disagree: tableau=%v revised=%v", tab.Objective, rev.Objective)
	}
}

// TestRevisedGomoryCuts_NoTableauFallback guards the throughput win itself:
// once revisedState is populated, generatePerNodeCuts must not silently pay
// for the extra tableau re-solve it used to require. We can't observe the
// call directly, but we can confirm resp.revisedState is populated on a
// normal revised solve — the precondition the switch in generatePerNodeCuts
// relies on to skip the fallback branch.
func TestRevisedGomoryCuts_NoTableauFallback(t *testing.T) {
	m := NewModel(Maximize)
	x := m.AddVar(VarSpec{Name: "x", Type: Integer})
	y := m.AddVar(VarSpec{Name: "y", Type: Integer})
	must(t, m.SetObjective(map[Var]float64{x: 5, y: 4}))
	must(t, m.AddConstraint(map[Var]float64{x: 6, y: 4}, LessOrEqual, 24))
	must(t, m.AddConstraint(map[Var]float64{x: 1, y: 2}, LessOrEqual, 6))

	opts := normalizeOptions(Options{LPBackend: LPBackendRevised})
	_, resp, _, err := solveLPModelWithBackend(context.Background(), m, nil, opts, nil)
	if err != nil {
		t.Fatalf("solve: %v", err)
	}
	if resp.revisedState == nil || resp.revisedState.solver == nil {
		t.Fatalf("revisedState not populated on a plain revised solve")
	}
	if len(resp.revisedState.rowKinds) == 0 {
		t.Fatalf("revisedState.rowKinds not populated by solveLPModelWithBackend")
	}
}

// checkModelFeasible verifies values satisfies m's constraints, variable
// bounds, and integrality — the same notion of feasibility bench/cmd/compare
// applies to a solver's reported incumbent.
func checkModelFeasible(t *testing.T, m *Model, values []float64, tol float64) {
	t.Helper()
	for i, spec := range m.vars {
		v := values[i]
		if v < *spec.Lower-tol || v > *spec.Upper+tol {
			t.Errorf("var %d (%s) = %v out of bounds [%v, %v]", i, spec.Name, v, *spec.Lower, *spec.Upper)
		}
		if (spec.Type == Integer || spec.Type == Binary) && math.Abs(v-math.Round(v)) > tol {
			t.Errorf("var %d (%s) = %v not integral", i, spec.Name, v)
		}
	}
	for ci, c := range m.constraints {
		lhs := 0.0
		for v, coef := range c.Coeffs {
			lhs += coef * values[v]
		}
		switch c.Sense {
		case LessOrEqual:
			if lhs > c.RHS+tol {
				t.Errorf("constraint %d (%s) violated: lhs=%v > rhs=%v", ci, c.Name, lhs, c.RHS)
			}
		case GreaterOrEqual:
			if lhs < c.RHS-tol {
				t.Errorf("constraint %d (%s) violated: lhs=%v < rhs=%v", ci, c.Name, lhs, c.RHS)
			}
		case Equal:
			if math.Abs(lhs-c.RHS) > tol {
				t.Errorf("constraint %d (%s) violated: lhs=%v != rhs=%v", ci, c.Name, lhs, c.RHS)
			}
		}
	}
}

// TestRevisedGomoryCuts_MIPLIBSmokeFeasible runs the native per-node cut path
// against two real MIPLIB instances with KNOWN optima (markshare_4_0=1,
// pk1=11 — see CLAUDE.md's 30-min reliability-branching results) under a
// short time budget. Any incumbent found must be genuinely feasible for the
// original model, and — since both are minimizations and a valid cut can
// only shrink the feasible region, never admit new points — its objective
// can never fall below the known true optimum. A regression here would mean
// a per-node cut is either infeasible-inducing or excludes the true optimum.
func TestRevisedGomoryCuts_MIPLIBSmokeFeasible(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping MIPLIB integration smoke in -short mode")
	}
	cases := []struct {
		file       string
		knownOptim float64
	}{
		{"bench/problems/miplib/markshare_4_0.json", 1},
		{"bench/problems/miplib/pk1.json", 11},
	}
	for _, tc := range cases {
		t.Run(tc.file, func(t *testing.T) {
			p := loadProblem(t, tc.file)
			m := modelFromProblem(t, p)
			res, err := m.Solve(context.Background(), Options{
				LPBackend:    LPBackendRevised,
				MIPBranching: MIPBranchReliability,
				MaxCutDepth:  5,
				TimeLimit:    8 * time.Second,
			})
			if err != nil {
				t.Fatalf("solve: %v", err)
			}
			if res.Values == nil {
				t.Logf("%s: no incumbent found in budget (acceptable — not a correctness issue)", tc.file)
				return
			}
			checkModelFeasible(t, m, res.Values, 1e-5)
			if res.Objective < tc.knownOptim-1e-6 {
				t.Fatalf("%s: reported objective %v is BETTER than the known true optimum %v — "+
					"a valid cut can only shrink the feasible region, so this means an invalid cut admitted "+
					"or fabricated a solution", tc.file, res.Objective, tc.knownOptim)
			}
			if res.Status == Optimal && math.Abs(res.Objective-tc.knownOptim) > 1e-6 {
				t.Fatalf("%s: proved Optimal at %v but the known true optimum is %v — "+
					"a cut likely excluded the true optimum", tc.file, res.Objective, tc.knownOptim)
			}
		})
	}
}

func TestRevisedAuxColumnLayoutRejectsMetadataMismatch(t *testing.T) {
	if _, _, ok := revisedAuxColumnLayout(2, 1, []Sense{GreaterOrEqual}); ok {
		t.Fatal("GE row needs surplus+artificial; two total columns must be rejected")
	}
	kind, row, ok := revisedAuxColumnLayout(3, 1, []Sense{GreaterOrEqual})
	if !ok || len(kind) != 3 || len(row) != 3 {
		t.Fatalf("valid layout rejected: ok=%t kind=%v row=%v", ok, kind, row)
	}
}
