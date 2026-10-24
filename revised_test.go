package qopt

import (
	"context"
	"math"
	"testing"
)

// runWithBackend solves m with the given backend choice and returns the
// result. It does not assert any particular status; callers compare against
// the tableau-backend result for cross-validation.
func runWithBackend(t *testing.T, m *Model, backend LPBackend, opts Options) Result {
	t.Helper()
	opts.LPBackend = backend
	res, err := m.Solve(context.Background(), opts)
	if err != nil {
		t.Fatalf("backend %s: solve error: %v", backend, err)
	}
	return res
}

func TestRevisedMatchesTableauOnLP(t *testing.T) {
	cases := []struct {
		name  string
		build func() *Model
	}{
		{"simple-max", func() *Model {
			m := NewModel(Maximize)
			x := m.AddVar(VarSpec{Name: "x"})
			y := m.AddVar(VarSpec{Name: "y"})
			must(t, m.SetObjective(map[Var]float64{x: 3, y: 2}))
			must(t, m.AddConstraint(map[Var]float64{x: 1, y: 1}, LessOrEqual, 4))
			must(t, m.AddConstraint(map[Var]float64{x: 1}, LessOrEqual, 2))
			must(t, m.AddConstraint(map[Var]float64{y: 1}, LessOrEqual, 3))
			return m
		}},
		{"equality", func() *Model {
			m := NewModel(Minimize)
			x := m.AddVar(VarSpec{Name: "x"})
			y := m.AddVar(VarSpec{Name: "y"})
			must(t, m.SetObjective(map[Var]float64{x: 1, y: 1}))
			must(t, m.AddConstraint(map[Var]float64{x: 1, y: 1}, Equal, 5))
			return m
		}},
		{"ge-min", func() *Model {
			m := NewModel(Minimize)
			x := m.AddVar(VarSpec{Name: "x"})
			must(t, m.SetObjective(map[Var]float64{x: 1}))
			must(t, m.AddConstraint(map[Var]float64{x: 1}, GreaterOrEqual, 3))
			return m
		}},
		{"finite-bounds", func() *Model {
			m := NewModel(Minimize)
			x := m.AddVar(VarSpec{Name: "x", Lower: Float64(2), Upper: Float64(5)})
			must(t, m.SetObjective(map[Var]float64{x: 1}))
			return m
		}},
		{"mixed-bounds-upper-active", func() *Model {
			m := NewModel(Maximize)
			x := m.AddVar(VarSpec{Name: "x", Upper: Float64(10)})
			y := m.AddVar(VarSpec{Name: "y", Upper: Float64(7)})
			must(t, m.SetObjective(map[Var]float64{x: 2, y: 3}))
			must(t, m.AddConstraint(map[Var]float64{x: 1, y: 1}, LessOrEqual, 100))
			return m
		}},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			base := runWithBackend(t, tc.build(), LPBackendTableau, Options{})
			rev := runWithBackend(t, tc.build(), LPBackendRevised, Options{})
			if base.Status != rev.Status {
				t.Fatalf("status mismatch: tableau=%s revised=%s", base.Status, rev.Status)
			}
			if math.Abs(base.Objective-rev.Objective) > 1e-6 {
				t.Fatalf("objective: tableau=%.12g revised=%.12g", base.Objective, rev.Objective)
			}
			if len(base.Values) != len(rev.Values) {
				t.Fatalf("value count: tableau=%d revised=%d", len(base.Values), len(rev.Values))
			}
		})
	}
}

func TestRevisedSparseLUMatchesDenseLU(t *testing.T) {
	cases := []struct {
		name  string
		build func() *Model
	}{
		{"simple-max", func() *Model {
			m := NewModel(Maximize)
			x := m.AddVar(VarSpec{Name: "x"})
			y := m.AddVar(VarSpec{Name: "y"})
			must(t, m.SetObjective(map[Var]float64{x: 3, y: 2}))
			must(t, m.AddConstraint(map[Var]float64{x: 1, y: 1}, LessOrEqual, 4))
			must(t, m.AddConstraint(map[Var]float64{x: 1}, LessOrEqual, 2))
			must(t, m.AddConstraint(map[Var]float64{y: 1}, LessOrEqual, 3))
			return m
		}},
		{"binary-knapsack", func() *Model {
			m := NewModel(Maximize)
			x := m.AddVar(VarSpec{Name: "x", Type: Binary})
			y := m.AddVar(VarSpec{Name: "y", Type: Binary})
			z := m.AddVar(VarSpec{Name: "z", Type: Binary})
			must(t, m.SetObjective(map[Var]float64{x: 6, y: 5, z: 8}))
			must(t, m.AddConstraint(map[Var]float64{x: 4, y: 3, z: 5}, LessOrEqual, 8))
			return m
		}},
		{"facility-mip", func() *Model {
			m := NewModel(Maximize)
			x := m.AddVar(VarSpec{Name: "x", Type: Integer})
			y := m.AddVar(VarSpec{Name: "y", Type: Integer})
			must(t, m.SetObjective(map[Var]float64{x: 5, y: 4}))
			must(t, m.AddConstraint(map[Var]float64{x: 6, y: 4}, LessOrEqual, 24))
			must(t, m.AddConstraint(map[Var]float64{x: 1, y: 2}, LessOrEqual, 6))
			return m
		}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			denseRes := runWithBackend(t, tc.build(), LPBackendRevised, Options{RevisedDenseLU: true})
			sparseRes := runWithBackend(t, tc.build(), LPBackendRevised, Options{})
			if denseRes.Status != sparseRes.Status {
				t.Fatalf("status: dense=%s sparse=%s", denseRes.Status, sparseRes.Status)
			}
			if math.Abs(denseRes.Objective-sparseRes.Objective) > 1e-6 {
				t.Fatalf("objective: dense=%.12g sparse=%.12g", denseRes.Objective, sparseRes.Objective)
			}
			if len(denseRes.Values) != len(sparseRes.Values) {
				t.Fatalf("values length: dense=%d sparse=%d", len(denseRes.Values), len(sparseRes.Values))
			}
		})
	}
}

// TestRevisedAntiCyclingBeale exercises the Bland's-rule fallback in
// runSimplex. Beale's example is the textbook LP on which Dantzig pricing
// cycles through degenerate pivots and never terminates. With a tight
// iteration cap, a cycling simplex would return IterationLimit; the anti-
// cycling fallback must instead reach Optimal on both LU variants.
func TestRevisedAntiCyclingBeale(t *testing.T) {
	build := func() *Model {
		m := NewModel(Minimize)
		x4 := m.AddVar(VarSpec{Name: "x4"})
		x5 := m.AddVar(VarSpec{Name: "x5"})
		x6 := m.AddVar(VarSpec{Name: "x6"})
		x7 := m.AddVar(VarSpec{Name: "x7"})
		must(t, m.SetObjective(map[Var]float64{x4: -0.75, x5: 150, x6: -0.02, x7: 6}))
		must(t, m.AddConstraint(map[Var]float64{x4: 0.25, x5: -60, x6: -0.04, x7: 9}, LessOrEqual, 0))
		must(t, m.AddConstraint(map[Var]float64{x4: 0.5, x5: -90, x6: -0.02, x7: 3}, LessOrEqual, 0))
		must(t, m.AddConstraint(map[Var]float64{x6: 1}, LessOrEqual, 1))
		return m
	}
	opts := Options{MaxIterations: 500, RevisedDenseLU: true}
	dense := runWithBackend(t, build(), LPBackendRevised, opts)
	sparse := runWithBackend(t, build(), LPBackendRevised, Options{MaxIterations: 500})
	if dense.Status != Optimal {
		t.Fatalf("dense LU: status=%s (anti-cycling failed: likely IterationLimit from a degenerate cycle)", dense.Status)
	}
	if sparse.Status != Optimal {
		t.Fatalf("sparse LU: status=%s (anti-cycling failed)", sparse.Status)
	}
	// Beale's optimum is z = -0.05.
	closeTo(t, dense.Objective, -0.05)
	closeTo(t, sparse.Objective, -0.05)
}

// TestRevisedPersistFactorMatchesRefactor checks that reusing the parent's
// factorization across B&B nodes (RevisedPersistFactor) produces exactly the
// same optima as the refactor-every-node path. A divergence would mean the
// reused factor or the drift guard is wrong.
func TestRevisedPersistFactorMatchesRefactor(t *testing.T) {
	cases := []struct {
		name  string
		build func() *Model
	}{
		{"integer-production", func() *Model {
			m := NewModel(Maximize)
			x := m.AddVar(VarSpec{Name: "x", Type: Integer})
			y := m.AddVar(VarSpec{Name: "y", Type: Integer})
			must(t, m.SetObjective(map[Var]float64{x: 5, y: 4}))
			must(t, m.AddConstraint(map[Var]float64{x: 6, y: 4}, LessOrEqual, 24))
			must(t, m.AddConstraint(map[Var]float64{x: 1, y: 2}, LessOrEqual, 6))
			return m
		}},
		{"binary-knapsack", func() *Model {
			m := NewModel(Maximize)
			x := m.AddVar(VarSpec{Name: "x", Type: Binary})
			y := m.AddVar(VarSpec{Name: "y", Type: Binary})
			z := m.AddVar(VarSpec{Name: "z", Type: Binary})
			must(t, m.SetObjective(map[Var]float64{x: 6, y: 5, z: 8}))
			must(t, m.AddConstraint(map[Var]float64{x: 4, y: 3, z: 5}, LessOrEqual, 8))
			return m
		}},
	}
	for _, tc := range cases {
		for _, sel := range []MIPNodeSelection{MIPDepthFirst, MIPBestBound, MIPBestEstimate} {
			t.Run(tc.name, func(t *testing.T) {
				base := Options{LPBackend: LPBackendRevised, MIPNodeSelection: sel}
				off := solveOK(t, tc.build(), base)
				on := base
				on.RevisedPersistFactor = true
				got := solveOK(t, tc.build(), on)
				if math.Abs(off.Objective-got.Objective) > 1e-7 {
					t.Fatalf("%s/%v: persist=%.12g refactor=%.12g", tc.name, sel, got.Objective, off.Objective)
				}
			})
		}
	}
}

// TestRevisedFocusSessionMatchesStateless checks that the focus-node LP
// session (RevisedFocusSession) reaches the same MIP optima as the stateless
// per-node path across node-selection modes. The objective must agree exactly
// at MIP optimality; a divergence means an in-place reoptimize accepted an
// unsound basis.
func TestRevisedFocusSessionMatchesStateless(t *testing.T) {
	cases := []struct {
		name  string
		build func() *Model
	}{
		{"integer-production", func() *Model {
			m := NewModel(Maximize)
			x := m.AddVar(VarSpec{Name: "x", Type: Integer})
			y := m.AddVar(VarSpec{Name: "y", Type: Integer})
			must(t, m.SetObjective(map[Var]float64{x: 5, y: 4}))
			must(t, m.AddConstraint(map[Var]float64{x: 6, y: 4}, LessOrEqual, 24))
			must(t, m.AddConstraint(map[Var]float64{x: 1, y: 2}, LessOrEqual, 6))
			return m
		}},
		{"binary-knapsack", func() *Model {
			m := NewModel(Maximize)
			x := m.AddVar(VarSpec{Name: "x", Type: Binary})
			y := m.AddVar(VarSpec{Name: "y", Type: Binary})
			z := m.AddVar(VarSpec{Name: "z", Type: Binary})
			must(t, m.SetObjective(map[Var]float64{x: 6, y: 5, z: 8}))
			must(t, m.AddConstraint(map[Var]float64{x: 4, y: 3, z: 5}, LessOrEqual, 8))
			return m
		}},
		{"mixed-with-equality", func() *Model {
			m := NewModel(Minimize)
			x := m.AddVar(VarSpec{Name: "x", Type: Integer})
			y := m.AddVar(VarSpec{Name: "y", Type: Integer})
			w := m.AddVar(VarSpec{Name: "w"})
			must(t, m.SetObjective(map[Var]float64{x: 2, y: 3, w: 1}))
			must(t, m.AddConstraint(map[Var]float64{x: 1, y: 1, w: 1}, GreaterOrEqual, 7))
			must(t, m.AddConstraint(map[Var]float64{x: 3, y: -1}, Equal, 2))
			must(t, m.AddConstraint(map[Var]float64{w: 1}, LessOrEqual, 4))
			return m
		}},
	}
	for _, tc := range cases {
		for _, sel := range []MIPNodeSelection{MIPDepthFirst, MIPBestBound, MIPBestEstimate} {
			t.Run(tc.name, func(t *testing.T) {
				base := Options{LPBackend: LPBackendRevised, MIPNodeSelection: sel}
				off := solveOK(t, tc.build(), base)
				on := base
				on.RevisedFocusSession = true
				got := solveOK(t, tc.build(), on)
				if math.Abs(off.Objective-got.Objective) > 1e-7 {
					t.Fatalf("%s/%v: session=%.12g stateless=%.12g", tc.name, sel, got.Objective, off.Objective)
				}
			})
		}
	}
}

func TestRevisedInfeasible(t *testing.T) {
	m := NewModel(Minimize)
	x := m.AddVar(VarSpec{Name: "x"})
	must(t, m.SetObjective(map[Var]float64{x: 1}))
	must(t, m.AddConstraint(map[Var]float64{x: 1}, GreaterOrEqual, 2))
	must(t, m.AddConstraint(map[Var]float64{x: 1}, LessOrEqual, 1))

	res, err := m.Solve(context.Background(), Options{LPBackend: LPBackendRevised})
	if err != nil {
		t.Fatal(err)
	}
	if res.Status != Infeasible {
		t.Fatalf("status = %s, want infeasible", res.Status)
	}
}

func TestRevisedUnbounded(t *testing.T) {
	m := NewModel(Maximize)
	x := m.AddVar(VarSpec{Name: "x"})
	must(t, m.SetObjective(map[Var]float64{x: 1}))

	res, err := m.Solve(context.Background(), Options{LPBackend: LPBackendRevised})
	if err != nil {
		t.Fatal(err)
	}
	if res.Status != Unbounded {
		t.Fatalf("status = %s, want unbounded", res.Status)
	}
}

func TestRevisedMatchesTableauOnMIP(t *testing.T) {
	// MIP test using revised LP backend; with step-3+4 not yet wired,
	// the B&B re-solves LPs from scratch using the revised backend, so
	// this is really a stress test of revised on MANY closely-related LPs.
	cases := []struct {
		name  string
		build func() *Model
	}{
		{"binary-knapsack", func() *Model {
			m := NewModel(Maximize)
			x := m.AddVar(VarSpec{Name: "x", Type: Binary})
			y := m.AddVar(VarSpec{Name: "y", Type: Binary})
			z := m.AddVar(VarSpec{Name: "z", Type: Binary})
			must(t, m.SetObjective(map[Var]float64{x: 6, y: 5, z: 8}))
			must(t, m.AddConstraint(map[Var]float64{x: 4, y: 3, z: 5}, LessOrEqual, 8))
			return m
		}},
		{"integer-production", func() *Model {
			m := NewModel(Maximize)
			x := m.AddVar(VarSpec{Name: "x", Type: Integer})
			y := m.AddVar(VarSpec{Name: "y", Type: Integer})
			must(t, m.SetObjective(map[Var]float64{x: 5, y: 4}))
			must(t, m.AddConstraint(map[Var]float64{x: 6, y: 4}, LessOrEqual, 24))
			must(t, m.AddConstraint(map[Var]float64{x: 1, y: 2}, LessOrEqual, 6))
			return m
		}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			base := runWithBackend(t, tc.build(), LPBackendTableau, Options{})
			rev := runWithBackend(t, tc.build(), LPBackendRevised, Options{})
			if base.Status != rev.Status {
				t.Fatalf("status: tableau=%s revised=%s", base.Status, rev.Status)
			}
			if math.Abs(base.Objective-rev.Objective) > 1e-6 {
				t.Fatalf("objective: tableau=%.12g revised=%.12g", base.Objective, rev.Objective)
			}
		})
	}
}
