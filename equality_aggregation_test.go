package qopt

import (
	"context"
	"math"
	"testing"
)

func TestContinuousEqualityAggregationObjectiveAndPostsolve(t *testing.T) {
	for _, backend := range []LPBackend{LPBackendRevised, LPBackendTableau} {
		t.Run(backend.String(), func(t *testing.T) {
			m := NewModel(Minimize)
			s := m.AddVar(VarSpec{Name: "s", Type: Continuous, Lower: Float64(0), Upper: Float64(4)})
			x := m.AddVar(VarSpec{Name: "x", Type: Integer, Lower: Float64(0), Upper: Float64(2)})
			if err := m.SetObjective(map[Var]float64{s: 1, x: 0.5}); err != nil {
				t.Fatal(err)
			}
			if err := m.AddConstraint(map[Var]float64{s: 1, x: 2}, Equal, 4); err != nil {
				t.Fatal(err)
			}

			res, err := m.Solve(context.Background(), Options{
				LPBackend:                     backend,
				AggregateContinuousEqualities: true,
			})
			if err != nil {
				t.Fatalf("solve: %v", err)
			}
			if res.Status != Optimal {
				t.Fatalf("status = %s, want optimal", res.Status)
			}
			if math.Abs(res.Objective-1) > 1e-7 || math.Abs(res.Values[s]) > 1e-7 || math.Abs(res.Values[x]-2) > 1e-7 {
				t.Fatalf("objective=%g values=%v, want objective=1 s=0 x=2", res.Objective, res.Values)
			}
			if math.Abs(res.Values[s]+2*res.Values[x]-4) > 1e-7 {
				t.Fatalf("postsolve violates original equality: values=%v", res.Values)
			}
			if res.Stats == nil || res.Stats.Presolve.AggregatedColumns != 1 {
				t.Fatalf("presolve stats = %+v, want one aggregation", res.Stats)
			}
		})
	}
}

func TestContinuousEqualityAggregationTranslatesNegativeCoefficientBounds(t *testing.T) {
	m := NewModel(Minimize)
	s := m.AddVar(VarSpec{Name: "s", Type: Continuous, Lower: Float64(1), Upper: Float64(5)})
	x := m.AddVar(VarSpec{Name: "x", Type: Integer, Lower: Float64(0), Upper: Float64(3)})
	y := m.AddVar(VarSpec{Name: "y", Type: Integer, Lower: Float64(0), Upper: Float64(3)})
	_ = m.SetObjective(map[Var]float64{x: 1, y: 1})
	_ = m.AddConstraint(map[Var]float64{s: -1, x: 1, y: 1}, Equal, 0)

	lp := canonicalizeWithOptions(m, nil, true, true)
	stats := lp.presolveStats(m)
	if stats.AggregatedColumns != 1 || stats.PresolvedColumns != 2 || stats.PresolvedRows != 2 {
		t.Fatalf("unexpected presolved shape: %+v", stats)
	}

	res, err := m.Solve(context.Background(), Options{AggregateContinuousEqualities: true})
	if err != nil {
		t.Fatal(err)
	}
	if res.Status != Optimal || math.Abs(res.Objective-1) > 1e-7 || math.Abs(res.Values[s]-1) > 1e-7 || math.Abs(res.Values[x]+res.Values[y]-1) > 1e-7 {
		t.Fatalf("status=%s objective=%g values=%v", res.Status, res.Objective, res.Values)
	}
}

func TestContinuousEqualityAggregationRequiresSingleOccurrence(t *testing.T) {
	m := NewModel(Minimize)
	s := m.AddVar(VarSpec{Name: "s", Type: Continuous, Lower: Float64(0), Upper: Float64(4)})
	x := m.AddVar(VarSpec{Name: "x", Type: Integer, Lower: Float64(0), Upper: Float64(2)})
	_ = m.AddConstraint(map[Var]float64{s: 1, x: 1}, Equal, 2)
	_ = m.AddConstraint(map[Var]float64{s: 1}, LessOrEqual, 3)
	if plans := continuousEqualityAggregations(m); len(plans) != 0 {
		t.Fatalf("aggregated a variable used by multiple rows: %+v", plans)
	}
}

func TestContinuousEqualityAggregationComposesWithRootCuts(t *testing.T) {
	m := NewModel(Maximize)
	s := m.AddVar(VarSpec{Name: "s", Type: Continuous, Lower: Float64(0), Upper: Float64(1)})
	x := m.AddVar(VarSpec{Name: "x", Type: Integer, Lower: Float64(0), Upper: Float64(1)})
	y := m.AddVar(VarSpec{Name: "y", Type: Integer, Lower: Float64(0), Upper: Float64(1)})
	must(t, m.SetObjective(map[Var]float64{x: 1, y: 1}))
	must(t, m.AddConstraint(map[Var]float64{s: 1, x: 2, y: 2}, Equal, 3))

	res, err := m.Solve(context.Background(), Options{
		LPBackend:                     LPBackendRevised,
		AggregateContinuousEqualities: true,
		MaxCutRounds:                  1,
	})
	if err != nil || res.Status != Optimal {
		t.Fatalf("solve: status=%s err=%v", res.Status, err)
	}
	if math.Abs(res.Objective-1) > 1e-7 || math.Abs(res.Values[s]-1) > 1e-7 || math.Abs(res.Values[x]+res.Values[y]-1) > 1e-7 {
		t.Fatalf("objective=%g values=%v, want objective=1 s=1 x+y=1", res.Objective, res.Values)
	}
	if res.Stats == nil || res.Stats.Presolve.AggregatedColumns != 1 {
		t.Fatalf("presolve stats = %+v, want one aggregation with root cuts enabled", res.Stats)
	}
	if res.Cuts == 0 {
		t.Fatal("cuts = 0, want an aggregated root GMI cut")
	}
}

func TestContinuousEqualityAggregationTimtab1Shape(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping MIPLIB structural integration test in short mode")
	}
	p := loadProblem(t, "bench/problems/miplib/timtab1.json")
	m := modelFromProblem(t, p)
	lp := canonicalizeWithOptions(m, nil, true, true)
	stats := lp.presolveStats(m)
	if stats.AggregatedColumns != 171 {
		t.Fatalf("aggregated columns = %d, want SCIP-matched 171; stats=%+v", stats.AggregatedColumns, stats)
	}
	if stats.PresolvedColumns > 226 {
		t.Fatalf("presolved columns = %d, want at least 43%% reduction from 397", stats.PresolvedColumns)
	}
	if stats.PresolvedRows > 2*stats.OriginalRows {
		t.Fatalf("two-sided bound expansion exceeded its structural cap: %+v", stats)
	}
}
