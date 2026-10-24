package qopt

import (
	"container/heap"
	"context"
	"math"
	"testing"
	"time"
)

// knapsackModel builds a small integer knapsack used by the dual-bound tests.
func knapsackModel() *Model {
	m := NewModel(Maximize)
	x := m.AddVar(VarSpec{Name: "x", Type: Integer})
	y := m.AddVar(VarSpec{Name: "y", Type: Integer})
	_ = m.SetObjective(map[Var]float64{x: 5, y: 4})
	_ = m.AddConstraint(map[Var]float64{x: 6, y: 4}, LessOrEqual, 24)
	_ = m.AddConstraint(map[Var]float64{x: 1, y: 2}, LessOrEqual, 6)
	return m
}

// TestDualBound_ProvenOptimalGapZero checks that when a MIP is solved to
// proven optimality (tree exhausted), the dual bound equals the incumbent and
// the reported gap is zero — the "gap == 0 ⇒ proven" contract.
func TestDualBound_ProvenOptimalGapZero(t *testing.T) {
	for _, sel := range []MIPNodeSelection{MIPDepthFirst, MIPBestBound, MIPBestEstimate} {
		res, err := knapsackModel().Solve(context.Background(), Options{MIPNodeSelection: sel})
		if err != nil {
			t.Fatalf("%v: %v", sel, err)
		}
		if res.Status != Optimal {
			t.Fatalf("%v: status=%s, want optimal", sel, res.Status)
		}
		if math.Abs(res.DualBound-res.Objective) > 1e-7 {
			t.Fatalf("%v: proven optimum but bound %.12g != objective %.12g", sel, res.DualBound, res.Objective)
		}
		if res.RelativeGap > 1e-9 {
			t.Fatalf("%v: proven optimum but gap=%g, want 0", sel, res.RelativeGap)
		}
	}
}

// TestDualBound_ValidWhenTruncated checks that when the search is cut short by
// a node limit, the dual bound is a *valid* bound (≥ incumbent for Maximize)
// and the gap is finite and non-negative — i.e. it brackets the true optimum.
func TestDualBound_ValidWhenTruncated(t *testing.T) {
	for _, sel := range []MIPNodeSelection{MIPDepthFirst, MIPBestBound, MIPBestEstimate} {
		res, err := knapsackModel().Solve(context.Background(), Options{MIPNodeSelection: sel, MaxNodes: 1})
		if err != nil {
			t.Fatalf("%v: %v", sel, err)
		}
		// With a 1-node cap the search may or may not have an incumbent, but if
		// it does the Maximize dual bound must not be below it (a valid upper
		// bound) and the gap must be finite and ≥ 0.
		if res.Values == nil {
			continue
		}
		if math.IsNaN(res.DualBound) || math.IsInf(res.DualBound, 0) {
			t.Fatalf("%v: incumbent present but dual bound is %g", sel, res.DualBound)
		}
		if res.DualBound < res.Objective-1e-7 {
			t.Fatalf("%v: Maximize dual bound %.12g below incumbent %.12g (invalid)", sel, res.DualBound, res.Objective)
		}
		if res.RelativeGap < 0 || math.IsInf(res.RelativeGap, 0) {
			t.Fatalf("%v: gap=%g, want finite ≥ 0", sel, res.RelativeGap)
		}
	}
}

func TestDualBound_BestBoundKeepsUnevaluatedSiblingAtNodeLimit(t *testing.T) {
	m := NewModel(Maximize)
	x := m.AddVar(VarSpec{Name: "x", Type: Integer})
	y := m.AddVar(VarSpec{Name: "y", Type: Integer})
	must(t, m.SetObjective(map[Var]float64{x: 5, y: 4}))
	must(t, m.AddConstraint(map[Var]float64{x: 6, y: 4}, LessOrEqual, 24))
	must(t, m.AddConstraint(map[Var]float64{x: 1, y: 2}, LessOrEqual, 6))

	// This deterministic cutoff lands while best-bound is evaluating a pair
	// of children. The unfinished sibling must retain the parent's LP bound.
	res, err := m.Solve(context.Background(), Options{
		MIPNodeSelection: MIPBestBound,
		MaxNodes:         4,
	})
	if err != nil {
		t.Fatal(err)
	}
	if res.Status != NodeLimit {
		t.Fatalf("status=%s, want node_limit", res.Status)
	}
	// The exact integer optimum is 20 (x=4,y=0). A maximize dual bound below
	// 20 would prove that an open subtree disappeared from the accounting.
	if res.DualBound < 20-1e-8 {
		t.Fatalf("invalid maximize dual bound %g below known optimum 20", res.DualBound)
	}
	if res.Values != nil && res.DualBound < res.Objective-1e-8 {
		t.Fatalf("dual bound %g below incumbent %g", res.DualBound, res.Objective)
	}
}

func TestBestEstimateHeapKeepsLPBoundAuthoritative(t *testing.T) {
	open := &mipBestBoundHeap{direction: Minimize, selection: MIPBestEstimate}
	heap.Init(open)
	heap.Push(open, mipCandidate{objective: 1, estimate: 10})
	heap.Push(open, mipCandidate{objective: 2, estimate: 5})

	got := heap.Pop(open).(mipCandidate)
	if got.objective != 2 {
		t.Fatalf("best-estimate popped objective %g, want the node with estimate 5", got.objective)
	}
	// The heuristic ordering must not replace the mathematically valid LP
	// bound used for proof accounting.
	heap.Push(open, got)
	if bound := heapDualBound(open, Minimize, false, 0); bound != 1 {
		t.Fatalf("dual bound = %g, want authoritative LP bound 1", bound)
	}
}

func TestBestEstimatePeriodicBestBoundSelection(t *testing.T) {
	open := &mipBestBoundHeap{direction: Minimize, selection: MIPBestEstimate}
	heap.Init(open)
	// Pure estimate prefers objective=2, estimate=5. The periodic proof step
	// must instead remove the node with the authoritative best LP bound.
	heap.Push(open, mipCandidate{objective: 1, estimate: 10})
	heap.Push(open, mipCandidate{objective: 2, estimate: 5})
	q := &bestBoundQueue{open: open, selected: 9}
	sc := &searchContext{
		ctx:   context.Background(),
		start: time.Now(),
		opts:  Options{BestEstimateBestBoundFreq: 10},
	}

	got, status, err := q.next(sc)
	if err != nil || status != nextBranch {
		t.Fatalf("next status=%v err=%v", status, err)
	}
	if got.objective != 1 {
		t.Fatalf("periodic best-bound selected objective %g, want 1", got.objective)
	}
}

func TestBestEstimateAdaptiveProofSwitch(t *testing.T) {
	open := &mipBestBoundHeap{direction: Minimize, selection: MIPBestEstimate}
	heap.Init(open)
	heap.Push(open, mipCandidate{objective: 1, estimate: 10})
	heap.Push(open, mipCandidate{objective: 2, estimate: 5})
	q := &bestBoundQueue{open: open}
	sc := &searchContext{
		ctx:           context.Background(),
		start:         time.Now(),
		work:          NewModel(Minimize),
		haveIncumbent: true,
		bestObj:       2.5,
		opts:          Options{BestEstimateProofGap: 0.65},
		stats:         &SolverStats{},
	}

	got, status, err := q.next(sc)
	if err != nil || status != nextBranch {
		t.Fatalf("next status=%v err=%v", status, err)
	}
	if got.objective != 1 || q.open.selection != MIPBestBound {
		t.Fatalf("selected objective=%g selection=%s, want best bound and permanent proof mode", got.objective, q.open.selection)
	}
	if sc.stats.Branching.BestEstimateProofSwitches != 1 || sc.stats.Branching.BestEstimateProofSwitchNode != 0 {
		t.Fatalf("switch telemetry = %+v", sc.stats.Branching)
	}
}

func TestMIPGapReturnsGapLimitAndRetainsBound(t *testing.T) {
	m := NewModel(Maximize)
	x := m.AddVar(VarSpec{Name: "x", Type: Integer})
	y := m.AddVar(VarSpec{Name: "y", Type: Integer})
	must(t, m.SetObjective(map[Var]float64{x: 5, y: 4}))
	must(t, m.AddConstraint(map[Var]float64{x: 6, y: 4}, LessOrEqual, 24))
	must(t, m.AddConstraint(map[Var]float64{x: 1, y: 2}, LessOrEqual, 6))

	for _, sel := range []MIPNodeSelection{MIPDepthFirst, MIPBestBound, MIPBestEstimate} {
		res, err := m.Solve(context.Background(), Options{MIPGap: 0.20, MIPNodeSelection: sel})
		if err != nil {
			t.Fatal(err)
		}
		if res.Status != GapLimit {
			t.Fatalf("%s: status=%s, want gap_limit", sel, res.Status)
		}
		if res.Values == nil {
			t.Fatalf("%s: gap-limited solve returned no incumbent", sel)
		}
		if res.DualBound <= res.Objective+1e-8 {
			t.Fatalf("%s: gap-limited solve collapsed bound %g to incumbent %g", sel, res.DualBound, res.Objective)
		}
		if res.RelativeGap <= 0 || res.RelativeGap > 0.20+1e-9 {
			t.Fatalf("%s: relative gap=%g, want (0, 0.20]", sel, res.RelativeGap)
		}
	}
}
