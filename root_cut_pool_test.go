package qopt

import "testing"

func TestRootCutPoolDeduplicatesScaledRows(t *testing.T) {
	x := Var(0)
	candidates := makeCutCandidates([]Constraint{
		{Name: "gomory_a", Coeffs: map[Var]float64{x: 1}, Sense: GreaterOrEqual, RHS: 1},
		{Name: "gomory_scaled", Coeffs: map[Var]float64{x: 2}, Sense: GreaterOrEqual, RHS: 2},
	}, []float64{0}, 1, 0)

	pool := newRootCutPool()
	stats := pool.addCandidates(candidates)
	if stats.added != 1 || stats.duplicates != 1 {
		t.Fatalf("add stats = %+v, want one added and one duplicate", stats)
	}
	if pool.size() != 1 {
		t.Fatalf("pool size = %d, want 1", pool.size())
	}
	if candidates[1].state != cutCandidateRejectedPoolDuplicate {
		t.Fatalf("scaled row state = %q, want pool duplicate", candidates[1].state)
	}
}

func TestRootCutPoolRetainsUnselectedViolatedRows(t *testing.T) {
	x, y := Var(0), Var(1)
	candidates := makeCutCandidates([]Constraint{
		{Name: "gomory_x", Coeffs: map[Var]float64{x: 1}, Sense: GreaterOrEqual, RHS: 5},
		{Name: "mir_y", Coeffs: map[Var]float64{y: 1}, Sense: GreaterOrEqual, RHS: 3},
	}, []float64{0, 0}, 2, 0)

	pool := newRootCutPool()
	pool.addCandidates(candidates)
	first, firstStats := pool.separationCandidates([]float64{0, 0}, 2, 0, cutRowprepMinEfficacy)
	selected := selectCutCandidates(first, 1)
	pool.setActive(first, selected, true)
	if firstStats.violated != 2 || len(selected) != 1 || pool.activeCount() != 1 {
		t.Fatalf("first scan = %+v selected=%v active=%d", firstStats, selected, pool.activeCount())
	}

	second, secondStats := pool.separationCandidates([]float64{0, 0}, 2, 1, cutRowprepMinEfficacy)
	if secondStats.tested != 1 || secondStats.retested != 1 || secondStats.violated != 1 || secondStats.reusedCandidates != 1 || len(second) != 1 {
		t.Fatalf("second scan = %+v candidates=%+v, want retained unselected row", secondStats, second)
	}
	if second[0].row.Name != "mir_y" || second[0].creationRound != 0 || second[0].source != "pool" {
		t.Fatalf("retained candidate metadata = %+v", second[0])
	}
}

func TestRootCutPoolCanReviveExpiredRows(t *testing.T) {
	x := Var(0)
	cut := Constraint{Name: "gomory_x", Coeffs: map[Var]float64{x: 1}, Sense: LessOrEqual, RHS: 1}
	pool := newRootCutPool()
	pool.addCandidates(makeCutCandidates([]Constraint{cut}, []float64{0}, 1, 0))
	for round := 0; round <= rootCutPoolAgeLimit; round++ {
		pool.separationCandidates([]float64{0}, 1, round, cutRowprepMinEfficacy)
	}

	readded := makeCutCandidates([]Constraint{cut}, []float64{2}, 1, rootCutPoolAgeLimit+1)
	stats := pool.addCandidates(readded)
	if stats.added != 1 || stats.duplicates != 0 || pool.size() != 1 {
		t.Fatalf("revive stats = %+v pool size=%d", stats, pool.size())
	}
	got, scan := pool.separationCandidates([]float64{2}, 1, rootCutPoolAgeLimit+1, cutRowprepMinEfficacy)
	if len(got) != 1 || scan.violated != 1 {
		t.Fatalf("revived scan = %+v candidates=%+v", scan, got)
	}
}

func TestRootCutPoolAgesOutInactiveUnviolatedRows(t *testing.T) {
	x := Var(0)
	candidates := makeCutCandidates([]Constraint{
		{Name: "gomory_x", Coeffs: map[Var]float64{x: 1}, Sense: LessOrEqual, RHS: 1},
	}, []float64{0}, 1, 0)

	pool := newRootCutPool()
	pool.addCandidates(candidates)
	for round := 0; round <= rootCutPoolAgeLimit; round++ {
		got, stats := pool.separationCandidates([]float64{0}, 1, round, cutRowprepMinEfficacy)
		if len(got) != 0 {
			t.Fatalf("round %d returned nonviolated cut: %+v", round, got)
		}
		if round < rootCutPoolAgeLimit && stats.agedOut != 0 {
			t.Fatalf("round %d aged out early: %+v", round, stats)
		}
		if round == rootCutPoolAgeLimit && stats.agedOut != 1 {
			t.Fatalf("final scan = %+v, want one aged-out cut", stats)
		}
	}
	if pool.size() != 0 {
		t.Fatalf("pool size = %d, want expired row excluded", pool.size())
	}
}
