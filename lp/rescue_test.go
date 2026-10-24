package lp

import "testing"

// TestDenseRescueBackoff drives the Incremental-level backoff state machine:
// futile dense redos below the streak limit never disable rescues, hitting
// the limit starts a cooldown that doubles with each further futile
// diagnosis (capped), a helpful redo resets the streak, and the cooldown is
// consumed one solve at a time.
func TestDenseRescueBackoff(t *testing.T) {
	inc := &Incremental{}

	// Below the streak limit nothing is disabled.
	for i := 0; i < denseRescueFutilityLimit-1; i++ {
		inc.noteDenseRescueOutcome(true)
		if inc.denseRescueDisabled() {
			t.Fatalf("rescue disabled after only %d futile diagnoses", i+1)
		}
	}

	// Hitting the limit starts a cooldown of exactly base solves.
	inc.noteDenseRescueOutcome(true)
	for i := 0; i < denseRescueCooldownBase; i++ {
		if !inc.denseRescueDisabled() {
			t.Fatalf("cooldown tick %d: rescue unexpectedly re-enabled", i)
		}
	}
	if inc.denseRescueDisabled() {
		t.Fatal("cooldown did not expire after base ticks")
	}

	// The next futile diagnosis doubles the cooldown.
	inc.noteDenseRescueOutcome(true)
	if inc.denseRescueCooldown != denseRescueCooldownBase*2 {
		t.Fatalf("cooldown = %d, want %d", inc.denseRescueCooldown, denseRescueCooldownBase*2)
	}

	// A helpful redo resets the streak: one futile diagnosis afterwards must
	// not disable rescues again.
	inc.denseRescueCooldown = 0
	inc.noteDenseRescueOutcome(false)
	inc.noteDenseRescueOutcome(true)
	if inc.denseRescueDisabled() {
		t.Fatal("single futile diagnosis after a helpful redo should not disable rescues")
	}

	// A long futile streak saturates at the cap without overflowing.
	for i := 0; i < 40; i++ {
		inc.noteDenseRescueOutcome(true)
	}
	if inc.denseRescueCooldown != denseRescueCooldownMax {
		t.Fatalf("cooldown = %d, want cap %d", inc.denseRescueCooldown, denseRescueCooldownMax)
	}
}

// TestDenseRescueWasFutile pins the futility predicate: only a redo that
// both misses tolerance and fails to improve the residual by ≥10× counts.
func TestDenseRescueWasFutile(t *testing.T) {
	const tol = 1e-6
	cases := []struct {
		name             string
		sparseRel, dense float64
		futile           bool
	}{
		{"under tolerance", 1e-2, 1e-8, false},
		{"exactly helped past 10x", 1e-2, 1e-4, false},
		{"above tol, barely improved", 1e-2, 5e-3, true},
		{"above tol, byte-comparable", 1e-2, 1e-2, true},
		{"above tol but 100x better", 1e-1, 1e-3, false},
	}
	for _, tc := range cases {
		if got := denseRescueWasFutile(tc.sparseRel, tc.dense, tol); got != tc.futile {
			t.Errorf("%s: denseRescueWasFutile(%g, %g) = %v, want %v",
				tc.name, tc.sparseRel, tc.dense, got, tc.futile)
		}
	}
}
