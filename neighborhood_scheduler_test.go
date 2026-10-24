package qopt

import (
	"testing"
	"time"
)

func TestNeighborhoodSchedulerUsesGapAwarePriors(t *testing.T) {
	candidates := func(gap float64) []neighborhoodCandidate {
		return []neighborhoodCandidate{
			{kind: neighborhoodProximity, prior: neighborhoodPrior(neighborhoodProximity, gap)},
			{kind: neighborhoodRINS, prior: neighborhoodPrior(neighborhoodRINS, gap)},
			{kind: neighborhoodLocalBranching, prior: neighborhoodPrior(neighborhoodLocalBranching, gap)},
		}
	}

	var low neighborhoodScheduler
	got, ok := low.selectCandidate(candidates(0.10))
	if !ok || got != neighborhoodLocalBranching {
		t.Fatalf("low-gap selection=%s ok=%t, want local_branching", got.name(), ok)
	}

	var high neighborhoodScheduler
	got, ok = high.selectCandidate(candidates(0.80))
	if !ok || got != neighborhoodProximity {
		t.Fatalf("high-gap selection=%s ok=%t, want proximity", got.name(), ok)
	}
}

func TestNeighborhoodSchedulerExploitsSuccessAndExploresAfterFailure(t *testing.T) {
	var scheduler neighborhoodScheduler
	scheduler.observe(neighborhoodProximity, 1, true, 100*time.Millisecond, 0.80)
	candidates := []neighborhoodCandidate{
		{kind: neighborhoodProximity, prior: 3},
		{kind: neighborhoodRINS, prior: 2},
	}
	got, ok := scheduler.selectCandidate(candidates)
	if !ok || got != neighborhoodProximity {
		t.Fatalf("selection=%s ok=%t, want successful proximity", got.name(), ok)
	}

	var failed neighborhoodScheduler
	failed.observe(neighborhoodProximity, 0, false, 100*time.Millisecond, 0.80)
	got, ok = failed.selectCandidate(candidates)
	if !ok || got != neighborhoodRINS {
		t.Fatalf("post-failure selection=%s ok=%t, want untried RINS", got.name(), ok)
	}
}

func TestNeighborhoodSchedulerLimitsLowGapProximity(t *testing.T) {
	var scheduler neighborhoodScheduler
	if !scheduler.allow(neighborhoodProximity, 0.10) {
		t.Fatal("first low-gap proximity call should be allowed")
	}
	scheduler.observe(neighborhoodProximity, 0.1, true, 100*time.Millisecond, 0.30)
	if scheduler.allow(neighborhoodProximity, 0.10) {
		t.Fatal("repeated low-gap proximity call should be blocked")
	}
	if !scheduler.allow(neighborhoodProximity, 0.80) {
		t.Fatal("high-gap proximity should remain eligible")
	}
}

func TestNeighborhoodSchedulerRunsLocalBranchingOnce(t *testing.T) {
	var scheduler neighborhoodScheduler
	if !scheduler.allow(neighborhoodLocalBranching, 0.10) {
		t.Fatal("first local branching call should be allowed")
	}
	scheduler.observe(neighborhoodLocalBranching, 0.2, true, 100*time.Millisecond, 0.30)
	if scheduler.allow(neighborhoodLocalBranching, 0.10) {
		t.Fatal("portfolio should not repeat local branching around its own incumbent")
	}
}

func TestNeighborhoodSchedulerStopsOnLowRewardOrExpensiveFailure(t *testing.T) {
	var lowReward neighborhoodScheduler
	lowReward.observe(neighborhoodProximity, 0.02, true, time.Second, 0.80)
	if lowReward.allow(neighborhoodRINS, 0.80) {
		t.Fatal("low reward should exhaust the portfolio")
	}

	var expensiveFailure neighborhoodScheduler
	expensiveFailure.observe(neighborhoodProximity, 0, false, time.Second, 0.80)
	if expensiveFailure.allow(neighborhoodRINS, 0.80) {
		t.Fatal("expensive failure should exhaust the portfolio")
	}

	var cheapFailure neighborhoodScheduler
	cheapFailure.observe(neighborhoodProximity, 0, false, 10*time.Millisecond, 0.80)
	if !cheapFailure.allow(neighborhoodRINS, 0.80) {
		t.Fatal("cheap failure should allow another arm")
	}
	cheapFailure.observe(neighborhoodRINS, 0, false, 10*time.Millisecond, 0.80)
	if cheapFailure.allow(neighborhoodLocalBranching, 0.80) {
		t.Fatal("two consecutive failures should exhaust the portfolio")
	}
}

func TestNeighborhoodRewardAccountsForQualityAndTime(t *testing.T) {
	fast := neighborhoodReward(Minimize, 100, 90, 0, 250*time.Millisecond, 1e-8)
	slow := neighborhoodReward(Minimize, 100, 90, 0, time.Second, 1e-8)
	small := neighborhoodReward(Minimize, 100, 99, 0, 250*time.Millisecond, 1e-8)
	if !(fast > slow && fast > small && slow > 0) {
		t.Fatalf("unexpected rewards fast=%g slow=%g small=%g", fast, slow, small)
	}
	if got := neighborhoodReward(Maximize, 10, 9, 20, time.Second, 1e-8); got != 0 {
		t.Fatalf("non-improving maximize reward=%g, want 0", got)
	}
}
