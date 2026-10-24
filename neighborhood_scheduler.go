package qopt

import (
	"math"
	"time"
)

const (
	neighborhoodLowGapThreshold = 0.25
	neighborhoodExploration     = 0.20
	neighborhoodUntriedBonus    = 0.75
	neighborhoodFailurePenalty  = 0.75
	neighborhoodEffortFloor     = 100 * time.Millisecond
	neighborhoodMinReward       = 0.05
	neighborhoodExpensiveFail   = 500 * time.Millisecond
)

type neighborhoodKind uint8

const (
	neighborhoodProximity neighborhoodKind = iota
	neighborhoodRINS
	neighborhoodLocalBranching
	neighborhoodKindCount
)

func (k neighborhoodKind) name() string {
	switch k {
	case neighborhoodProximity:
		return "proximity"
	case neighborhoodRINS:
		return "rins"
	case neighborhoodLocalBranching:
		return "local_branching"
	default:
		return "unknown"
	}
}

type neighborhoodCandidate struct {
	kind  neighborhoodKind
	prior float64
}

type neighborhoodArm struct {
	calls            int
	successes        int
	reward           float64
	consecutiveFails int
}

type neighborhoodScheduler struct {
	arms             [neighborhoodKindCount]neighborhoodArm
	totalCalls       int
	consecutiveFails int
	exhausted        bool
}

func neighborhoodPrior(kind neighborhoodKind, primalGap float64) float64 {
	if primalGap < neighborhoodLowGapThreshold {
		switch kind {
		case neighborhoodLocalBranching:
			return 3
		case neighborhoodProximity:
			return 1
		case neighborhoodRINS:
			return 0
		}
	}
	switch kind {
	case neighborhoodProximity:
		return 3
	case neighborhoodRINS:
		return 2
	case neighborhoodLocalBranching:
		return 1
	default:
		return 0
	}
}

func (s *neighborhoodScheduler) allow(kind neighborhoodKind, primalGap float64) bool {
	if s.exhausted {
		return false
	}
	if kind == neighborhoodLocalBranching {
		return s.arms[kind].calls == 0
	}
	if kind == neighborhoodProximity && primalGap < neighborhoodLowGapThreshold {
		return s.arms[kind].calls == 0
	}
	return true
}

func (s *neighborhoodScheduler) selectCandidate(candidates []neighborhoodCandidate) (neighborhoodKind, bool) {
	if len(candidates) == 0 {
		return 0, false
	}
	best := candidates[0]
	bestScore := math.Inf(-1)
	for _, candidate := range candidates {
		arm := s.arms[candidate.kind]
		score := candidate.prior
		if arm.calls == 0 {
			score += neighborhoodUntriedBonus
		} else {
			meanReward := arm.reward / float64(arm.calls)
			explore := neighborhoodExploration * math.Sqrt(math.Log(float64(s.totalCalls)+1)/float64(arm.calls))
			score += meanReward + explore - neighborhoodFailurePenalty*float64(arm.consecutiveFails)
		}
		if score > bestScore || (score == bestScore && candidate.kind < best.kind) {
			best = candidate
			bestScore = score
		}
	}
	return best.kind, true
}

func (s *neighborhoodScheduler) observe(kind neighborhoodKind, reward float64, success bool, elapsed time.Duration, primalGap float64) {
	arm := &s.arms[kind]
	arm.calls++
	s.totalCalls++
	arm.reward += reward
	if success {
		arm.successes++
		arm.consecutiveFails = 0
		s.consecutiveFails = 0
	} else {
		arm.consecutiveFails++
		s.consecutiveFails++
	}
	if success {
		if reward < neighborhoodMinReward || primalGap < neighborhoodLowGapThreshold {
			s.exhausted = true
		}
	} else if elapsed >= neighborhoodExpensiveFail || s.consecutiveFails >= 2 {
		s.exhausted = true
	}
}

func neighborhoodReward(direction Direction, before, after, lpBound float64, elapsed time.Duration, tol float64) float64 {
	improvement := objectiveImprovement(direction, before, after)
	if improvement <= tol {
		return 0
	}
	remaining := objectiveImprovement(direction, before, lpBound)
	if remaining < improvement {
		remaining = improvement
	}
	quality := improvement / math.Max(remaining, tol)
	effort := elapsed
	if effort < neighborhoodEffortFloor {
		effort = neighborhoodEffortFloor
	}
	reward := quality / effort.Seconds()
	if reward > 1 {
		return 1
	}
	return reward
}

func objectiveImprovement(direction Direction, before, after float64) float64 {
	if direction == Maximize {
		return after - before
	}
	return before - after
}
