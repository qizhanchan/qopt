package qopt

import (
	"fmt"
	"math"
	"sort"
	"strings"
)

const rootCutPoolAgeLimit = 10

type rootCutPoolEntry struct {
	row           Constraint
	family        string
	creationRound int
	active        bool
	age           int
	expired       bool
}

type rootCutPool struct {
	entries     []rootCutPoolEntry
	bySignature map[string]int
}

type rootCutPoolAddStats struct {
	added      int
	duplicates int
}

type rootCutPoolScanStats struct {
	tested           int
	violated         int
	retested         int
	reusedCandidates int
	agedOut          int
}

func newRootCutPool() *rootCutPool {
	return &rootCutPool{bySignature: make(map[string]int)}
}

func (p *rootCutPool) addCandidates(candidates []cutCandidate) rootCutPoolAddStats {
	var stats rootCutPoolAddStats
	if p == nil {
		return stats
	}
	for i := range candidates {
		if candidates[i].state != cutCandidateGenerated {
			continue
		}
		signature := normalizedCutPoolSignature(candidates[i].row)
		if signature == "" {
			candidates[i].state = cutCandidateRejectedRowprepInvalid
			continue
		}
		if existing, exists := p.bySignature[signature]; exists && !p.entries[existing].expired {
			candidates[i].state = cutCandidateRejectedPoolDuplicate
			stats.duplicates++
			continue
		}
		if existing, exists := p.bySignature[signature]; exists {
			p.entries[existing] = rootCutPoolEntry{
				row:           cloneConstraint(candidates[i].row),
				family:        candidates[i].family,
				creationRound: candidates[i].creationRound,
			}
			candidates[i].poolIndex = existing
			stats.added++
			continue
		}
		idx := len(p.entries)
		p.entries = append(p.entries, rootCutPoolEntry{
			row:           cloneConstraint(candidates[i].row),
			family:        candidates[i].family,
			creationRound: candidates[i].creationRound,
		})
		p.bySignature[signature] = idx
		candidates[i].poolIndex = idx
		stats.added++
	}
	return stats
}

func (p *rootCutPool) separationCandidates(values []float64, nvars, round int, minEfficacy float64) ([]cutCandidate, rootCutPoolScanStats) {
	var stats rootCutPoolScanStats
	if p == nil {
		return nil, stats
	}
	candidates := make([]cutCandidate, 0, len(p.entries))
	for idx := range p.entries {
		entry := &p.entries[idx]
		if entry.active || entry.expired {
			continue
		}
		stats.tested++
		if entry.creationRound < round {
			stats.retested++
		}
		candidate := makeCutCandidates([]Constraint{entry.row}, values, nvars, round)[0]
		candidate.family = entry.family
		candidate.creationRound = entry.creationRound
		candidate.poolIndex = idx
		candidate.source = "pool"
		if candidate.efficacy < minEfficacy {
			entry.age++
			if entry.age > rootCutPoolAgeLimit {
				entry.expired = true
				stats.agedOut++
			}
			continue
		}
		entry.age = 0
		stats.violated++
		if entry.creationRound < round {
			stats.reusedCandidates++
		}
		candidates = append(candidates, candidate)
	}
	return candidates, stats
}

func (p *rootCutPool) setActive(candidates []cutCandidate, selected []int, active bool) {
	if p == nil {
		return
	}
	for _, candidateIdx := range selected {
		if candidateIdx < 0 || candidateIdx >= len(candidates) {
			continue
		}
		poolIdx := candidates[candidateIdx].poolIndex
		if poolIdx >= 0 && poolIdx < len(p.entries) {
			p.entries[poolIdx].active = active
		}
	}
}

// setActiveIndex toggles one pool entry's active flag directly. The
// persistent root loop uses it when row aging empties an active slot: the
// entry returns to the retestable pool population (SCIP's row-to-pool
// demotion) instead of staying counted as an LP row.
func (p *rootCutPool) setActiveIndex(idx int, active bool) {
	if p == nil || idx < 0 || idx >= len(p.entries) {
		return
	}
	p.entries[idx].active = active
}

func (p *rootCutPool) size() int {
	if p == nil {
		return 0
	}
	n := 0
	for i := range p.entries {
		if !p.entries[i].expired {
			n++
		}
	}
	return n
}

func (p *rootCutPool) activeCount() int {
	if p == nil {
		return 0
	}
	n := 0
	for i := range p.entries {
		if p.entries[i].active && !p.entries[i].expired {
			n++
		}
	}
	return n
}

// normalizedCutPoolSignature follows SCIP's cut-pool identity convention at
// the level q-opt currently needs: orient the inequality as <= and scale the
// largest absolute coefficient to one before hashing coefficients and side.
// This deduplicates positive scalar multiples without merging numerically
// distinct rows.
func normalizedCutPoolSignature(row Constraint) string {
	if row.Sense == Equal || len(row.Coeffs) == 0 {
		return ""
	}
	orientation := 1.0
	if row.Sense == GreaterOrEqual {
		orientation = -1
	} else if row.Sense != LessOrEqual {
		return ""
	}
	maxAbs := 0.0
	for _, coef := range row.Coeffs {
		if math.IsNaN(coef) || math.IsInf(coef, 0) {
			return ""
		}
		maxAbs = math.Max(maxAbs, math.Abs(coef))
	}
	if maxAbs == 0 || math.IsNaN(row.RHS) || math.IsInf(row.RHS, 0) {
		return ""
	}
	pairs := make([]string, 0, len(row.Coeffs))
	for v, coef := range row.Coeffs {
		normalized := orientation * coef / maxAbs
		if normalized == 0 {
			continue
		}
		pairs = append(pairs, fmt.Sprintf("%d:%s", v, floatKey(normalized)))
	}
	sort.Strings(pairs)
	return fmt.Sprintf("%s|%s", floatKey(orientation*row.RHS/maxAbs), strings.Join(pairs, ","))
}
