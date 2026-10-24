package qopt

import "math"

// domainData is the immutable propagation graph of a model. It is cached on
// Model and shared by every B&B node: rows are deterministic sorted slices and
// varRows maps a changed variable to exactly the rows whose activities need to
// be recomputed.
type domainData struct {
	rows     [][]coefEntry
	varRows  [][]int
	integral []bool
}

type boundReasonKind uint8

const (
	reasonNone boundReasonKind = iota
	reasonRowLower
	reasonRowUpper
)

// boundReason records the latest row inference responsible for a variable
// bound. Milestone 1 heuristics use the row identity for deterministic repair;
// future conflict analysis can extend this with antecedent bound literals.
type boundReason struct {
	Row      int
	Kind     boundReasonKind
	Previous float64
	Bound    float64
}

type rowActivity struct {
	min, max       float64
	minInf, maxInf int
}

// domainState is one mutable domain over immutable domainData.
type domainState struct {
	m     *Model
	data  *domainData
	lower []float64
	upper []float64
	tol   float64

	lowerReason []boundReason
	upperReason []boundReason
	queue       []int
	queueHead   int
	queued      []bool
}

func (m *Model) propagationData() *domainData {
	if m.domainData != nil && len(m.domainData.integral) == len(m.vars) && len(m.domainData.rows) == len(m.constraints) {
		return m.domainData
	}
	d := &domainData{
		rows:     m.sortedRows(),
		varRows:  make([][]int, len(m.vars)),
		integral: make([]bool, len(m.vars)),
	}
	for i, v := range m.vars {
		d.integral[i] = v.Type == Integer || v.Type == Binary
	}
	for r, row := range d.rows {
		for _, e := range row {
			if e.v >= 0 && e.v < len(d.varRows) && e.coef != 0 {
				d.varRows[e.v] = append(d.varRows[e.v], r)
			}
		}
	}
	m.domainData = d
	return d
}

func newDomainState(m *Model, lower, upper []float64, tol float64) *domainState {
	if tol <= 0 {
		tol = 1e-9
	}
	return &domainState{
		m:           m,
		data:        m.propagationData(),
		lower:       lower,
		upper:       upper,
		tol:         tol,
		lowerReason: make([]boundReason, len(m.vars)),
		upperReason: make([]boundReason, len(m.vars)),
		queued:      make([]bool, len(m.constraints)),
	}
}

func (d *domainState) enqueueRow(row int) {
	if row < 0 || row >= len(d.queued) || d.queued[row] {
		return
	}
	d.queued[row] = true
	d.queue = append(d.queue, row)
}

func (d *domainState) enqueueVarRows(v int) {
	for _, row := range d.data.varRows[v] {
		d.enqueueRow(row)
	}
}

func (d *domainState) activity(row int) rowActivity {
	var a rowActivity
	for _, e := range d.data.rows[row] {
		v, c := e.v, e.coef
		switch {
		case c > 0:
			if math.IsInf(d.lower[v], -1) {
				a.minInf++
			} else {
				a.min += c * d.lower[v]
			}
			if math.IsInf(d.upper[v], 1) {
				a.maxInf++
			} else {
				a.max += c * d.upper[v]
			}
		case c < 0:
			if math.IsInf(d.upper[v], 1) {
				a.minInf++
			} else {
				a.min += c * d.upper[v]
			}
			if math.IsInf(d.lower[v], -1) {
				a.maxInf++
			} else {
				a.max += c * d.lower[v]
			}
		}
	}
	return a
}

func (d *domainState) tightenLower(v int, b float64, row int) bool {
	if d.data.integral[v] {
		b = math.Ceil(b - d.tol)
	}
	if b <= d.lower[v]+d.tol {
		return true
	}
	old := d.lower[v]
	d.lower[v] = b
	d.lowerReason[v] = boundReason{Row: row, Kind: reasonRowLower, Previous: old, Bound: b}
	if d.lower[v] > d.upper[v]+d.tol {
		return false
	}
	d.enqueueVarRows(v)
	return true
}

func (d *domainState) tightenUpper(v int, b float64, row int) bool {
	if d.data.integral[v] {
		b = math.Floor(b + d.tol)
	}
	if b >= d.upper[v]-d.tol {
		return true
	}
	old := d.upper[v]
	d.upper[v] = b
	d.upperReason[v] = boundReason{Row: row, Kind: reasonRowUpper, Previous: old, Bound: b}
	if d.lower[v] > d.upper[v]+d.tol {
		return false
	}
	d.enqueueVarRows(v)
	return true
}

func (d *domainState) rowInfeasible(row int, a rowActivity) bool {
	c := d.m.constraints[row]
	switch c.Sense {
	case LessOrEqual:
		return a.minInf == 0 && a.min > c.RHS+d.tol
	case GreaterOrEqual:
		return a.maxInf == 0 && a.max < c.RHS-d.tol
	case Equal:
		return (a.minInf == 0 && a.min > c.RHS+d.tol) ||
			(a.maxInf == 0 && a.max < c.RHS-d.tol)
	}
	return false
}

func (d *domainState) forceMin(row int) bool {
	for _, e := range d.data.rows[row] {
		if e.coef > 0 {
			if !d.tightenUpper(e.v, d.lower[e.v], row) {
				return false
			}
		} else if e.coef < 0 {
			if !d.tightenLower(e.v, d.upper[e.v], row) {
				return false
			}
		}
	}
	return true
}

func (d *domainState) forceMax(row int) bool {
	for _, e := range d.data.rows[row] {
		if e.coef > 0 {
			if !d.tightenLower(e.v, d.upper[e.v], row) {
				return false
			}
		} else if e.coef < 0 {
			if !d.tightenUpper(e.v, d.lower[e.v], row) {
				return false
			}
		}
	}
	return true
}

func (d *domainState) propagateRow(row int) bool {
	cons := d.m.constraints[row]
	entries := d.data.rows[row]
	a := d.activity(row)
	if d.rowInfeasible(row, a) {
		debugf("propagate: row %q infeasible (min=%g max=%g rhs=%g)", cons.Name, a.min, a.max, cons.RHS)
		return false
	}

	// Exact forcing only: tolerance-based equality can remove a narrow but
	// valuable continuous interval.
	if (cons.Sense == LessOrEqual || cons.Sense == Equal) && a.minInf == 0 && a.min == cons.RHS {
		return d.forceMin(row)
	}
	if (cons.Sense == GreaterOrEqual || cons.Sense == Equal) && a.maxInf == 0 && a.max == cons.RHS {
		return d.forceMax(row)
	}

	for _, e := range entries {
		v, coef := e.v, e.coef
		var minWO, maxWO float64
		var minWOInf, maxWOInf int
		if coef > 0 {
			if math.IsInf(d.lower[v], -1) {
				minWOInf, minWO = a.minInf-1, a.min
			} else {
				minWOInf, minWO = a.minInf, a.min-coef*d.lower[v]
			}
			if math.IsInf(d.upper[v], 1) {
				maxWOInf, maxWO = a.maxInf-1, a.max
			} else {
				maxWOInf, maxWO = a.maxInf, a.max-coef*d.upper[v]
			}
		} else {
			if math.IsInf(d.upper[v], 1) {
				minWOInf, minWO = a.minInf-1, a.min
			} else {
				minWOInf, minWO = a.minInf, a.min-coef*d.upper[v]
			}
			if math.IsInf(d.lower[v], -1) {
				maxWOInf, maxWO = a.maxInf-1, a.max
			} else {
				maxWOInf, maxWO = a.maxInf, a.max-coef*d.lower[v]
			}
		}

		deriveLE := func() bool {
			if minWOInf > 0 {
				return true
			}
			b := (cons.RHS - minWO) / coef
			if coef > 0 {
				return d.tightenUpper(v, b, row)
			}
			return d.tightenLower(v, b, row)
		}
		deriveGE := func() bool {
			if maxWOInf > 0 {
				return true
			}
			b := (cons.RHS - maxWO) / coef
			if coef > 0 {
				return d.tightenLower(v, b, row)
			}
			return d.tightenUpper(v, b, row)
		}

		switch cons.Sense {
		case LessOrEqual:
			if !deriveLE() {
				return false
			}
		case GreaterOrEqual:
			if !deriveGE() {
				return false
			}
		case Equal:
			if !deriveLE() || !deriveGE() {
				return false
			}
		}
	}
	return true
}

// propagate processes a deterministic row event queue. maxPasses preserves
// tightenBounds' historical work cap: at most maxPasses full-row equivalents
// are visited, while bound changes only enqueue affected rows.
func (d *domainState) propagate(maxPasses int) bool {
	if maxPasses <= 0 || len(d.data.rows) == 0 {
		return true
	}
	for row := range d.data.rows {
		d.enqueueRow(row)
	}
	maxVisits := maxPasses * len(d.data.rows)
	visits := 0
	for d.queueHead < len(d.queue) && visits < maxVisits {
		row := d.queue[d.queueHead]
		d.queueHead++
		d.queued[row] = false
		visits++
		if !d.propagateRow(row) {
			return false
		}
	}
	return true
}
