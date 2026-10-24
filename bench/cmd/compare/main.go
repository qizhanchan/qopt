// Command compare diffs two result files written by the benchmark runners.
//
// The first file (-baseline) is treated as the reference (typically OR-Tools);
// the second (-candidate) is checked against it. Objective values, variable
// values, and statuses must agree within the configured tolerance.
package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"math"
	"os"
	"sort"
	"strings"

	"github.com/qizhanchan/qopt/bench/internal/problem"
)

type diffRow struct {
	name      string
	level     string // "ok", "warn", "fail"
	statusBL  string
	statusCND string
	objBL     float64
	objCND    float64
	objDelta  float64
	maxVarErr float64
	msBL      float64
	msCND     float64
	itersBL   int
	itersCND  int
	nodesBL   int
	nodesCND  int
	note      string
}

func main() {
	baselinePath := flag.String("baseline", "bench/results/ortools.json", "reference results file")
	candidatePath := flag.String("candidate", "bench/results/qopt.json", "candidate results file")
	tol := flag.Float64("tol", 1e-5, "absolute/relative tolerance for objective and values")
	problemsDir := flag.String("problems", "",
		"if set, verify each reported incumbent against its problem JSON; violations fail")
	flag.Parse()

	baseline, err := loadResults(*baselinePath)
	if err != nil {
		die("load baseline: %v", err)
	}
	candidate, err := loadResults(*candidatePath)
	if err != nil {
		die("load candidate: %v", err)
	}

	var problems map[string]*problem.Problem
	if *problemsDir != "" {
		probs, err := problem.LoadDir(*problemsDir)
		if err != nil {
			die("load problems: %v", err)
		}
		problems = make(map[string]*problem.Problem, len(probs))
		for _, p := range probs {
			problems[p.Name] = p
		}
	}

	rows := compare(baseline, candidate, *tol)
	if problems != nil {
		applyFeasibilityChecks(rows, baseline, candidate, problems, *tol)
	}

	printAccuracyTable(rows)
	fails, warns := countLevels(rows)
	fmt.Printf("Accuracy: %d ok, %d warn, %d fail  (tol=%g)\n",
		len(rows)-fails-warns, warns, fails, *tol)
	fmt.Println()

	printPerfTable(rows, baseline.Solver, candidate.Solver)
	printPerfAggregate(rows, baseline.Solver, candidate.Solver)

	fmt.Printf("baseline=%s (%s)  candidate=%s (%s)\n",
		baseline.Solver, baseline.Version, candidate.Solver, candidate.Version)

	if fails > 0 {
		os.Exit(1)
	}
}

func countLevels(rows []diffRow) (fails, warns int) {
	for _, r := range rows {
		switch r.level {
		case "fail":
			fails++
		case "warn":
			warns++
		}
	}
	return
}

func printAccuracyTable(rows []diffRow) {
	maxName := len("PROBLEM")
	for _, r := range rows {
		if len(r.name) > maxName {
			maxName = len(r.name)
		}
	}
	header := fmt.Sprintf("%-*s  %-6s  %-12s  %-12s  %-12s  %-12s  %-10s  %s",
		maxName, "PROBLEM", "LEVEL", "BASELINE", "CANDIDATE", "OBJ_BL", "OBJ_CND", "ABS_ERR", "NOTE")
	fmt.Println("== Accuracy ==")
	fmt.Println(header)
	fmt.Println(strings.Repeat("-", len(header)))
	for _, r := range rows {
		fmt.Printf("%-*s  %-6s  %-12s  %-12s  %-12g  %-12g  %-10.2e  %s\n",
			maxName, r.name, r.level, r.statusBL, r.statusCND,
			r.objBL, r.objCND, math.Max(math.Abs(r.objDelta), r.maxVarErr), r.note)
	}
	fmt.Println(strings.Repeat("-", len(header)))
}

func printPerfTable(rows []diffRow, blSolver, cndSolver string) {
	maxName := len("PROBLEM")
	for _, r := range rows {
		if len(r.name) > maxName {
			maxName = len(r.name)
		}
	}
	bl := strings.ToUpper(blSolver) + "_MS"
	cnd := strings.ToUpper(cndSolver) + "_MS"
	header := fmt.Sprintf("%-*s  %12s  %12s  %10s  %8s  %8s  %8s  %8s",
		maxName, "PROBLEM", bl, cnd, "SPEEDUP", "ITER_BL", "ITER_CND", "NODE_BL", "NODE_CND")
	fmt.Println("== Performance (median per-solve wall time) ==")
	fmt.Println(header)
	fmt.Println(strings.Repeat("-", len(header)))
	for _, r := range rows {
		speedup := "—"
		if r.msCND > 0 {
			speedup = fmt.Sprintf("%.2fx", r.msBL/r.msCND)
		}
		fmt.Printf("%-*s  %12.3f  %12.3f  %10s  %8d  %8d  %8d  %8d\n",
			maxName, r.name, r.msBL, r.msCND, speedup,
			r.itersBL, r.itersCND, r.nodesBL, r.nodesCND)
	}
	fmt.Println(strings.Repeat("-", len(header)))
}

func printPerfAggregate(rows []diffRow, blSolver, cndSolver string) {
	type agg struct {
		count int
		blMS  float64
		cndMS float64
	}
	var lp, mip, all agg
	for _, r := range rows {
		all.count++
		all.blMS += r.msBL
		all.cndMS += r.msCND
		bucket := &lp
		// Best-effort classifier: MIP if either solver reported B&B nodes >0,
		// or the name signals a MIP via the local naming convention.
		if r.nodesBL > 0 || r.nodesCND > 0 || strings.HasPrefix(r.name, "mip_") {
			bucket = &mip
		}
		bucket.count++
		bucket.blMS += r.msBL
		bucket.cndMS += r.msCND
	}
	printRow := func(label string, a agg) {
		if a.count == 0 {
			return
		}
		speedup := "—"
		if a.cndMS > 0 {
			speedup = fmt.Sprintf("%.2fx", a.blMS/a.cndMS)
		}
		fmt.Printf("  %-8s  n=%-3d  %s=%9.3fms   %s=%9.3fms   speedup=%s\n",
			label, a.count,
			strings.ToUpper(blSolver), a.blMS,
			strings.ToUpper(cndSolver), a.cndMS,
			speedup)
	}
	fmt.Println("== Aggregate (total median wall time across the suite) ==")
	printRow("LP", lp)
	printRow("MIP", mip)
	printRow("ALL", all)
	fmt.Println()
}

func compare(baseline, candidate *problem.Results, tol float64) []diffRow {
	byName := make(map[string]problem.Item, len(candidate.Items))
	for _, it := range candidate.Items {
		byName[it.Name] = it
	}
	seen := make(map[string]struct{}, len(baseline.Items))

	rows := make([]diffRow, 0, len(baseline.Items))
	for _, bl := range baseline.Items {
		seen[bl.Name] = struct{}{}
		cnd, ok := byName[bl.Name]
		if !ok {
			rows = append(rows, diffRow{
				name:      bl.Name,
				level:     "fail",
				statusBL:  bl.Status,
				statusCND: "<missing>",
				msBL:      bl.ElapsedMS,
				itersBL:   bl.Iterations,
				nodesBL:   bl.Nodes,
				note:      "candidate missing this problem",
			})
			continue
		}
		rows = append(rows, diffOne(bl, cnd, tol))
	}
	// Surface extra problems only present in the candidate.
	for _, it := range candidate.Items {
		if _, ok := seen[it.Name]; ok {
			continue
		}
		rows = append(rows, diffRow{
			name:      it.Name,
			level:     "warn",
			statusBL:  "<missing>",
			statusCND: it.Status,
			objCND:    it.Objective,
			msCND:     it.ElapsedMS,
			itersCND:  it.Iterations,
			nodesCND:  it.Nodes,
			note:      "baseline has no entry for this problem",
		})
	}
	sort.Slice(rows, func(i, j int) bool { return rows[i].name < rows[j].name })
	return rows
}

// equivalentStatuses captures known cases where two solvers legitimately label
// the same situation differently. These pairs are reported as "warn" rather
// than "fail" so the suite stays useful in the presence of solver quirks.
var equivalentStatuses = []map[string]bool{
	// GLOP (the default LP backend in OR-Tools) reports "infeasible" for
	// unbounded LPs because it does not differentiate the two on the primal
	// side. q-opt detects unboundedness explicitly via the simplex ratio
	// test, so the labels diverge while the underlying truth is the same.
	{"infeasible": true, "unbounded": true},
}

func statusesEquivalent(a, b string) bool {
	if a == b {
		return true
	}
	for _, group := range equivalentStatuses {
		if group[a] && group[b] {
			return true
		}
	}
	return false
}

// feasViolation describes the worst single constraint / bound / integrality
// breach in a reported incumbent. It is computed lazily from the original
// problem JSON; if -problems isn't passed, this analysis is skipped.
type feasViolation struct {
	kind    string  // "constraint", "bound", "integrality"
	subject string  // constraint or var name
	amount  float64 // magnitude of the breach
}

func (v feasViolation) String() string {
	return fmt.Sprintf("%s %q breach=%.3g", v.kind, v.subject, v.amount)
}

// verifyIncumbent walks every constraint, bound, and integrality requirement of
// `prob` and returns the largest violation observed for `values`. Returns nil
// (no violation) if all are within `tol`. Missing variables are treated as 0.
func verifyIncumbent(prob *problem.Problem, values map[string]float64, tol float64) *feasViolation {
	var worst *feasViolation
	track := func(v feasViolation) {
		if worst == nil || v.amount > worst.amount {
			cp := v
			worst = &cp
		}
	}
	for _, c := range prob.Constraints {
		var lhs float64
		for name, coef := range c.Coeffs {
			lhs += coef * values[name]
		}
		var breach float64
		switch c.Sense {
		case "<=":
			breach = lhs - c.RHS
		case ">=":
			breach = c.RHS - lhs
		case "=":
			breach = math.Abs(lhs - c.RHS)
		}
		if breach > tol {
			track(feasViolation{kind: "constraint", subject: c.Name, amount: breach})
		}
	}
	for _, v := range prob.Vars {
		val := values[v.Name]
		if v.Lower != nil && val < *v.Lower-tol {
			track(feasViolation{kind: "bound", subject: v.Name + "_lower",
				amount: *v.Lower - val})
		}
		if v.Upper != nil && val > *v.Upper+tol {
			track(feasViolation{kind: "bound", subject: v.Name + "_upper",
				amount: val - *v.Upper})
		}
		if v.Type == "integer" || v.Type == "binary" {
			if d := math.Abs(val - math.Round(val)); d > tol {
				track(feasViolation{kind: "integrality", subject: v.Name, amount: d})
			}
		}
	}
	return worst
}

// applyFeasibilityChecks tightens row verdicts when an incumbent isn't really
// feasible. If only one side has the bug, the row's verdict moves to "fail"
// regardless of where it stood before — solving the wrong problem is the
// strongest possible signal we can surface.
func applyFeasibilityChecks(rows []diffRow, baseline, candidate *problem.Results,
	problems map[string]*problem.Problem, tol float64) {
	blByName := indexItems(baseline.Items)
	cndByName := indexItems(candidate.Items)
	for i := range rows {
		r := &rows[i]
		prob, ok := problems[r.name]
		if !ok {
			continue
		}
		blViol := violationFor(prob, blByName[r.name], tol)
		cndViol := violationFor(prob, cndByName[r.name], tol)
		if blViol == nil && cndViol == nil {
			continue
		}
		extra := []string{}
		if blViol != nil {
			extra = append(extra, "baseline incumbent INFEASIBLE: "+blViol.String())
		}
		if cndViol != nil {
			extra = append(extra, "candidate incumbent INFEASIBLE: "+cndViol.String())
		}
		r.level = "fail"
		if r.note != "" {
			r.note = strings.Join(append([]string{r.note}, extra...), "; ")
		} else {
			r.note = strings.Join(extra, "; ")
		}
	}
}

func violationFor(prob *problem.Problem, it *problem.Item, tol float64) *feasViolation {
	if it == nil || it.Values == nil {
		return nil
	}
	return verifyIncumbent(prob, it.Values, tol)
}

func indexItems(items []problem.Item) map[string]*problem.Item {
	out := make(map[string]*problem.Item, len(items))
	for i := range items {
		out[items[i].Name] = &items[i]
	}
	return out
}

// limitedStatuses are non-optimal terminations that mean "stopped early".
// They imply an incumbent objective may or may not be present, but the solver
// did NOT prove optimality. Comparing two limited solvers is informational,
// not a correctness check.
var limitedStatuses = map[string]bool{
	"time_limit":      true,
	"iteration_limit": true,
	"node_limit":      true,
	"gap_limit":       true,
	"feasible":        true, // OR-Tools FEASIBLE: incumbent without optimality proof
}

func isLimited(s string) bool { return limitedStatuses[s] }
func hasIncumbent(it problem.Item) bool {
	return it.Status == "optimal" || it.Status == "feasible" ||
		(isLimited(it.Status) && it.Values != nil)
}

func diffOne(bl, cnd problem.Item, tol float64) diffRow {
	row := diffRow{
		name:      bl.Name,
		statusBL:  bl.Status,
		statusCND: cnd.Status,
		objBL:     bl.Objective,
		objCND:    cnd.Objective,
		msBL:      bl.ElapsedMS,
		msCND:     cnd.ElapsedMS,
		itersBL:   bl.Iterations,
		itersCND:  cnd.Iterations,
		nodesBL:   bl.Nodes,
		nodesCND:  cnd.Nodes,
	}

	if bl.Status != cnd.Status {
		if statusesEquivalent(bl.Status, cnd.Status) {
			row.level = "warn"
			row.note = "status mismatch within known solver equivalence (e.g. GLOP unbounded≈infeasible)"
			return row
		}
		// optimal vs limited: not a correctness bug — candidate just didn't
		// prove optimality within the budget. If it found a feasible solution
		// we still check whether its objective is consistent with the proven
		// bound; otherwise we record it as DNF without failing.
		if (bl.Status == "optimal" && isLimited(cnd.Status)) ||
			(cnd.Status == "optimal" && isLimited(bl.Status)) {
			return diffEarlyStop(bl, cnd, row, tol)
		}
		// Both stopped early but reported it differently (e.g. OR-Tools
		// "feasible" vs q-opt "time_limit"). Same situation, different labels.
		if isLimited(bl.Status) && isLimited(cnd.Status) {
			row.objDelta = cnd.Objective - bl.Objective
			row.level = "warn"
			if hasIncumbent(bl) && hasIncumbent(cnd) {
				row.note = fmt.Sprintf("both stopped early (%s vs %s); incumbents differ by %.3g",
					bl.Status, cnd.Status, math.Abs(row.objDelta))
			} else {
				row.note = fmt.Sprintf("both stopped early (%s vs %s)", bl.Status, cnd.Status)
			}
			return row
		}
		row.level = "fail"
		row.note = fmt.Sprintf("status mismatch: baseline=%s candidate=%s", bl.Status, cnd.Status)
		return row
	}

	if isLimited(bl.Status) {
		// Both ran out of budget on the same problem — only meaningful test is
		// whether their incumbents agree, and we accept either's score.
		row.objDelta = cnd.Objective - bl.Objective
		if hasIncumbent(bl) && hasIncumbent(cnd) && !within(bl.Objective, cnd.Objective, tol) {
			row.level = "warn"
			row.note = fmt.Sprintf("both stopped early; incumbents differ by %.3g", math.Abs(row.objDelta))
		} else {
			row.level = "warn"
			row.note = "both stopped early (no optimality proof from either side)"
		}
		return row
	}

	if bl.Status != "optimal" {
		// Non-optimal statuses agree; nothing else to check (no values).
		row.level = "ok"
		return row
	}

	row.objDelta = cnd.Objective - bl.Objective
	if !within(bl.Objective, cnd.Objective, tol) {
		row.level = "fail"
		row.note = fmt.Sprintf("objective mismatch: |Δ|=%.3g > tol=%g", math.Abs(row.objDelta), tol)
		return row
	}

	// Value comparison is informational: LPs with degenerate optima have
	// multiple equivalent solutions, so we only fail when the constraints
	// or objective contributions would differ. Reporting max var error keeps
	// it visible in the table without forcing a hard fail.
	row.maxVarErr = maxValueDelta(bl.Values, cnd.Values)
	if row.maxVarErr > tol {
		row.level = "warn"
		row.note = fmt.Sprintf("variable values differ (max |Δ|=%.3g); objectives still match — likely degenerate optimum",
			row.maxVarErr)
		return row
	}

	row.level = "ok"
	return row
}

// diffEarlyStop handles the asymmetric case where one side proved optimality
// and the other was stopped by a budget. We rate it "warn" — the candidate
// solver didn't prove optimal, but that's not a correctness fault. If the
// non-optimal side has a feasible incumbent, we additionally compare its
// objective: equality means it "got the right answer but didn't certify it",
// inequality means it's lagging behind.
func diffEarlyStop(bl, cnd problem.Item, row diffRow, tol float64) diffRow {
	row.level = "warn"
	provenSide, limitedSide := "baseline", "candidate"
	if cnd.Status == "optimal" {
		provenSide, limitedSide = "candidate", "baseline"
	}
	row.objDelta = cnd.Objective - bl.Objective
	switch {
	case !hasIncumbent(bl) || !hasIncumbent(cnd):
		row.note = fmt.Sprintf("%s proved optimum; %s stopped without incumbent", provenSide, limitedSide)
	case within(bl.Objective, cnd.Objective, tol):
		row.note = fmt.Sprintf("%s found the optimum but did not prove it", limitedSide)
	default:
		row.note = fmt.Sprintf("%s incumbent off by %.3g vs %s optimum",
			limitedSide, math.Abs(row.objDelta), provenSide)
	}
	return row
}

func maxValueDelta(a, b map[string]float64) float64 {
	var max float64
	keys := make(map[string]struct{}, len(a)+len(b))
	for k := range a {
		keys[k] = struct{}{}
	}
	for k := range b {
		keys[k] = struct{}{}
	}
	for k := range keys {
		d := math.Abs(a[k] - b[k])
		if d > max {
			max = d
		}
	}
	return max
}

func within(a, b, tol float64) bool {
	diff := math.Abs(a - b)
	if diff <= tol {
		return true
	}
	scale := math.Max(math.Abs(a), math.Abs(b))
	return diff <= tol*scale
}

func loadResults(path string) (*problem.Results, error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var r problem.Results
	if err := json.Unmarshal(raw, &r); err != nil {
		return nil, err
	}
	return &r, nil
}

func die(format string, args ...any) {
	fmt.Fprintf(os.Stderr, format+"\n", args...)
	os.Exit(2)
}
