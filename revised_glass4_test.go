package qopt

import (
	"context"
	"encoding/json"
	"os"
	"testing"
	"time"
)

// jsonProblem is a minimal mirror of bench/internal/problem.Problem — the root
// package can't import that internal package, so we re-declare just the fields
// the MIPLIB JSON files use.
type jsonProblem struct {
	Name        string             `json:"name"`
	Direction   string             `json:"direction"`
	Vars        []jsonVar          `json:"vars"`
	Objective   map[string]float64 `json:"objective"`
	Constraints []jsonConstraint   `json:"constraints"`
}

type jsonVar struct {
	Name  string   `json:"name"`
	Type  string   `json:"type"`
	Lower *float64 `json:"lower,omitempty"`
	Upper *float64 `json:"upper,omitempty"`
}

type jsonConstraint struct {
	Name   string             `json:"name,omitempty"`
	Coeffs map[string]float64 `json:"coeffs"`
	Sense  string             `json:"sense"`
	RHS    float64            `json:"rhs"`
}

// modelFromProblem mirrors the bench runner's problem→Model conversion (see
// bench/cmd/qopt_bench/main.go toVarSpec/parseSense) closely enough to load a
// MIPLIB JSON instance inside a unit test.
func modelFromProblem(t *testing.T, p *jsonProblem) *Model {
	t.Helper()
	dir := Minimize
	if p.Direction == "max" || p.Direction == "maximize" {
		dir = Maximize
	}
	m := NewModel(dir)
	byName := make(map[string]Var, len(p.Vars))
	for _, v := range p.Vars {
		spec := VarSpec{Name: v.Name}
		switch v.Type {
		case "", "continuous":
			spec.Type = Continuous
		case "integer", "int":
			spec.Type = Integer
		case "binary", "bool":
			spec.Type = Binary
		default:
			t.Fatalf("var %q: unknown type %q", v.Name, v.Type)
		}
		// Forward optional bounds as pointers; nil → AddVar default
		// (0 / +Inf, or [0,1] for Binary). Preserves explicit upper:0.
		if v.Lower != nil {
			spec.Lower = Float64(*v.Lower)
		}
		if v.Upper != nil {
			spec.Upper = Float64(*v.Upper)
		}
		byName[v.Name] = m.AddVar(spec)
	}
	obj := make(map[Var]float64, len(p.Objective))
	for name, c := range p.Objective {
		obj[byName[name]] = c
	}
	if err := m.SetObjective(obj); err != nil {
		t.Fatalf("set objective: %v", err)
	}
	for _, c := range p.Constraints {
		var sense Sense
		switch c.Sense {
		case "<=", "le":
			sense = LessOrEqual
		case ">=", "ge":
			sense = GreaterOrEqual
		case "=", "==", "eq":
			sense = Equal
		default:
			t.Fatalf("constraint %q: unknown sense %q", c.Name, c.Sense)
		}
		coeffs := make(map[Var]float64, len(c.Coeffs))
		for name, v := range c.Coeffs {
			coeffs[byName[name]] = v
		}
		if err := m.AddConstraint(coeffs, sense, c.RHS); err != nil {
			t.Fatalf("add constraint %q: %v", c.Name, err)
		}
	}
	return m
}

func loadProblem(t *testing.T, path string) *jsonProblem {
	t.Helper()
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Skipf("problem file unavailable: %v", err)
	}
	var p jsonProblem
	if err := json.Unmarshal(raw, &p); err != nil {
		t.Fatalf("parse %s: %v", path, err)
	}
	return &p
}

// TestRevisedSparseLU_Glass4NoIterationLimit guards the sparse-LU → dense-LU
// fallback. glass4's ill-conditioned, degenerate LP used to steer the sparse LU
// into a non-terminating degenerate plateau (Bland's tie-breaks defeated by
// sparse-factor float noise), burning a single LP to MaxIterations and making
// the whole MIP report iteration_limit. The dense LU clears the same LP in tens
// of pivots, so the backend now pins a stalling solve to the dense LU. Without
// the fix this test reports IterationLimit; with it, glass4 behaves like the
// dense backend (time_limit, no incumbent in this tiny budget).
func TestRevisedSparseLU_Glass4NoIterationLimit(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping glass4 sparse-LU integration test in -short mode")
	}
	p := loadProblem(t, "bench/problems/miplib/glass4.json")
	opts := Options{
		LPBackend:     LPBackendRevised, // also the default now; explicit for clarity
		MIPBranching:  MIPBranchReliability,
		TimeLimit:     4 * time.Second,
		MaxIterations: 8000,
	}
	res, err := modelFromProblem(t, p).Solve(context.Background(), opts)
	if err != nil {
		t.Fatalf("solve: %v", err)
	}
	if res.Status == IterationLimit {
		t.Fatalf("glass4 under sparse-LU returned IterationLimit — the dense-LU stall fallback regressed")
	}
}
