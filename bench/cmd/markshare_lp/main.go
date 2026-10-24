package main

import (
	"context"
	"fmt"
	"time"

	qopt "github.com/qizhanchan/qopt"
	"github.com/qizhanchan/qopt/bench/internal/problem"
)

func buildModel(p *problem.Problem, relax bool) *qopt.Model {
	dir := qopt.Minimize
	if p.Direction == "max" { dir = qopt.Maximize }
	m := qopt.NewModel(dir)
	idx := map[string]qopt.Var{}
	for _, v := range p.Vars {
		t := qopt.Continuous
		if !relax {
			switch v.Type {
			case "integer": t = qopt.Integer
			case "binary":  t = qopt.Binary
			}
		}
		spec := qopt.VarSpec{Name: v.Name, Type: t}
		// Forward optional bounds as pointers; nil → AddVar default.
		if v.Lower != nil { spec.Lower = qopt.Float64(*v.Lower) }
		if v.Upper != nil { spec.Upper = qopt.Float64(*v.Upper) }
		idx[v.Name] = m.AddVar(spec)
	}
	obj := map[qopt.Var]float64{}
	for n, c := range p.Objective { obj[idx[n]] = c }
	m.SetObjective(obj)
	for _, c := range p.Constraints {
		co := map[qopt.Var]float64{}
		for n, v := range c.Coeffs { co[idx[n]] = v }
		var s qopt.Sense
		switch c.Sense {
		case "<=": s = qopt.LessOrEqual
		case ">=": s = qopt.GreaterOrEqual
		case "=":  s = qopt.Equal
		}
		m.AddConstraint(co, s, c.RHS)
	}
	return m
}

func main() {
	for _, name := range []string{"markshare_4_0", "gen-ip054", "gen-ip002", "neos5"} {
		p, err := problem.Load("bench/problems/miplib/"+name+".json")
		if err != nil { fmt.Println(err); continue }
		// LP relaxation only
		lpM := buildModel(p, true)
		lpRes, _ := lpM.Solve(context.Background(), qopt.Options{TimeLimit: 30*time.Second})
		// Root + heavy Gomory + MIR, no branching
		mipM := buildModel(p, false)
		opts := qopt.Options{
			TimeLimit: 60*time.Second,
			MaxCutRounds: 500,
			CutTimeLimit: 60*time.Second,
			UseMIRCuts: true,
			MaxNodes: 1, // halt right after root LP after cuts
		}
		mr, _ := mipM.Solve(context.Background(), opts)
		fmt.Printf("%-20s LP=%-12g  Cuts(root)=%-4d  IP_at_node1=%-12g  Status=%v\n",
			name, lpRes.Objective, mr.Cuts, mr.Objective, mr.Status)
	}
}
