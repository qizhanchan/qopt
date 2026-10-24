// detsnap solves every problem in a directory under a deterministic budget
// (fixed MaxNodes, no TimeLimit, no wall-clock-dependent options) and prints
// full-precision results. Two runs on the same code must produce identical
// output, so diffing detsnap output before/after a refactor is a byte-exact
// regression check for "the refactor changed no numerical behavior".
//
// Usage:
//
//	go run ./bench/cmd/detsnap -problems bench/problems -max-nodes 0 > before.txt
//	go run ./bench/cmd/detsnap -problems bench/problems/miplib -max-nodes 300 >> before.txt
//	... refactor ...
//	diff before.txt after.txt
package main

import (
	"context"
	"flag"
	"fmt"
	"log"
	"sort"
	"strings"
	"sync"

	qopt "github.com/qizhanchan/qopt"
	"github.com/qizhanchan/qopt/bench/internal/problem"
)

func main() {
	problemsDir := flag.String("problems", "bench/problems", "directory of *.json problems")
	maxNodes := flag.Int("max-nodes", 300, "MIP node budget (0 = solve to completion)")
	skip := flag.String("skip", "", "comma-separated problem names to skip")
	parallel := flag.Int("parallel", 3, "problems solved concurrently (does not affect results)")
	nodeSelection := flag.String("node-selection", "dfs", "dfs, bestbound, or bestestimate")
	bestEstimateBestBoundFreq := flag.Int("best-estimate-best-bound-freq", 0, "best-estimate periodic best-bound frequency (0 disables)")
	branching := flag.String("branching", "mostfrac", "mostfrac, pseudocost, or reliability")
	useRINS := flag.Bool("rins", false, "enable RINS")
	rinsAdaptiveBudget := flag.Bool("rins-adaptive-budget", false, "enable adaptive RINS node budgets")
	useLocalBranching := flag.Bool("local-branching", false, "enable binary local branching")
	localBranchingRadius := flag.Int("local-branching-radius", 0, "local branching Hamming radius (0 = 18)")
	useProximitySearch := flag.Bool("proximity-search", false, "enable proximity search")
	proximityMinImprove := flag.Float64("proximity-min-improve", 0, "proximity minimum improvement fraction (0 = 0.02)")
	cutDepth := flag.Int("cut-depth", 0, "MaxCutDepth (0 disables per-node cuts)")
	cutRounds := flag.Int("cut-rounds", 0, "MaxCutRounds (0 disables root cuts)")
	flag.Parse()

	opts := qopt.Options{
		MaxNodes: *maxNodes, MaxCutDepth: *cutDepth, MaxCutRounds: *cutRounds,
		BestEstimateBestBoundFreq: *bestEstimateBestBoundFreq,
		UseRINS:                   *useRINS, RINSAdaptiveBudget: *rinsAdaptiveBudget,
		UseLocalBranching: *useLocalBranching, LocalBranchingRadius: *localBranchingRadius,
		UseProximitySearch:  *useProximitySearch,
		ProximityMinImprove: *proximityMinImprove,
	}
	switch *nodeSelection {
	case "dfs":
	case "bestbound":
		opts.MIPNodeSelection = qopt.MIPBestBound
	case "bestestimate":
		opts.MIPNodeSelection = qopt.MIPBestEstimate
	default:
		log.Fatalf("unknown -node-selection %q", *nodeSelection)
	}
	switch *branching {
	case "mostfrac":
	case "pseudocost":
		opts.MIPBranching = qopt.MIPBranchPseudocost
	case "reliability":
		opts.MIPBranching = qopt.MIPBranchReliability
	default:
		log.Fatalf("unknown -branching %q", *branching)
	}

	skipSet := map[string]bool{}
	for _, s := range strings.Split(*skip, ",") {
		if s != "" {
			skipSet[s] = true
		}
	}

	probs, err := problem.LoadDir(*problemsDir)
	if err != nil {
		log.Fatalf("load problems: %v", err)
	}
	sort.Slice(probs, func(i, j int) bool { return probs[i].Name < probs[j].Name })

	type line struct {
		idx int
		s   string
	}
	jobs := make(chan int)
	out := make(chan line)
	var wg sync.WaitGroup
	for w := 0; w < *parallel; w++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for idx := range jobs {
				out <- line{idx, solveOne(probs[idx], opts)}
			}
		}()
	}
	go func() {
		for i := range probs {
			if skipSet[probs[i].Name] {
				out <- line{i, fmt.Sprintf("%-32s SKIPPED", probs[i].Name)}
				continue
			}
			jobs <- i
		}
		close(jobs)
		wg.Wait()
		close(out)
	}()

	lines := make([]string, len(probs))
	for l := range out {
		lines[l.idx] = l.s
		fmt.Fprintln(flag.CommandLine.Output(), l.s)
	}
	for _, s := range lines {
		fmt.Println(s)
	}
}

func solveOne(p *problem.Problem, opts qopt.Options) string {
	m, err := buildModel(p)
	if err != nil {
		return fmt.Sprintf("%-32s ERROR %v", p.Name, err)
	}
	res, err := m.Solve(context.Background(), opts)
	if err != nil {
		return fmt.Sprintf("%-32s ERROR %v", p.Name, err)
	}
	return fmt.Sprintf("%-32s status=%-16s obj=%.17g bound=%.17g nodes=%d iters=%d",
		p.Name, res.Status, res.Objective, res.DualBound, res.Nodes, res.Iterations)
}

func buildModel(p *problem.Problem) (*qopt.Model, error) {
	var direction qopt.Direction
	switch p.Direction {
	case "min", "minimize":
		direction = qopt.Minimize
	case "max", "maximize":
		direction = qopt.Maximize
	default:
		return nil, fmt.Errorf("unknown direction %q", p.Direction)
	}

	m := qopt.NewModel(direction)
	varByName := make(map[string]qopt.Var, len(p.Vars))
	for _, v := range p.Vars {
		spec := qopt.VarSpec{Name: v.Name}
		switch v.Type {
		case "", "continuous":
			spec.Type = qopt.Continuous
		case "integer", "int":
			spec.Type = qopt.Integer
		case "binary", "bool":
			spec.Type = qopt.Binary
		default:
			return nil, fmt.Errorf("var %q: unknown type %q", v.Name, v.Type)
		}
		if v.Lower != nil {
			spec.Lower = qopt.Float64(*v.Lower)
		}
		if v.Upper != nil {
			spec.Upper = qopt.Float64(*v.Upper)
		}
		varByName[v.Name] = m.AddVar(spec)
	}

	obj := make(map[qopt.Var]float64, len(p.Objective))
	for name, c := range p.Objective {
		v, ok := varByName[name]
		if !ok {
			return nil, fmt.Errorf("objective references unknown var %q", name)
		}
		obj[v] = c
	}
	if err := m.SetObjective(obj); err != nil {
		return nil, err
	}

	for _, c := range p.Constraints {
		coeffs := make(map[qopt.Var]float64, len(c.Coeffs))
		for name, v := range c.Coeffs {
			ref, ok := varByName[name]
			if !ok {
				return nil, fmt.Errorf("constraint %q references unknown var %q", c.Name, name)
			}
			coeffs[ref] = v
		}
		var sense qopt.Sense
		switch c.Sense {
		case "<=", "le":
			sense = qopt.LessOrEqual
		case ">=", "ge":
			sense = qopt.GreaterOrEqual
		case "=", "==", "eq":
			sense = qopt.Equal
		default:
			return nil, fmt.Errorf("unknown sense %q", c.Sense)
		}
		if err := m.AddNamedConstraint(c.Name, coeffs, sense, c.RHS); err != nil {
			return nil, err
		}
	}
	return m, nil
}
