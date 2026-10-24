// Command qopt_bench runs the shared benchmark problems through q-opt and
// writes the result file consumed by bench/cmd/compare.
package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"math"
	"os"
	"os/exec"
	"runtime/debug"
	"runtime/pprof"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"

	qopt "github.com/qizhanchan/qopt"
	"github.com/qizhanchan/qopt/bench/internal/problem"
)

// Per-node cut tuning knobs, set from flags in main and read in solveOne.
// Globals avoid threading two niche experiment parameters through the already
// long solveAll → solveRepeated → solveOne argument chain.
var (
	cutTuningCap                  int
	rootCutCap                    int
	cutTuningSkipValidationBelow  int
	persistFactor                 bool
	noCertifyInfeasible           bool
	focusSession                  bool
	useDualProofs                 bool
	pseudoObjPropagation          bool
	objectiveCutoffPropagation    bool
	reducedCostFixing             bool
	cutRowAgeLimit                int
	cutRowsDriveBranching         bool
	rootCutStallRounds            int
	rootCutRowAgeLimit            int
	useRINS                       bool
	rinsAdaptiveBudget            bool
	useLocalBranching             bool
	localBranchingRadius          int
	useProximitySearch            bool
	proximityMinImprove           float64
	disablePrimalHeuristics       bool
	repairAdaptiveCadence         bool
	detectStructure               bool
	aggregateContinuousEqualities bool
	bestEstimateBestBoundFreq     int
	bestEstimateProofGap          float64
	maxNodes                      int
	reliabilityThreshold          int
	reliabilityAdaptiveThreshold  bool
	reliabilityMaxProbes          int
	reliabilityPrimalBudget       int
	reliabilityProbeInference     bool
	useLPConflictLearning         bool
	lpConflictMaxLiterals         int
	lpConflictMinimizeProbes      int
)

func main() {
	problemsDir := flag.String("problems", "bench/problems", "directory of *.json problems")
	outPath := flag.String("out", "bench/results/qopt.json", "output results file")
	timeLimit := flag.Duration("time-limit", 30*time.Second, "per-problem time limit")
	repeat := flag.Int("repeat", 1, "solve each problem N times; record median wall time")
	warmup := flag.Int("warmup", 0, "warmup runs per problem (not counted in timing)")
	cutRounds := flag.Int("cut-rounds", 0, "Gomory cut rounds at MIP root (0 disables)")
	cutTimeLimit := flag.Duration("cut-time-limit", 5*time.Second, "total time budget for root cuts when cuts are enabled; 0 disables the separate cut budget")
	cutDepth := flag.Int("cut-depth", 0, "max B&B depth at which Gomory cuts are derived per node (0 disables)")
	cutDepthStride := flag.Int("cut-depth-stride", 1, "depth stride for per-node cuts: cut only at depths 0, stride, 2*stride, ... (1 = every depth, CBC's stall default is 5)")
	useMIR := flag.Bool("mir", false, "enable Marchand-Wolsey row-MIR cuts alongside Gomory at the same firing points")
	nodeSelectionFlag := flag.String("node-selection", "dfs", "MIP node selection strategy: dfs, best-bound, or best-estimate")
	bestEstimateBestBoundFreqFlag := flag.Int("best-estimate-best-bound-freq", 0, "under best-estimate, select the best LP-bound node every Nth selection (0 disables; SCIP default is 10)")
	bestEstimateProofGapFlag := flag.Float64("best-estimate-proof-gap", 0, "switch best-estimate permanently to best-bound after an incumbent reaches this global relative gap (0 disables)")
	rounding := flag.Bool("rounding", false, "enable simple LP-relaxation rounding heuristic for MIP incumbents")
	parallel := flag.Int("parallel", 1, "number of problems to solve concurrently")
	lpBackendFlag := flag.String("lp-backend", "revised", "LP backend: revised (default, bounded-variable revised simplex with sparse LU + dual reoptimize) or tableau (dense two-phase)")
	branchingFlag := flag.String("branching", "most-fractional", "MIP branching rule: most-fractional (default), pseudocost, or reliability")
	reliabilityThresholdFlag := flag.Int("reliability-threshold", 0, "observations per direction before pseudocost is reliable (0 = dynamic SCIP-style threshold 1..5)")
	reliabilityAdaptiveThresholdFlag := flag.Bool("reliability-adaptive-threshold", false, "lower reliability threshold to 2 after sustained cheap node LPs (explicit threshold wins)")
	reliabilityMaxProbesFlag := flag.Int("reliability-max-probes", 0, "maximum unreliable candidates probed per node (0 = q-opt default 8; larger values use a nine-candidate lookahead stop)")
	reliabilityPrimalBudgetFlag := flag.Int("reliability-primal-iteration-budget", 0, "global strong-branching iteration offset (legacy flag name; 0 = solver default)")
	reliabilityProbeInferenceFlag := flag.Bool("reliability-probe-inference", false, "score reliability candidates using bound reductions found during probe propagation")
	useLPConflictLearningFlag := flag.Bool("lp-conflicts", false, "learn integral bound conflicts from cold-verified infeasible node LPs")
	lpConflictMaxLiteralsFlag := flag.Int("lp-conflict-max-literals", 0, "maximum branch-path literals eligible for LP conflict learning (0 = solver default 12)")
	lpConflictMinimizeProbesFlag := flag.Int("lp-conflict-minimize-probes", 0, "fallback deletion-filter LP solves per learned conflict after Phase-I certificate tests (0 disables)")
	revisedEtaUpdates := flag.Int("revised-eta-updates", 0, "revised backend eta updates before full refactor: 0=default, negative=full refactor every pivot")
	revisedEtaResidualFactor := flag.Float64("revised-eta-residual-factor", 0, "revised backend eta residual refactor threshold multiplier: residual > factor*feasTol*scale; 0=default")
	revisedDenseLU := flag.Bool("revised-dense-lu", false, "revised backend: force the dense LU basis factor (default uses the sparse LU with dense fallback)")
	revisedPersistFactor := flag.Bool("revised-persist-factor", false, "revised backend: reuse the parent node's LU factorization across B&B nodes instead of refactoring each node")
	revisedNoCertifyInfeasible := flag.Bool("revised-no-certify-infeasible", false, "revised backend: disable the aggregate Farkas validation of warm-node LP infeasibility exits and re-prove each with cold Phase I (A/B control; certificates are on by default)")
	revisedFocusSession := flag.Bool("revised-focus-session", false, "revised backend: keep a persistent focus-node LP per canonical body and dual-reoptimize it in place across the tree walk (SCIP SCIPlpSolveAndEval port)")
	dualProofsFlag := flag.Bool("dual-proofs", false, "pool certified Farkas proof rows from infeasible node exits and prune later nodes whose boxes violate a pooled proof (SCIP dual-ray conflict propagation; propagate-only, never LP rows)")
	perNodeCutCap := flag.Int("per-node-cut-cap", 0, "max cuts kept per B&B node (0 = solver default of 20); lower keeps the cut-augmented LP sparse")
	rootCutCapFlag := flag.Int("root-cut-cap", 0, "max cuts retained per root separation round (0 follows per-node cap/default)")
	rootCutStallRoundsFlag := flag.Int("cut-stall-rounds", 0, "persistent root separation: stop only after this many consecutive stalled rounds (SCIP maxstallroundsroot semantics, SCIP default 10); -cut-rounds stays the hard cap. 0 keeps fixed-round behavior")
	rootCutRowAgeLimitFlag := flag.Int("root-cut-row-age-limit", 0, "persistent root separation: deactivate an active root cut row after this many consecutive nonbinding re-solves (SCIP lp/rowagelimit, default 10); 0 keeps applied root cuts permanent")
	skipCutValidationBelow := flag.Int("skip-cut-validation-below", 0, "skip the per-node cut validation re-solve when a node emits <= this many cuts (0 = always validate)")
	pseudoObjPropagationFlag := flag.Bool("pseudo-obj-propagation", false, "enable pre-LP pseudo-objective node prune (box relaxation bound, no matrix work)")
	objectiveCutoffPropagationFlag := flag.Bool("objective-cutoff-propagation", false, "propagate incumbent objective cutoff into variable bounds before node LP solves")
	reducedCostFixingFlag := flag.Bool("reduced-cost-fixing", false, "enable post-solve reduced-cost bound tightening (revised backend only)")
	cutRowAgeLimitFlag := flag.Int("cut-row-age-limit", 0, "replace a per-node or inherited root cut with an empty redundant row after this many consecutive non-binding checks (0 disables aging; SCIP's default is 10)")
	cutRowsDriveBranchingFlag := flag.Bool("cut-rows-drive-branching", false, "use node-local cut LP values for primal heuristics and branch selection")
	useRINSFlag := flag.Bool("rins", false, "enable Relaxation Induced Neighborhood Search: periodically fix vars agreeing with the incumbent and re-solve the reduced sub-problem")
	rinsAdaptiveBudgetFlag := flag.Bool("rins-adaptive-budget", false, "use SCIP-style success/usage-adjusted RINS node budgets (requires -rins)")
	useLocalBranchingFlag := flag.Bool("local-branching", false, "enable incumbent-centered binary local branching")
	localBranchingRadiusFlag := flag.Int("local-branching-radius", 0, "binary Hamming radius for local branching (0 = SCIP default 18)")
	useProximitySearchFlag := flag.Bool("proximity-search", false, "enable SCIP-style incumbent proximity search")
	proximityMinImproveFlag := flag.Float64("proximity-min-improve", 0, "fraction of incumbent-to-LP gap required per proximity call (0 = SCIP default 0.02)")
	disablePrimalHeuristicsFlag := flag.Bool("disable-primal-heuristics", false, "disable repair, feasibility pump, and pseudocost diving for A/B benchmarks")
	repairAdaptiveCadenceFlag := flag.Bool("repair-adaptive-cadence", false, "back off rounded repair after consecutive non-improving incumbent calls")
	detectStructureFlag := flag.Bool("detect-structure", false, "enable row-conflict cuts and verified partitioning-orbitope bound fixings")
	aggregateContinuousEqualitiesFlag := flag.Bool("aggregate-equalities", false, "substitute continuous variables that occur in exactly one equality")
	maxNodesFlag := flag.Int("max-nodes", 0, "per-problem B&B node limit (0 disables)")
	onlyFlag := flag.String("only", "", "comma-separated problem names to run (default: all in -problems)")
	cpuProfile := flag.String("cpuprofile", "", "write a CPU profile covering all solves to this file")
	flag.Parse()
	cutTuningCap = *perNodeCutCap
	rootCutCap = *rootCutCapFlag
	cutTuningSkipValidationBelow = *skipCutValidationBelow
	persistFactor = *revisedPersistFactor
	noCertifyInfeasible = *revisedNoCertifyInfeasible
	focusSession = *revisedFocusSession
	useDualProofs = *dualProofsFlag
	pseudoObjPropagation = *pseudoObjPropagationFlag
	objectiveCutoffPropagation = *objectiveCutoffPropagationFlag
	reducedCostFixing = *reducedCostFixingFlag
	cutRowAgeLimit = *cutRowAgeLimitFlag
	cutRowsDriveBranching = *cutRowsDriveBranchingFlag
	rootCutStallRounds = *rootCutStallRoundsFlag
	rootCutRowAgeLimit = *rootCutRowAgeLimitFlag
	useRINS = *useRINSFlag
	rinsAdaptiveBudget = *rinsAdaptiveBudgetFlag
	useLocalBranching = *useLocalBranchingFlag
	localBranchingRadius = *localBranchingRadiusFlag
	useProximitySearch = *useProximitySearchFlag
	proximityMinImprove = *proximityMinImproveFlag
	disablePrimalHeuristics = *disablePrimalHeuristicsFlag
	repairAdaptiveCadence = *repairAdaptiveCadenceFlag
	detectStructure = *detectStructureFlag
	aggregateContinuousEqualities = *aggregateContinuousEqualitiesFlag
	maxNodes = *maxNodesFlag
	bestEstimateBestBoundFreq = *bestEstimateBestBoundFreqFlag
	bestEstimateProofGap = *bestEstimateProofGapFlag
	reliabilityThreshold = *reliabilityThresholdFlag
	reliabilityAdaptiveThreshold = *reliabilityAdaptiveThresholdFlag
	reliabilityMaxProbes = *reliabilityMaxProbesFlag
	reliabilityPrimalBudget = *reliabilityPrimalBudgetFlag
	reliabilityProbeInference = *reliabilityProbeInferenceFlag
	useLPConflictLearning = *useLPConflictLearningFlag
	lpConflictMaxLiterals = *lpConflictMaxLiteralsFlag
	lpConflictMinimizeProbes = *lpConflictMinimizeProbesFlag

	if *repeat < 1 {
		*repeat = 1
	}
	if *parallel < 1 {
		*parallel = 1
	}
	if *cutRounds <= 0 {
		*cutTimeLimit = 0
	}
	nodeSelection, err := parseNodeSelection(*nodeSelectionFlag)
	if err != nil {
		log.Fatal(err)
	}
	lpBackend, err := parseLPBackend(*lpBackendFlag)
	if err != nil {
		log.Fatal(err)
	}
	branching, err := parseBranching(*branchingFlag)
	if err != nil {
		log.Fatal(err)
	}

	probs, err := problem.LoadDir(*problemsDir)
	if err != nil {
		log.Fatalf("load problems: %v", err)
	}
	if *onlyFlag != "" {
		want := make(map[string]bool)
		for _, name := range strings.Split(*onlyFlag, ",") {
			want[strings.TrimSpace(name)] = true
		}
		filtered := probs[:0]
		for _, p := range probs {
			if want[p.Name] {
				filtered = append(filtered, p)
			}
		}
		probs = filtered
		if len(probs) == 0 {
			log.Fatalf("-only %q matched no problems", *onlyFlag)
		}
	}

	if *cpuProfile != "" {
		f, err := os.Create(*cpuProfile)
		if err != nil {
			log.Fatalf("cpuprofile: %v", err)
		}
		if err := pprof.StartCPUProfile(f); err != nil {
			log.Fatalf("cpuprofile: %v", err)
		}
		defer pprof.StopCPUProfile()
	}

	results := problem.Results{Solver: "qopt", Version: benchmarkVersion()}
	results.Items = solveAll(probs, *timeLimit, *warmup, *repeat, *cutRounds, *cutTimeLimit, *cutDepth, *cutDepthStride, *useMIR, nodeSelection, *rounding, *parallel, lpBackend, branching, *revisedEtaUpdates, *revisedEtaResidualFactor, !*revisedDenseLU, func(item problem.Item) {
		fmt.Printf("%-32s %-12s obj=%-14g bound=%-14s gap=%-9s iter=%-6d nodes=%-4d search=%-10s cuts=%-4d struct=%d/%d/%d/%d cut_ms=%-9.3f median=%.3fms (n=%d)\n",
			item.Name, item.Status, item.Objective, displayMetric(item.DualBound, item.DualBoundKnown), displayMetric(item.RelativeGap, item.RelativeGapKnown), item.Iterations, item.Nodes, item.NodeSelection, item.Cuts, item.StructureConflicts, item.StructureCovers, item.BigMTightenings, item.OrbitopeFixings, item.CutMS, item.ElapsedMS, *repeat)
	})

	if *parallel > 1 {
		sort.Slice(results.Items, func(i, j int) bool {
			return results.Items[i].Name < results.Items[j].Name
		})
	}

	// Solving is done — flush the profile before the results write, whose
	// log.Fatalf would skip the deferred stop and truncate the file.
	if *cpuProfile != "" {
		pprof.StopCPUProfile()
	}

	if err := writeJSON(*outPath, results); err != nil {
		log.Fatalf("write results: %v", err)
	}
	fmt.Printf("wrote %s (%d items)\n", *outPath, len(results.Items))
}

func benchmarkVersion() string {
	info, ok := debug.ReadBuildInfo()
	var revision, modified string
	if ok {
		for _, setting := range info.Settings {
			switch setting.Key {
			case "vcs.revision":
				revision = setting.Value
			case "vcs.modified":
				modified = setting.Value
			}
		}
	}
	if revision == "" {
		output, err := exec.Command("git", "rev-parse", "--short=12", "HEAD").Output()
		if err != nil {
			return "dev"
		}
		revision = strings.TrimSpace(string(output))
		if output, err := exec.Command("git", "status", "--porcelain", "--untracked-files=no").Output(); err == nil && len(output) > 0 {
			modified = "true"
		}
	}
	if len(revision) > 12 {
		revision = revision[:12]
	}
	if modified == "true" {
		return revision + "-dirty"
	}
	return revision
}

func solveAll(probs []*problem.Problem, timeLimit time.Duration, warmup, repeat, cutRounds int, cutTimeLimit time.Duration, cutDepth, cutDepthStride int, useMIR bool, nodeSelection qopt.MIPNodeSelection, rounding bool, parallel int, lpBackend qopt.LPBackend, branching qopt.MIPBranching, revisedEtaUpdates int, revisedEtaResidualFactor float64, revisedSparseLU bool, report func(problem.Item)) []problem.Item {
	if len(probs) == 0 {
		return nil
	}
	if parallel < 1 {
		parallel = 1
	}
	if parallel > len(probs) {
		parallel = len(probs)
	}

	type result struct {
		index int
		item  problem.Item
	}
	jobs := make(chan int)
	done := make(chan result)
	var wg sync.WaitGroup
	for w := 0; w < parallel; w++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for idx := range jobs {
				done <- result{
					index: idx,
					item:  solveRepeated(probs[idx], timeLimit, warmup, repeat, cutRounds, cutTimeLimit, cutDepth, cutDepthStride, useMIR, nodeSelection, rounding, lpBackend, branching, revisedEtaUpdates, revisedEtaResidualFactor, revisedSparseLU),
				}
			}
		}()
	}
	go func() {
		for i := range probs {
			jobs <- i
		}
		close(jobs)
		wg.Wait()
		close(done)
	}()

	items := make([]problem.Item, len(probs))
	for r := range done {
		items[r.index] = r.item
		if report != nil {
			report(r.item)
		}
	}
	return items
}

func solveRepeated(p *problem.Problem, timeLimit time.Duration, warmup, repeat, cutRounds int, cutTimeLimit time.Duration, cutDepth, cutDepthStride int, useMIR bool, nodeSelection qopt.MIPNodeSelection, rounding bool, lpBackend qopt.LPBackend, branching qopt.MIPBranching, revisedEtaUpdates int, revisedEtaResidualFactor float64, revisedSparseLU bool) problem.Item {
	for i := 0; i < warmup; i++ {
		_ = solveOne(p, timeLimit, cutRounds, cutTimeLimit, cutDepth, cutDepthStride, useMIR, nodeSelection, rounding, lpBackend, branching, revisedEtaUpdates, revisedEtaResidualFactor, revisedSparseLU)
	}
	samples := make([]float64, 0, repeat)
	var last problem.Item
	for i := 0; i < repeat; i++ {
		it := solveOne(p, timeLimit, cutRounds, cutTimeLimit, cutDepth, cutDepthStride, useMIR, nodeSelection, rounding, lpBackend, branching, revisedEtaUpdates, revisedEtaResidualFactor, revisedSparseLU)
		samples = append(samples, it.ElapsedMS)
		last = it
	}
	sort.Float64s(samples)
	last.ElapsedMS = samples[len(samples)/2]
	return last
}

func solveOne(p *problem.Problem, timeLimit time.Duration, cutRounds int, cutTimeLimit time.Duration, cutDepth, cutDepthStride int, useMIR bool, nodeSelection qopt.MIPNodeSelection, rounding bool, lpBackend qopt.LPBackend, branching qopt.MIPBranching, revisedEtaUpdates int, revisedEtaResidualFactor float64, revisedSparseLU bool) problem.Item {
	item := problem.Item{Name: p.Name}

	direction, err := parseDirection(p.Direction)
	if err != nil {
		item.Status = "error"
		item.Error = err.Error()
		return item
	}

	m := qopt.NewModel(direction)
	varByName := make(map[string]qopt.Var, len(p.Vars))
	for _, v := range p.Vars {
		spec, err := toVarSpec(v)
		if err != nil {
			item.Status = "error"
			item.Error = err.Error()
			return item
		}
		varByName[v.Name] = m.AddVar(spec)
	}

	obj := make(map[qopt.Var]float64, len(p.Objective))
	for name, c := range p.Objective {
		v, ok := varByName[name]
		if !ok {
			item.Status = "error"
			item.Error = fmt.Sprintf("objective references unknown var %q", name)
			return item
		}
		obj[v] = c
	}
	if err := m.SetObjective(obj); err != nil {
		item.Status = "error"
		item.Error = err.Error()
		return item
	}

	for _, c := range p.Constraints {
		coeffs := make(map[qopt.Var]float64, len(c.Coeffs))
		for name, v := range c.Coeffs {
			ref, ok := varByName[name]
			if !ok {
				item.Status = "error"
				item.Error = fmt.Sprintf("constraint %q references unknown var %q", c.Name, name)
				return item
			}
			coeffs[ref] = v
		}
		sense, err := parseSense(c.Sense)
		if err != nil {
			item.Status = "error"
			item.Error = err.Error()
			return item
		}
		if err := m.AddNamedConstraint(c.Name, coeffs, sense, c.RHS); err != nil {
			item.Status = "error"
			item.Error = err.Error()
			return item
		}
	}

	start := time.Now()
	res, err := m.Solve(context.Background(), qopt.Options{
		TimeLimit:                        timeLimit,
		MaxCutRounds:                     cutRounds,
		RootCutStallRounds:               rootCutStallRounds,
		RootCutRowAgeLimit:               rootCutRowAgeLimit,
		CutTimeLimit:                     cutTimeLimit,
		MaxCutDepth:                      cutDepth,
		CutDepthStride:                   cutDepthStride,
		UseMIRCuts:                       useMIR,
		MIPNodeSelection:                 nodeSelection,
		BestEstimateBestBoundFreq:        bestEstimateBestBoundFreq,
		BestEstimateProofGap:             bestEstimateProofGap,
		RoundingHeuristic:                rounding,
		LPBackend:                        lpBackend,
		MIPBranching:                     branching,
		ReliabilityThreshold:             reliabilityThreshold,
		ReliabilityAdaptiveThreshold:     reliabilityAdaptiveThreshold,
		ReliabilityMaxProbes:             reliabilityMaxProbes,
		ReliabilityPrimalIterationBudget: reliabilityPrimalBudget,
		ReliabilityProbeInference:        reliabilityProbeInference,
		UseLPConflictLearning:            useLPConflictLearning,
		LPConflictMaxLiterals:            lpConflictMaxLiterals,
		LPConflictMinimizeProbes:         lpConflictMinimizeProbes,
		RevisedMaxEtaUpdates:             revisedEtaUpdates,
		RevisedEtaResidualFactor:         revisedEtaResidualFactor,
		RevisedDenseLU:                   !revisedSparseLU,
		PerNodeCutCap:                    cutTuningCap,
		RootCutCap:                       rootCutCap,
		SkipCutValidationBelow:           cutTuningSkipValidationBelow,
		RevisedPersistFactor:             persistFactor,
		RevisedNoCertifyInfeasible:       noCertifyInfeasible,
		RevisedFocusSession:              focusSession,
		UseDualProofs:                    useDualProofs,
		PseudoObjPropagation:             pseudoObjPropagation,
		ObjectiveCutoffPropagation:       objectiveCutoffPropagation,
		ReducedCostFixing:                reducedCostFixing,
		CutRowAgeLimit:                   cutRowAgeLimit,
		CutRowsDriveBranching:            cutRowsDriveBranching,
		UseRINS:                          useRINS,
		RINSAdaptiveBudget:               rinsAdaptiveBudget,
		UseLocalBranching:                useLocalBranching,
		LocalBranchingRadius:             localBranchingRadius,
		UseProximitySearch:               useProximitySearch,
		ProximityMinImprove:              proximityMinImprove,
		DisablePrimalHeuristics:          disablePrimalHeuristics,
		RepairAdaptiveCadence:            repairAdaptiveCadence,
		DetectStructure:                  detectStructure,
		AggregateContinuousEqualities:    aggregateContinuousEqualities,
		MaxNodes:                         maxNodes,
	})
	item.ElapsedMS = float64(time.Since(start).Microseconds()) / 1000.0
	if err != nil {
		item.Status = "error"
		item.Error = err.Error()
		return item
	}
	item.Status = statusString(res.Status)
	item.Iterations = res.Iterations
	item.Nodes = res.Nodes
	if hasIntegerVars(p) {
		item.NodeSelection = res.NodeSelection.String()
		if isFinite(res.DualBound) {
			item.DualBound = sanitize(res.DualBound)
			item.DualBoundKnown = true
		}
		if isFinite(res.RelativeGap) {
			item.RelativeGap = sanitize(res.RelativeGap)
			item.RelativeGapKnown = true
		}
	}
	item.CutRounds = res.CutRounds
	item.Cuts = res.Cuts
	item.CutMS = float64(res.CutElapsed.Microseconds()) / 1000.0
	if res.Stats != nil {
		if len(res.Stats.RootSeparation) > 0 {
			item.RootSeparation = make([]problem.RootSeparationRoundStats, 0, len(res.Stats.RootSeparation))
			for _, round := range res.Stats.RootSeparation {
				families := make(map[string]problem.CutFamilyRoundStats, len(round.Families))
				for name, family := range round.Families {
					families[name] = problem.CutFamilyRoundStats{
						Generated:           family.Generated,
						Selected:            family.Selected,
						Applied:             family.Applied,
						RowprepModified:     family.RowprepModified,
						CoefficientsRemoved: family.CoefficientsRemoved,
						CoefficientsRounded: family.CoefficientsRounded,
					}
				}
				validationStatus := ""
				if round.ValidationAttempted {
					validationStatus = round.ValidationStatus.String()
				}
				item.RootSeparation = append(item.RootSeparation, problem.RootSeparationRoundStats{
					Round:                     round.Round,
					InputRows:                 round.InputRows,
					LPStatus:                  round.LPStatus.String(),
					LPObjective:               sanitize(round.LPObjective),
					LPIterations:              round.LPIterations,
					ValidationAttempted:       round.ValidationAttempted,
					ValidationStatus:          validationStatus,
					ValidationObjective:       sanitize(round.ValidationObjective),
					ValidationIterations:      round.ValidationIterations,
					ObjectiveGain:             sanitize(round.ObjectiveGain),
					ElapsedMS:                 float64(round.Elapsed.Microseconds()) / 1000.0,
					Generated:                 round.Generated,
					Selected:                  round.Selected,
					Applied:                   round.Applied,
					PoolSize:                  round.PoolSize,
					PoolAdded:                 round.PoolAdded,
					PoolDuplicates:            round.PoolDuplicates,
					PoolCandidates:            round.PoolCandidates,
					PoolTested:                round.PoolTested,
					PoolRetested:              round.PoolRetested,
					PoolReusedCandidates:      round.PoolReusedCandidates,
					PoolSelectedRetained:      round.PoolSelectedRetained,
					PoolAgedOut:               round.PoolAgedOut,
					ActiveRows:                round.ActiveRows,
					RowprepModified:           round.RowprepModified,
					CoefficientsRemoved:       round.CoefficientsRemoved,
					CoefficientsRounded:       round.CoefficientsRounded,
					NonzerosBefore:            round.NonzerosBefore,
					NonzerosAfter:             round.NonzerosAfter,
					Families:                  families,
					Rejections:                round.Rejections,
					MaxReferenceLoss:          sanitize(round.MaxReferenceLoss),
					MaxEfficacyBefore:         sanitize(round.MaxEfficacyBefore),
					MaxEfficacy:               sanitize(round.MaxEfficacy),
					MaxDensity:                sanitize(round.MaxDensity),
					MaxCoefficientBefore:      sanitize(round.MaxCoefficientBefore),
					MaxCoefficient:            sanitize(round.MaxCoefficient),
					MaxCoefficientRatioBefore: sanitize(round.MaxCoefficientRatioBefore),
					MaxCoefficientRatio:       sanitize(round.MaxCoefficientRatio),
				})
			}
		}
		if res.Stats.Presolve.OriginalColumns > 0 {
			item.Presolve = &problem.PresolveStats{
				OriginalColumns:   res.Stats.Presolve.OriginalColumns,
				OriginalRows:      res.Stats.Presolve.OriginalRows,
				OriginalNonzeros:  res.Stats.Presolve.OriginalNonzeros,
				PresolvedColumns:  res.Stats.Presolve.PresolvedColumns,
				PresolvedRows:     res.Stats.Presolve.PresolvedRows,
				PresolvedNonzeros: res.Stats.Presolve.PresolvedNonzeros,
				AggregatedColumns: res.Stats.Presolve.AggregatedColumns,
			}
		}
		item.StructureConflicts = res.Stats.Structural.ConflictCuts
		item.StructureCovers = res.Stats.Structural.CoverCuts
		item.StructureImpliedBounds = res.Stats.Structural.ImpliedBoundCuts
		item.StructureFlowCovers = res.Stats.Structural.FlowCoverCuts
		item.BigMTightenings = res.Stats.Structural.BigMTightenings
		item.OrbitopeFixings = res.Stats.Structural.OrbitopeFixings
		item.VariableBoundRows = res.Stats.Structural.VariableBoundRows
		item.StructureImplications = res.Stats.Structural.Implications
		item.ProbeSides = res.Stats.Structural.ProbeSides
		item.InfeasibleProbeSides = res.Stats.Structural.InfeasibleSides
		item.GlobalTightenings = res.Stats.Structural.GlobalTightenings
		item.InferenceBranches = res.Stats.Structural.InferenceBranches
		item.RootCleanSolves = res.Stats.Structural.RootCleanSolves
		item.RootCleanMS = float64(res.Stats.Structural.RootCleanElapsed.Microseconds()) / 1000
		item.RootCutTreeRows = res.Stats.CutLifecycle.RootRows
		item.RootCutTreeNodeSamples = res.Stats.CutLifecycle.NodeSamples
		item.RootCutTreeActiveRowsTotal = res.Stats.CutLifecycle.ActiveRowsTotal
		item.RootCutTreePeakActiveRows = res.Stats.CutLifecycle.PeakActiveRows
		item.RootCutTreePoolRetests = res.Stats.CutLifecycle.PoolRetests
		item.RootCutTreeRowsDeactivated = res.Stats.CutLifecycle.RowsDeactivated
		item.RootCutTreeRowsReactivated = res.Stats.CutLifecycle.RowsReactivated
		item.ReliabilityCandidates = res.Stats.Branching.ReliabilityCandidates
		item.ReliabilityLPSolves = res.Stats.Branching.ReliabilityLPSolves
		item.ReliabilityIterations = res.Stats.Branching.ReliabilityIterations
		item.ReliabilityInfeasible = res.Stats.Branching.ReliabilityInfeasibleSides
		item.ReliabilitySkipped = res.Stats.Branching.ReliabilityChildrenSkipped
		item.ReliabilityBoundReuses = res.Stats.Branching.ReliabilityBoundReuses
		item.ReliabilityWarmReuses = res.Stats.Branching.ReliabilityWarmReuses
		item.ReliabilityDomainReductions = res.Stats.Branching.ReliabilityDomainReductions
		item.ReliabilitySelectedProbed = res.Stats.Branching.ReliabilitySelectedProbed
		item.ReliabilityDiscardedProbes = res.Stats.Branching.ReliabilityDiscardedProbes
		item.ReliabilityAdaptiveLowNodes = res.Stats.Branching.ReliabilityAdaptiveLowNodes
		item.ReliabilityAdaptiveHighNodes = res.Stats.Branching.ReliabilityAdaptiveHighNodes
		item.ReliabilityNativeProbes = res.Stats.Branching.ReliabilityNativeProbes
		item.ReliabilityFallbackProbes = res.Stats.Branching.ReliabilityFallbackProbes
		item.ReliabilityIterationLimits = res.Stats.Branching.ReliabilityIterationLimits
		item.ReliabilityValidLimitBounds = res.Stats.Branching.ReliabilityValidLimitBounds
		item.ReliabilityUncertInfeasible = res.Stats.Branching.ReliabilityUncertInfeasible
		item.ReliabilityLookaheadStops = res.Stats.Branching.ReliabilityLookaheadStops
		item.BestEstimateProofSwitches = res.Stats.Branching.BestEstimateProofSwitches
		item.BestEstimateProofSwitchNode = res.Stats.Branching.BestEstimateProofSwitchNode
		item.BestEstimateProofSwitchGap = res.Stats.Branching.BestEstimateProofSwitchGap
		item.LPConflictsLearned = res.Stats.Branching.LPConflictsLearned
		item.LPConflictLiterals = res.Stats.Branching.LPConflictLiterals
		item.LPConflictPrunes = res.Stats.Branching.LPConflictPrunes
		item.LPConflictPropagations = res.Stats.Branching.LPConflictPropagations
		item.LPConflictLPSolves = res.Stats.Branching.LPConflictLPSolves
		item.LPConflictIterations = res.Stats.Branching.LPConflictIterations
		item.LPConflictCertificateTests = res.Stats.Branching.LPConflictCertificateTests
		item.LPConflictCertificateProofs = res.Stats.Branching.LPConflictCertificateProofs
		item.ObjectiveCutoffReductions = res.Stats.Branching.ObjectiveCutoffReductions
		item.ObjectiveCutoffPrunes = res.Stats.Branching.ObjectiveCutoffPrunes
		item.DualProofsPooled = res.Stats.DualProofs.Pooled
		item.DualProofsRejectedDense = res.Stats.DualProofs.RejectedDense
		item.DualProofsRejectedInfinite = res.Stats.DualProofs.RejectedInfinite
		item.DualProofsEvicted = res.Stats.DualProofs.Evicted
		item.DualProofPrunes = res.Stats.DualProofs.Prunes
		item.DualProofVerifyContradicted = res.Stats.DualProofs.VerifyContradicted
		if res.Stats.Revised.LPSolves > 0 {
			item.RevisedStats = &problem.RevisedStats{
				LPSolves:                   res.Stats.Revised.LPSolves,
				Refactors:                  res.Stats.Revised.Refactors,
				EtaUpdates:                 res.Stats.Revised.EtaUpdates,
				EtaRefactorFallbacks:       res.Stats.Revised.EtaRefactorFallbacks,
				MaxEtaDepth:                res.Stats.Revised.MaxEtaDepth,
				MaxBasisResidual:           sanitize(res.Stats.Revised.MaxBasisResidual),
				MaxBasisRelResidual:        sanitize(res.Stats.Revised.MaxBasisRelResidual),
				MinPivot:                   sanitize(res.Stats.Revised.MinPivot),
				MaxGrowth:                  sanitize(res.Stats.Revised.MaxGrowth),
				FactorReuses:               res.Stats.Revised.FactorReuses,
				SparseDenseFallbacks:       res.Stats.Revised.SparseDenseFallbacks,
				PhaseIIterations:           res.Stats.Revised.PhaseIIterations,
				PhaseIIIterations:          res.Stats.Revised.PhaseIIIterations,
				DualIterations:             res.Stats.Revised.DualIterations,
				SparseFactors:              res.Stats.Revised.SparseFactors,
				DenseFactors:               res.Stats.Revised.DenseFactors,
				ColdStarts:                 res.Stats.Revised.ColdStarts,
				WarmReopts:                 res.Stats.Revised.WarmReopts,
				WarmFlipRemaps:             res.Stats.Revised.WarmFlipRemaps,
				WarmFailures:               res.Stats.Revised.WarmFailures,
				WarmRejected:               res.Stats.Revised.WarmRejected,
				WarmInstallFailures:        res.Stats.Revised.WarmInstallFailures,
				WarmDualInfeasible:         res.Stats.Revised.WarmDualInfeasible,
				WarmInfeasibleCandidates:   res.Stats.Revised.WarmInfeasibleCandidates,
				WarmInfeasibleCertificates: res.Stats.Revised.WarmInfeasibleCertificates,
				WarmInfeasibleChecks:       res.Stats.Revised.WarmInfeasibleChecks,
				WarmInfeasibleInconclusive: res.Stats.Revised.WarmInfeasibleInconclusive,
				WarmInfeasibleContradicted: res.Stats.Revised.WarmInfeasibleContradicted,
				WarmCertRejectBasis:        res.Stats.Revised.WarmCertRejectBasis,
				WarmCertRejectInfinite:     res.Stats.Revised.WarmCertRejectInfinite,
				WarmCertRejectMargin:       res.Stats.Revised.WarmCertRejectMargin,
				WarmCertRejectInvalid:      res.Stats.Revised.WarmCertRejectInvalid,
				WarmDualIterationCaps:      res.Stats.Revised.WarmDualIterationCaps,
				WarmArtificialFailure:      res.Stats.Revised.WarmArtificialFailure,
				WarmColdInfeasible:         res.Stats.Revised.WarmColdInfeasible,
				WarmCertAccepts:            res.Stats.Revised.WarmCertAccepts,
				WarmCertVerifies:           res.Stats.Revised.WarmCertVerifies,
				WarmCertVerifyFails:        res.Stats.Revised.WarmCertVerifyFails,
				WarmCertBoundCorrections:   res.Stats.Revised.WarmCertBoundCorrections,
				WarmCertVerifyResumes:      res.Stats.Revised.WarmCertVerifyResumes,
				WarmCertBoundCorrectionMax: sanitize(res.Stats.Revised.WarmCertBoundCorrectionMax),
				DualStalls:                 res.Stats.Revised.DualStalls,
				DualNoEntering:             res.Stats.Revised.DualNoEntering,
				DualIterationCaps:          res.Stats.Revised.DualIterationCaps,
				SessionReopts:              res.Stats.Revised.SessionReopts,
				SessionFallbacks:           res.Stats.Revised.SessionFallbacks,
				SessionAdoptions:           res.Stats.Revised.SessionAdoptions,
				SessionInfeasibleRetries:   res.Stats.Revised.SessionInfeasibleRetries,
				PhaseIMS:                   float64(res.Stats.Revised.PhaseINanos) / 1e6,
				PhaseIIMS:                  float64(res.Stats.Revised.PhaseIINanos) / 1e6,
				DualMS:                     float64(res.Stats.Revised.DualNanos) / 1e6,
				SparseFactorMS:             float64(res.Stats.Revised.SparseFactorNanos) / 1e6,
				DenseFactorMS:              float64(res.Stats.Revised.DenseFactorNanos) / 1e6,
			}
		}
		if len(res.Stats.Incumbents) > 0 {
			item.Incumbents = make([]problem.IncumbentEvent, 0, len(res.Stats.Incumbents))
			for _, ev := range res.Stats.Incumbents {
				item.Incumbents = append(item.Incumbents, problem.IncumbentEvent{
					ElapsedMS: float64(ev.Elapsed.Microseconds()) / 1000.0,
					Node:      ev.Node,
					Objective: sanitize(ev.Objective),
					Source:    ev.Source,
				})
			}
		}
		if len(res.Stats.Bounds) > 0 {
			item.Bounds = make([]problem.BoundEvent, 0, len(res.Stats.Bounds))
			for _, ev := range res.Stats.Bounds {
				item.Bounds = append(item.Bounds, problem.BoundEvent{
					ElapsedMS: float64(ev.Elapsed.Microseconds()) / 1000.0,
					Node:      ev.Node, Iterations: ev.Iterations, DualBound: sanitize(ev.DualBound),
				})
			}
		}
		if len(res.Stats.Heuristics) > 0 {
			item.Heuristics = make(map[string]problem.HeuristicStats, len(res.Stats.Heuristics))
			for name, hs := range res.Stats.Heuristics {
				item.Heuristics[name] = problem.HeuristicStats{
					Attempts: hs.Attempts, Successes: hs.Successes, LPSolves: hs.LPSolves,
					ElapsedMS: float64(hs.Elapsed.Microseconds()) / 1000, Reward: hs.Reward,
				}
			}
		}
	}
	// Surface the incumbent any time the solver has one — at optimality
	// always, and when it stopped early carrying its best feasible point.
	if res.Values != nil {
		item.Objective = sanitize(res.Objective)
		item.Values = make(map[string]float64, len(res.Values))
		for i, v := range res.Values {
			item.Values[p.Vars[i].Name] = sanitize(v)
		}
	}
	return item
}

func hasIntegerVars(p *problem.Problem) bool {
	for _, v := range p.Vars {
		switch v.Type {
		case "integer", "int", "binary", "bool":
			return true
		}
	}
	return false
}

func parseLPBackend(s string) (qopt.LPBackend, error) {
	switch s {
	case "", "tableau", "dense":
		return qopt.LPBackendTableau, nil
	case "revised":
		return qopt.LPBackendRevised, nil
	default:
		return 0, fmt.Errorf("unknown lp-backend %q", s)
	}
}

func parseBranching(s string) (qopt.MIPBranching, error) {
	switch s {
	case "", "most-fractional", "most_fractional", "mostfrac":
		return qopt.MIPBranchMostFractional, nil
	case "pseudocost", "pcost":
		return qopt.MIPBranchPseudocost, nil
	case "reliability", "rel":
		return qopt.MIPBranchReliability, nil
	default:
		return 0, fmt.Errorf("unknown branching %q", s)
	}
}

func parseNodeSelection(s string) (qopt.MIPNodeSelection, error) {
	switch s {
	case "", "dfs", "depth-first", "depth_first":
		return qopt.MIPDepthFirst, nil
	case "best-bound", "best_bound":
		return qopt.MIPBestBound, nil
	case "best-estimate", "best_estimate", "estimate":
		return qopt.MIPBestEstimate, nil
	default:
		return 0, fmt.Errorf("unknown node selection %q", s)
	}
}

func parseDirection(s string) (qopt.Direction, error) {
	switch s {
	case "min", "minimize":
		return qopt.Minimize, nil
	case "max", "maximize":
		return qopt.Maximize, nil
	default:
		return 0, fmt.Errorf("unknown direction %q", s)
	}
}

func parseSense(s string) (qopt.Sense, error) {
	switch s {
	case "<=", "le":
		return qopt.LessOrEqual, nil
	case ">=", "ge":
		return qopt.GreaterOrEqual, nil
	case "=", "==", "eq":
		return qopt.Equal, nil
	default:
		return 0, fmt.Errorf("unknown sense %q", s)
	}
}

func toVarSpec(v problem.VarSpec) (qopt.VarSpec, error) {
	out := qopt.VarSpec{Name: v.Name}
	switch v.Type {
	case "", "continuous":
		out.Type = qopt.Continuous
	case "integer", "int":
		out.Type = qopt.Integer
	case "binary", "bool":
		out.Type = qopt.Binary
	default:
		return qopt.VarSpec{}, fmt.Errorf("var %q: unknown type %q", v.Name, v.Type)
	}
	// Forward the optional bounds verbatim as pointers. A nil Lower/Upper means
	// "unspecified", which qopt.AddVar resolves to the default (0 / +Inf, or
	// [0,1] for Binary). Crucially this preserves an explicit `upper: 0` for a
	// variable fixed at zero — a plain float64 field could not distinguish that
	// from "unset" and silently turned the variable unbounded (the neos17 /
	// glass4 / timtab1 infeasible-incumbent class).
	if v.LowerUnbounded {
		if v.Lower != nil {
			return qopt.VarSpec{}, fmt.Errorf("var %q: lower and lower_unbounded are mutually exclusive", v.Name)
		}
		if out.Type == qopt.Binary {
			return qopt.VarSpec{}, fmt.Errorf("var %q: binary variable cannot be lower-unbounded", v.Name)
		}
		out.Lower = qopt.Float64(math.Inf(-1))
	} else if v.Lower != nil {
		out.Lower = qopt.Float64(*v.Lower)
	}
	if v.Upper != nil {
		out.Upper = qopt.Float64(*v.Upper)
	}
	return out, nil
}

func statusString(s qopt.Status) string {
	switch s {
	case qopt.Optimal:
		return "optimal"
	case qopt.Infeasible:
		return "infeasible"
	case qopt.Unbounded:
		return "unbounded"
	case qopt.IterationLimit:
		return "iteration_limit"
	case qopt.NodeLimit:
		return "node_limit"
	case qopt.TimeLimit:
		return "time_limit"
	case qopt.GapLimit:
		return "gap_limit"
	default:
		return "unknown"
	}
}

func sanitize(v float64) float64 {
	// JSON cannot encode NaN/±Inf — they arise legitimately (e.g. RelativeGap
	// is +Inf and DualBound is NaN when an instance finds no incumbent), so
	// map them to 0 rather than letting the whole results write fail.
	if math.IsNaN(v) || math.IsInf(v, 0) {
		return 0
	}
	if math.Abs(v) < 1e-12 {
		return 0
	}
	return v
}

func isFinite(v float64) bool {
	return !math.IsNaN(v) && !math.IsInf(v, 0)
}

func displayMetric(v float64, known bool) string {
	if !known {
		return "n/a"
	}
	return strconv.FormatFloat(v, 'g', 6, 64)
}

func writeJSON(path string, v any) error {
	if err := os.MkdirAll(parentDir(path), 0o755); err != nil {
		return err
	}
	raw, err := json.MarshalIndent(v, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(path, append(raw, '\n'), 0o644)
}

func parentDir(p string) string {
	for i := len(p) - 1; i >= 0; i-- {
		if p[i] == '/' {
			return p[:i]
		}
	}
	return "."
}
