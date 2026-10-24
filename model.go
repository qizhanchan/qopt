package qopt

import (
	"context"
	"errors"
	"math"
	"time"

	"github.com/qizhanchan/qopt/lp"
)

type Direction int

const (
	Minimize Direction = iota
	Maximize
)

type VarType int

const (
	Continuous VarType = iota
	Integer
	Binary
)

// Sense and Status are shared with the LP engine subpackage (lp defines
// them; qopt re-exports them as aliases so both layers speak one vocabulary
// without conversions at the backend boundary).
type Sense = lp.Sense

const (
	LessOrEqual    = lp.LessOrEqual
	GreaterOrEqual = lp.GreaterOrEqual
	Equal          = lp.Equal
)

type Status = lp.Status

const (
	Optimal        = lp.Optimal
	Infeasible     = lp.Infeasible
	Unbounded      = lp.Unbounded
	IterationLimit = lp.IterationLimit
	NodeLimit      = lp.NodeLimit
	TimeLimit      = lp.TimeLimit
	GapLimit       = lp.GapLimit
)

type Var int

type MIPNodeSelection int

const (
	MIPDepthFirst MIPNodeSelection = iota
	MIPBestBound
	// MIPBestEstimate orders open nodes by a pseudocost estimate of the best
	// integer objective reachable in their subtree. The LP objective remains
	// the authoritative pruning/dual bound; the estimate changes search order
	// only.
	MIPBestEstimate
)

func (s MIPNodeSelection) String() string {
	switch s {
	case MIPDepthFirst:
		return "dfs"
	case MIPBestBound:
		return "best-bound"
	case MIPBestEstimate:
		return "best-estimate"
	default:
		return "unknown"
	}
}

// LPBackend selects which LP solver implementation handles relaxations.
// The default (zero value) is the revised backend: bounded-variable revised
// simplex with a sparse LU basis (dense-LU stability fallback) and dual
// re-optimization for MIP node warm-starts — the measured best across the
// MIPLIB sub-suite. LPBackendTableau selects the original dense two-phase
// primal simplex (still the sole producer of the simplex state per-node Gomory
// cut generation reads).
type LPBackend int

const (
	// LPBackendRevised is the zero value so a plain Options{} gets the
	// recommended backend.
	LPBackendRevised LPBackend = iota
	LPBackendTableau
)

func (b LPBackend) String() string {
	switch b {
	case LPBackendTableau:
		return "tableau"
	case LPBackendRevised:
		return "revised"
	default:
		return "unknown"
	}
}

// MIPBranching selects the strategy used to pick which fractional integer
// variable to branch on at each B&B node. Default is most-fractional, a
// dumb but deterministic rule that matches v1's documented behavior.
// Pseudocost maintains per-variable down/up gain estimates updated from
// past LP-objective degradations and is the standard "smart" choice.
type MIPBranching int

const (
	MIPBranchMostFractional MIPBranching = iota
	MIPBranchPseudocost
	// MIPBranchReliability is pseudocost branching augmented with on-the-fly
	// strong-branching probes at every node for variables whose pseudocost
	// history is still "unreliable" (fewer than reliabilityThreshold
	// observations in either direction). Each unreliable candidate gets a
	// down- and up-probe (a warm-started LP solve) whose real degradation is
	// folded into the pseudocost table before the branch variable is chosen,
	// so the selection is driven by actual LP movement rather than a cold
	// prior. Reliable variables skip the probe and trust their accumulated
	// average, so the per-node probe cost falls off as the tree deepens.
	MIPBranchReliability
)

func (b MIPBranching) String() string {
	switch b {
	case MIPBranchMostFractional:
		return "most-fractional"
	case MIPBranchPseudocost:
		return "pseudocost"
	case MIPBranchReliability:
		return "reliability"
	default:
		return "unknown"
	}
}

// VarSpec describes a variable to add to a Model. Lower and Upper are optional:
// a nil pointer means "unspecified", which AddVar resolves to the natural
// default (Lower → 0, Upper → +∞). Use a non-nil pointer to set an explicit
// bound — including an explicit 0, which a plain float64 field could not
// distinguish from "unset" (the bug that let a variable fixed at [0,0] become
// unbounded and produce infeasible incumbents on neos17/glass4/timtab1). The
// Float64 helper builds a pointer from a literal.
type VarSpec struct {
	Name  string
	Type  VarType
	Lower *float64
	Upper *float64
}

// Float64 returns a pointer to v, for setting explicit VarSpec bounds from a
// literal (e.g. VarSpec{Lower: Float64(0), Upper: Float64(0)} for a fixed
// variable).
func Float64(v float64) *float64 { return &v }

type Constraint struct {
	Coeffs map[Var]float64
	Sense  Sense
	RHS    float64
	Name   string
}

type Options struct {
	FeasibilityTol float64
	IntegralityTol float64
	OptimalityTol  float64
	MaxIterations  int
	MaxNodes       int
	TimeLimit      time.Duration
	// MIPGap stops exploring a node once its valid LP bound is within this
	// relative gap of the incumbent. Such a solve returns GapLimit with the
	// retained bound/gap; it never promotes a tolerance stop to Optimal.
	MIPGap float64
	// AggregateContinuousEqualities enables a structural presolve pass for
	// continuous variables that occur in exactly one equality. The variable is
	// substituted out of that row and the objective, its bounds become at most
	// two rows over the remaining variables, and postsolve reconstructs its
	// original value. This deliberately narrow form keeps the transformation
	// exact and the postsolve map stable across B&B nodes. It composes with
	// global root cuts, which are generated in original space before the
	// aggregated cache is built. It remains disabled with per-node cuts because
	// their basis metadata still assumes a one-to-one original-row mapping.
	AggregateContinuousEqualities bool
	// MaxCutRounds enables Gomory fractional cuts at the MIP root node.
	// Each round re-solves the LP relaxation and adds any newly-derived cuts
	// before B&B begins. Zero disables cuts (default; v1 is conservative).
	// Root cuts are generated on the tableau backend internally (Gomory needs
	// the simplex tableau the revised backend doesn't expose), but the cuts
	// are valid for all integer points, so they compose with any B&B
	// LPBackend — LPBackendRevised now sees root cuts too.
	MaxCutRounds int
	// RootCutStallRounds ports SCIP's separating/maxstallroundsroot to the
	// persistent root separation loop (revised backend only): separation
	// continues past objective plateaus until this many consecutive rounds
	// neither improve the root LP bound by more than a 1e-4 relative step nor
	// reduce the fractional-variable count (the SCIP stall test in
	// solve.c:priceAndCutLoop). MaxCutRounds remains the hard round cap.
	// Zero (default) keeps the fixed-round behavior: exactly MaxCutRounds
	// rounds with no stall-based continuation or early stop. SCIP's default
	// is 10.
	RootCutStallRounds int
	// RootCutRowAgeLimit ports SCIP's lp/rowagelimit to active root cut rows:
	// a cut row that stays nonbinding at the root LP optimum for more than
	// this many consecutive separation re-solves is deactivated (its slot is
	// emptied in place, preserving the LP row layout for warm starts) while
	// its cut-pool entry becomes retestable and can re-enter later rounds.
	// Deactivation is checked every (limit+1)/2+1 solves, matching SCIP's
	// removal cadence. Zero (default) keeps applied root cuts permanently
	// active. SCIP's default is 10.
	RootCutRowAgeLimit int
	// CutTimeLimit caps total time spent generating and validating root cuts.
	// Zero leaves cut generation governed only by context cancellation and the
	// overall solve time limit.
	CutTimeLimit time.Duration
	// MaxCutDepth enables Gomory cuts inside the B&B tree. Cuts are derived at
	// every node with `depth ≤ MaxCutDepth` (root is depth 0) and inherited by
	// that node's subtree as additional row constraints. Zero (default)
	// disables per-node cuts; root-only cuts still run when MaxCutRounds > 0.
	// Per-node cuts compose with either LPBackend: under the revised backend
	// (which exposes no simplex state) the cut-generating node is re-solved
	// once on the tableau backend to recover the tableau Gomory needs, then the
	// valid cuts flow into the revised warm-start subtree like any constraint.
	MaxCutDepth int
	// CutDepthStride sparsifies per-node cut firing along depth. The cut gate
	// fires only at depths d ∈ {0, stride, 2·stride, ...} (still capped at
	// MaxCutDepth). Zero is treated as 1 (cut at every depth up to MaxCutDepth,
	// the most aggressive setting). CBC's stall-mode default is 5, which keeps
	// the LP from being tightened so densely that the rounding heuristic lands
	// on worse integer points than the no-cut search would have found.
	CutDepthStride int
	// UseMIRCuts enables Marchand-Wolsey row MIR cuts alongside Gomory at the
	// same cut-generation points (root via MaxCutRounds, per-node via
	// MaxCutDepth). Row MIR processes each user constraint at a fixed set of
	// scaling factors (currently {1, 1/2}), applies the integer rounding
	// formula, and emits any cut violated by the current LP relaxation. The
	// 1/2 scale recovers the 0-1/2 / Chvátal closure that fires on the
	// markshare-style equality constraints (pk1 / markshare_4_0). Default off.
	// Unlike Gomory, MIR works on user constraints rather than basis rows, so
	// it composes with both tableau and revised backends.
	UseMIRCuts bool
	// RoundingHeuristic is retained for source compatibility but no longer
	// gates the rounding heuristic — the solver now always runs LP-relaxation
	// rounding candidates before branching, since on MIPLIB instances where
	// no integer feasible solution naturally emerges (ic97_potential,
	// neos17, timtab1, …) it is the difference between returning an
	// incumbent and returning nothing.
	RoundingHeuristic bool
	// MIPNodeSelection controls which open B&B node is solved next. The
	// default is deterministic depth-first search; best-bound and
	// best-estimate are opt-in because they change time-limited incumbent
	// discovery behavior.
	MIPNodeSelection MIPNodeSelection
	// BestEstimateBestBoundFreq makes best-estimate select the open node with
	// the best authoritative LP bound every Nth selection. Zero disables the
	// periodic proof step. SCIP's estimate selector defaults to 10.
	BestEstimateBestBoundFreq int
	// BestEstimateProofGap switches an active best-estimate queue to pure
	// best-bound selection after a newly improved incumbent brings the global
	// relative gap at or below this value. The heap is rebuilt once, so proof
	// selections remain O(log n). Zero disables the adaptive switch.
	BestEstimateProofGap float64
	// LPBackend selects the LP relaxation solver. The zero value (default) is
	// LPBackendRevised — bounded-variable revised simplex with a sparse LU
	// basis and dual re-optimization for B&B warm starts. Set LPBackendTableau
	// to use the original dense two-phase simplex.
	LPBackend LPBackend
	// MIPBranching selects the variable-selection rule used at each B&B
	// node. Default MIPBranchMostFractional preserves v1's deterministic
	// most-fractional rule. MIPBranchPseudocost uses online per-variable
	// gain estimates derived from past LP-objective degradations.
	MIPBranching MIPBranching
	// ReliabilityThreshold is the number of observations required in both
	// directions before reliability branching trusts a variable's pseudocost.
	// Zero uses a SCIP-style dynamic threshold between 1 and 5 based on the
	// remaining global strong-branch budget. An explicit value disables that
	// interpolation.
	ReliabilityThreshold int
	// ReliabilityAdaptiveThreshold lowers the default reliability threshold
	// from 5 to 2 after enough cheap node LPs have been observed. Expensive LP
	// trees keep the conservative threshold. An explicit ReliabilityThreshold
	// always wins. Default off pending matched MIPLIB measurement.
	ReliabilityAdaptiveThreshold bool
	// ReliabilityMaxProbes caps the number of unreliable candidates strong-
	// branched at one node. Zero uses the q-opt-calibrated default of 8. A
	// separate SCIP-style lookahead stop applies when a larger value is set.
	ReliabilityMaxProbes int
	// ReliabilityPrimalIterationBudget is the legacy name for the global
	// strong-branching iteration offset. It now applies throughout the search;
	// zero uses the built-in default. q-opt's LP iteration economics differ
	// from SoPlex, so benchmark profiles may deliberately set this below
	// SCIP's 100000.
	ReliabilityPrimalIterationBudget int
	// ReliabilityProbeInference adds the domain reductions found by each
	// strong-branch LP's existing bound-propagation pass to the candidate
	// score. It is experimental and disabled by default.
	ReliabilityProbeInference bool
	// UseLPConflictLearning learns integral bound conflicts from B&B nodes whose
	// LP relaxation is cold-verified infeasible. Learned conflicts are applied
	// as pre-LP cutoffs and unit propagation. The learner re-verifies assumptions
	// against the global model whenever inferred bounds or local rows were
	// present, so reduced-cost fixings and local cuts cannot leak into a global
	// conflict. Default off pending matched MIPLIB measurement.
	UseLPConflictLearning bool
	// LPConflictMaxLiterals limits the branch depth eligible for LP conflict
	// learning. Zero uses 12; longer paths are skipped instead of storing weak,
	// expensive no-goods.
	LPConflictMaxLiterals int
	// LPConflictMinimizeProbes limits fallback deletion-filter LP solves per
	// learned conflict after cheap Phase-I certificate tests. Zero disables
	// fallback solves; certificate-only shrinking still runs.
	LPConflictMinimizeProbes int
	// RevisedMaxEtaUpdates controls how many product-form eta basis updates
	// the revised LP backend keeps before rebuilding a fresh LU factorization.
	// Zero uses the backend default. Negative disables eta updates and full
	// refactors after every pivot.
	RevisedMaxEtaUpdates int
	// RevisedEtaResidualFactor controls the scaled basis residual threshold
	// that triggers a full LU refactor after an eta update:
	// residual > factor * FeasibilityTol * scale. Zero uses the backend
	// default.
	RevisedEtaResidualFactor float64
	// RevisedDenseLU forces the dense LU basis factor inside the revised
	// backend. The default (false) uses the sparse LU factor — which has a
	// dense-LU stability fallback for ill-conditioned bases — and is the
	// measured throughput win on larger/sparser instances. Set true to pin the
	// revised backend to the dense LU. Affects only LPBackendRevised.
	RevisedDenseLU bool
	// RevisedPersistFactor carries the LU factorization (base factor + eta
	// chain) across B&B nodes in the warm-start handle and reuses it instead of
	// re-factoring at every node. A child node differs from its parent by a
	// single variable bound, so the basis matrix — and hence the
	// factorization — is identical; only x_B needs recomputing. Reuse is
	// guarded by an eta-chain cap and a basis-residual check that fall back to
	// a fresh refactor on drift, so correctness is preserved. Cuts the dominant
	// per-node cost (~2 refactors/node) on warm-start B&B. Default off because
	// the carried eta chains add memory proportional to the open-node set,
	// which best-bound can make large. Affects only LPBackendRevised.
	RevisedPersistFactor bool
	// RevisedNoCertifyInfeasible disables the aggregate Farkas validation of
	// warm dual "no entering column" exits and re-proves every such node with
	// cold Phase I instead. Certificates are ON by default — the SCIP-parity
	// posture (SCIPlpGetDualfarkas under lp/checkfarkas): the matched MIPLIB
	// suite showed no regression on any instance, node-identical retained
	// proofs on markshare_4_0/pk1, and 4-5x node throughput on
	// neos17/timtab1/murg, with the toy suite unchanged against OR-Tools.
	// The disable switch exists for A/B control runs.
	RevisedNoCertifyInfeasible bool
	// UseDualProofs pools the aggregate Farkas proof rows behind certified
	// infeasible node exits and reuses them as propagating conflicts: a later
	// node whose (post-propagation) box provably violates a pooled proof is
	// pruned as certified infeasible without preparing its LP. This is SCIP's
	// default dual-ray conflict path (conflict/useinflp, dualray proofs stored
	// as propagate-only linear constraints in the conflictstore) — proofs are
	// never added to the LP, matching the measured verdict that conflict rows
	// in the LP are a net loss. Requires the revised backend with infeasibility
	// certificates enabled (the default). Default off pending matched MIPLIB
	// measurement.
	UseDualProofs bool
	// RevisedFocusSession enables the SCIP-style focus-node LP: each canonical
	// LP body keeps a persistent, already-assembled revised solver (basis,
	// bounds, factorization, eta chain) that follows the tree walk. A node
	// whose LP shape matches the live state rewrites only the RHS and the
	// structural upper bounds in place and dual-reoptimizes from the current
	// basis, skipping the per-node warm-start classification, solver build,
	// and install refactor. Shape changes and unsound in-place outcomes fall
	// back to the stateless path (which then re-seeds the session), so the
	// acceptance rules — certified or explicitly verified dual optimum, cold
	// Phase I as the infeasibility reference — are unchanged. Default off
	// pending matched MIPLIB measurement. Affects only LPBackendRevised.
	RevisedFocusSession bool
	// PerNodeCutCap overrides the maximum number of cuts kept from a single
	// B&B node's cut batch. Zero uses the built-in default (perNodeCutCap).
	// Lowering it keeps the cut-augmented LP sparse so the per-node second LP
	// solve stays cheap — the dominant cost when per-node cuts are enabled.
	PerNodeCutCap int
	// RootCutCap overrides the cuts retained per root separation round without
	// changing PerNodeCutCap. Zero preserves compatibility by falling back to
	// PerNodeCutCap and then the built-in default.
	RootCutCap int
	// SkipCutValidationBelow skips the per-node cut validation re-solve when a
	// node emits at most this many cuts. The validation re-solve is a full
	// extra LP per cut-generating node that exists only to catch numerically
	// poisoned cuts (false Unbounded/Infeasible); the non-root Unbounded prune
	// and the warm-start cold-verify already catch those at the child, so for
	// small cut batches the re-solve is pure overhead. Zero (default) always
	// validates, preserving v1 behavior.
	SkipCutValidationBelow int
	// DetectSymmetry enables variable-permutation symmetry detection at the
	// MIP root. Candidate orbits are found via Weisfeiler-Leman color
	// refinement over the colored variable/constraint graph, then every
	// candidate pair is independently verified against the actual
	// coefficient data (does swapping the two columns, together with some
	// permutation of the rows, reproduce the model exactly?) before it is
	// trusted — so this can only remove symmetric duplicate solutions, never
	// cut off the true optimum. Verified orbits get a lexicographic-ordering
	// constraint chain, breaking permutation symmetry without an LP solve.
	// Default off: detection cost and the extra constraints are pure
	// overhead on instances with no symmetry (most of the current suite).
	DetectSymmetry bool
	// DetectStructure enables root-level structural reductions. The
	// first pass recognizes row-implied binary conflicts and verified
	// partitioning-orbitope column symmetry.
	// It is opt-in while the structural-cut families are being validated.
	DetectStructure bool
	// PseudoObjPropagation enables a pre-LP node prune: before solving a
	// node's LP relaxation, compute the box relaxation's objective (each
	// variable pinned to whichever of its current lower/upper bound helps
	// the objective most, ignoring every constraint row) and check it
	// against the incumbent with the same boundCannotImprove logic used for
	// the real LP bound. Dropping every row can only enlarge the feasible
	// region, so this box value is a valid — if weaker — bound; whenever it
	// already can't beat the incumbent, the LP solve for that node is
	// skipped entirely. O(n) arithmetic per node, no matrix work. Default
	// off pending measurement; mirrors SCIP's prop_pseudoobj.c.
	PseudoObjPropagation bool
	// ObjectiveCutoffPropagation strengthens pseudo-objective pruning by using
	// the incumbent-improvement cutoff as a temporary single row for bound
	// propagation. It alternates with ordinary row propagation without adding
	// a row to the LP, so revised-simplex warm-start shape stays unchanged.
	// Default off pending matched proof-tail measurement.
	ObjectiveCutoffPropagation bool
	// ReducedCostFixing enables post-solve bound tightening from a B&B
	// node's LP-optimal reduced costs (revised backend only — the tableau
	// backend's simplexState carries no reduced-cost accessor). For each
	// nonbasic structural column with a reduced cost strong enough that
	// moving it any further would already beat the LP-to-incumbent gap, the
	// implied tighter bound is derived (a standard LP-duality bound; see
	// reducedCostBounds) and attached to that node's children alongside the
	// branching bound. Requires a known incumbent — with none, there is no
	// gap to derive a bound from. Never contradicts existing bounds; any
	// resulting infeasibility is caught by the same tightenBounds/
	// canonicalize checks that already run at every node. Default off
	// pending measurement; mirrors SCIP's prop_redcost.c.
	ReducedCostFixing bool
	// CutRowAgeLimit enables tree cut-row aging for both inherited global root
	// cuts and subtree-local cuts: a row that stays nonbinding for more than
	// this many consecutive checks along a path is replaced by an empty
	// redundant slot. Every pooled row is still checked against each new LP
	// point and reactivates when it becomes tight or violated. Stable row
	// positions preserve revised-simplex warm starts; see cutaging.go. Zero
	// (default) keeps every emitted/applied cut permanent, matching v1.
	// Root-cut aging currently requires the revised non-aggregated cache.
	CutRowAgeLimit int
	// CutRowsDriveBranching lets node-local cut relaxations provide the point
	// used by primal heuristics and branch-variable selection. The default
	// false retains the historical clean-LP decoupling, which protects primal
	// trajectories on several instances but discards the tree-steering effect
	// of sustained separation. Global root cuts already drive branching and are
	// unaffected by this option.
	CutRowsDriveBranching bool
	// UseRINS enables Relaxation Induced Neighborhood Search: periodically,
	// once an incumbent exists, fix every integer/binary variable whose LP
	// relaxation value already agrees with the incumbent and re-solve the
	// reduced sub-problem with its own small budget, installing the result
	// as the new incumbent if it's better. Targets the "found a feasible
	// solution but can't refine it" gap measured against HiGHS/SCIP on
	// mad/neos-3046615-murg/neos-3754480-nidda. Default off: matched MIPLIB
	// measurement found a useful murg incumbent gain but mixed proof impact.
	// See rins.go and docs/solver-milestones.md.
	UseRINS bool
	// RINSAdaptiveBudget enables SCIP-style success- and usage-adjusted node
	// limits for RINS sub-solves. It requires UseRINS. Zero-value false keeps
	// the legacy fixed formula. The adaptive mode ranges from 50 to 5000 nodes,
	// accounts for prior calls, and only fires after 4000 outer nodes when the
	// current LP-to-incumbent relative gap is at least 50%. A failed first
	// built neighborhood disables further adaptive RINS calls for that solve.
	RINSAdaptiveBudget bool
	// UseLocalBranching enables an incumbent-centered binary Hamming-distance
	// neighborhood sub-MIP. It is opt-in: matched MIPLIB measurement improves
	// assign1-5-8 from 218.249 to the SCIP/HiGHS incumbent 212, but the other
	// eligible instances only consume the capped one-second call.
	UseLocalBranching bool
	// LocalBranchingRadius is the maximum number of incumbent binary values
	// that may change in the sub-MIP. Zero uses SCIP's default radius 18.
	LocalBranchingRadius int
	// UseProximitySearch enables a SCIP-style sub-MIP that constrains the
	// original objective to improve and minimizes binary Hamming distance to
	// the incumbent. Default off: it improves murg/neos5 in matched short runs,
	// but consumes proof throughput and competes with RINS/local branching.
	UseProximitySearch bool
	// ProximityMinImprove is the fraction of the incumbent-to-LP-bound gap
	// that each proximity sub-MIP must close. Zero uses SCIP's default 0.02.
	ProximityMinImprove float64
	// DisablePrimalHeuristics disables Milestone-1 repair, feasibility-pump,
	// and diving calls. It is intended for deterministic A/B benchmarks and
	// recursive heuristic sub-solves; the default false enables the scheduler.
	DisablePrimalHeuristics bool
	// RepairAdaptiveCadence exponentially backs off the CPU-only rounded repair
	// heuristic after consecutive non-improving calls once an incumbent exists.
	// Plain LP rounding remains active at every node. Default off pending
	// proof-tail throughput measurement.
	RepairAdaptiveCadence bool

	// Internal per-node incumbent cutoff, populated by the MIP driver after
	// options normalization. These fields are deliberately unexported so the
	// public option remains the boolean policy above.
	objectiveCutoffActive bool
	objectiveCutoff       float64
}

type Result struct {
	Status        Status
	Objective     float64
	Values        []float64
	Iterations    int
	Nodes         int
	CutRounds     int
	Cuts          int
	CutElapsed    time.Duration
	NodeSelection MIPNodeSelection
	Stats         *SolverStats
	// DualBound is the best proven bound on the MIP optimum at the time the
	// search ended: a lower bound for Minimize, an upper bound for Maximize.
	// It combines the open B&B frontier with inherited bounds retained for any
	// unresolved or gap-pruned subtree, and equals Objective exactly when the
	// tree was exhausted (Status == Optimal ⇒ proven). GapLimit means the
	// requested MIPGap was reached without claiming an exact proof. NaN when no
	// bound is known (e.g. the root LP never finished, or Unbounded/error). For
	// MIP results only; LP results leave it at 0.
	DualBound float64
	// RelativeGap is |Objective − DualBound| / max(1, |Objective|), the
	// optimality gap at termination. 0 means proven optimal; +Inf means no
	// incumbent or no bound yet. This is the "how far from a proof" readout.
	RelativeGap float64
}

type SolverStats struct {
	Revised        RevisedStats
	Presolve       PresolveStats
	Structural     StructuralStats
	Branching      BranchingStats
	CutLifecycle   CutLifecycleStats
	DualProofs     DualProofStats
	RootSeparation []RootSeparationRoundStats
	Incumbents     []IncumbentEvent
	Bounds         []BoundEvent
	Heuristics     map[string]HeuristicStats
}

type CutLifecycleStats struct {
	RootRows        int
	NodeSamples     int
	ActiveRowsTotal int
	PeakActiveRows  int
	PoolRetests     int
	RowsDeactivated int
	RowsReactivated int
}

type RootSeparationRoundStats struct {
	Round                     int
	InputRows                 int
	LPStatus                  Status
	LPObjective               float64
	LPIterations              int
	ValidationAttempted       bool
	ValidationStatus          Status
	ValidationObjective       float64
	ValidationIterations      int
	ObjectiveGain             float64
	Elapsed                   time.Duration
	Generated                 int
	Selected                  int
	Applied                   int
	PoolSize                  int
	PoolAdded                 int
	PoolDuplicates            int
	PoolCandidates            int
	PoolTested                int
	PoolRetested              int
	PoolReusedCandidates      int
	PoolSelectedRetained      int
	PoolAgedOut               int
	ActiveRows                int
	RowprepModified           int
	CoefficientsRemoved       int
	CoefficientsRounded       int
	NonzerosBefore            int
	NonzerosAfter             int
	Families                  map[string]CutFamilyRoundStats
	Rejections                map[string]int
	MaxReferenceLoss          float64
	MaxEfficacyBefore         float64
	MaxEfficacy               float64
	MaxDensity                float64
	MaxCoefficientBefore      float64
	MaxCoefficient            float64
	MaxCoefficientRatioBefore float64
	MaxCoefficientRatio       float64
	// Persistent-loop lifecycle telemetry (SCIP priceAndCutLoop parity).
	Fractionals int  // fractional discrete variables at the post-round LP optimum
	StallRounds int  // consecutive stall rounds after this round (SCIP nsepastallrounds)
	Deactivated int  // active rows aged out and emptied during this round
	SlotsReused int  // newly applied cuts that filled a previously emptied slot
	WarmSolve   bool // the round's re-solve reused the persistent LP warm start
}

type CutFamilyRoundStats struct {
	Generated           int
	Selected            int
	Applied             int
	RowprepModified     int
	CoefficientsRemoved int
	CoefficientsRounded int
}

type BoundEvent struct {
	Elapsed    time.Duration
	Node       int
	Iterations int
	DualBound  float64
}

type PresolveStats struct {
	OriginalColumns   int
	OriginalRows      int
	OriginalNonzeros  int
	PresolvedColumns  int
	PresolvedRows     int
	PresolvedNonzeros int
	AggregatedColumns int
}

type BranchingStats struct {
	ReliabilityCandidates        int
	ReliabilityLPSolves          int
	ReliabilityIterations        int
	ReliabilityInfeasibleSides   int
	ReliabilityChildrenSkipped   int
	ReliabilityBoundReuses       int
	ReliabilityWarmReuses        int
	ReliabilityDomainReductions  int
	ReliabilitySelectedProbed    int
	ReliabilityDiscardedProbes   int
	ReliabilityAdaptiveLowNodes  int
	ReliabilityAdaptiveHighNodes int
	ReliabilityNativeProbes      int
	ReliabilityFallbackProbes    int
	ReliabilityIterationLimits   int
	ReliabilityValidLimitBounds  int
	ReliabilityUncertInfeasible  int
	ReliabilityLookaheadStops    int
	BestEstimateProofSwitches    int
	BestEstimateProofSwitchNode  int
	BestEstimateProofSwitchGap   float64
	LPConflictsLearned           int
	LPConflictLiterals           int
	LPConflictPrunes             int
	LPConflictPropagations       int
	LPConflictLPSolves           int
	LPConflictIterations         int
	LPConflictCertificateTests   int
	LPConflictCertificateProofs  int
	ObjectiveCutoffReductions    int
	ObjectiveCutoffPrunes        int
}

type StructuralStats struct {
	ConflictCuts      int
	CoverCuts         int
	ImpliedBoundCuts  int
	FlowCoverCuts     int
	BigMTightenings   int
	OrbitopeFixings   int
	VariableBoundRows int
	Implications      int
	ProbeSides        int
	InfeasibleSides   int
	GlobalTightenings int
	InferenceBranches int
	RootCleanSolves   int
	RootCleanElapsed  time.Duration
}

type HeuristicStats struct {
	Attempts  int
	Successes int
	LPSolves  int
	Elapsed   time.Duration
	Reward    float64
}

// RevisedStats aggregates the LP engine's numerical/diagnostic counters. It
// is an alias of the engine's own stats type; see lp.Stats for field docs.
type RevisedStats = lp.Stats

type IncumbentEvent struct {
	Elapsed   time.Duration
	Node      int
	Objective float64
	Source    string
}

func (s *SolverStats) addRevised(r RevisedStats) {
	if s == nil || r.LPSolves == 0 {
		return
	}
	s.Revised.Add(r)
}

func (s *SolverStats) recordIncumbent(elapsed time.Duration, node int, objective float64, source string) {
	if s == nil {
		return
	}
	s.Incumbents = append(s.Incumbents, IncumbentEvent{
		Elapsed:   elapsed,
		Node:      node,
		Objective: objective,
		Source:    source,
	})
}

func (s *SolverStats) recordHeuristic(name string, success bool, lpSolves int, elapsed time.Duration) {
	if s == nil {
		return
	}
	if s.Heuristics == nil {
		s.Heuristics = make(map[string]HeuristicStats)
	}
	h := s.Heuristics[name]
	if elapsed == 0 && !success && lpSolves == 0 {
		h.Attempts++
	} else {
		if success {
			h.Successes++
		}
		h.LPSolves += lpSolves
		h.Elapsed += elapsed
	}
	s.Heuristics[name] = h
}

func (s *SolverStats) recordHeuristicReward(name string, reward float64) {
	if s == nil || reward == 0 {
		return
	}
	if s.Heuristics == nil {
		s.Heuristics = make(map[string]HeuristicStats)
	}
	h := s.Heuristics[name]
	h.Reward += reward
	s.Heuristics[name] = h
}

type Model struct {
	direction   Direction
	vars        []VarSpec
	objective   map[Var]float64
	constraints []Constraint
	// rowsSorted caches each constraint's coefficients in ascending variable
	// order (see coeffs.go) so solve-path float accumulations are
	// deterministic regardless of map iteration order. Lazily rebuilt when
	// the constraint count changes.
	rowsSorted [][]coefEntry
	// domainData caches the immutable row/column adjacency used by queued bound
	// propagation. Like rowsSorted it is rebuilt when the model shape changes.
	domainData *domainData
}

func NewModel(direction Direction) *Model {
	return &Model{
		direction: direction,
		objective: make(map[Var]float64),
	}
}

func (m *Model) AddVar(spec VarSpec) Var {
	// Resolve optional bounds into concrete values stored on the spec, so the
	// rest of the solver can read m.vars[i].Lower/Upper as always-non-nil. A
	// nil pointer means "use the default"; a non-nil pointer (including an
	// explicit 0) is honored verbatim.
	lower, upper := 0.0, math.Inf(1)
	if spec.Type == Binary {
		upper = 1
	}
	if spec.Lower != nil {
		lower = *spec.Lower
	}
	if spec.Upper != nil {
		upper = *spec.Upper
	}
	if spec.Type == Binary {
		// Intersect explicit bounds with [0,1]. This preserves fixed binaries
		// such as [1,1]; an empty intersection is detected as infeasible later.
		lower = math.Max(lower, 0)
		upper = math.Min(upper, 1)
	}
	spec.Lower = &lower
	spec.Upper = &upper
	v := Var(len(m.vars))
	m.vars = append(m.vars, spec)
	return v
}

func (m *Model) SetObjective(coeffs map[Var]float64) error {
	next := make(map[Var]float64, len(coeffs))
	for v, c := range coeffs {
		if err := m.checkVar(v); err != nil {
			return err
		}
		if c != 0 {
			next[v] = c
		}
	}
	m.objective = next
	return nil
}

func (m *Model) AddConstraint(coeffs map[Var]float64, sense Sense, rhs float64) error {
	return m.AddNamedConstraint("", coeffs, sense, rhs)
}

func (m *Model) AddNamedConstraint(name string, coeffs map[Var]float64, sense Sense, rhs float64) error {
	if sense != LessOrEqual && sense != GreaterOrEqual && sense != Equal {
		return errors.New("qopt: invalid constraint sense")
	}
	next := make(map[Var]float64, len(coeffs))
	for v, c := range coeffs {
		if err := m.checkVar(v); err != nil {
			return err
		}
		if c != 0 {
			next[v] = c
		}
	}
	m.constraints = append(m.constraints, Constraint{
		Coeffs: next,
		Sense:  sense,
		RHS:    rhs,
		Name:   name,
	})
	return nil
}

func (m *Model) Solve(ctx context.Context, opts Options) (Result, error) {
	if ctx == nil {
		ctx = context.Background()
	}
	opts = normalizeOptions(opts)
	if hasIntegerVars(m.vars) {
		return solveMIP(ctx, m, opts)
	}
	return solveLPModel(ctx, m, nil, opts)
}

func (m *Model) checkVar(v Var) error {
	if v < 0 || int(v) >= len(m.vars) {
		return errors.New("qopt: variable does not belong to model")
	}
	return nil
}

func normalizeOptions(opts Options) Options {
	if opts.FeasibilityTol <= 0 {
		opts.FeasibilityTol = 1e-8
	}
	if opts.IntegralityTol <= 0 {
		opts.IntegralityTol = 1e-8
	}
	if opts.OptimalityTol <= 0 {
		opts.OptimalityTol = 1e-9
	}
	if opts.MaxIterations <= 0 {
		opts.MaxIterations = 100000
	}
	if opts.MIPGap < 0 {
		opts.MIPGap = 0
	}
	if opts.BestEstimateBestBoundFreq < 0 {
		opts.BestEstimateBestBoundFreq = 0
	}
	if opts.BestEstimateProofGap < 0 {
		opts.BestEstimateProofGap = 0
	}
	if opts.BestEstimateProofGap > 1 {
		opts.BestEstimateProofGap = 1
	}
	if opts.LocalBranchingRadius < 0 {
		opts.LocalBranchingRadius = 0
	}
	if opts.ProximityMinImprove < 0 {
		opts.ProximityMinImprove = 0
	}
	if opts.ProximityMinImprove > 1 {
		opts.ProximityMinImprove = 1
	}
	return opts
}

func hasIntegerVars(vars []VarSpec) bool {
	for _, v := range vars {
		if v.Type == Integer || v.Type == Binary {
			return true
		}
	}
	return false
}
