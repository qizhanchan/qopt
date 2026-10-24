// Package problem describes the shared JSON benchmark format used by both the
// q-opt runner and the OR-Tools runner. See bench/README.md for the schema.
package problem

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sort"
)

type VarSpec struct {
	Name           string   `json:"name"`
	Type           string   `json:"type"`
	Lower          *float64 `json:"lower,omitempty"`
	LowerUnbounded bool     `json:"lower_unbounded,omitempty"`
	Upper          *float64 `json:"upper,omitempty"`
}

type Constraint struct {
	Name   string             `json:"name,omitempty"`
	Coeffs map[string]float64 `json:"coeffs"`
	Sense  string             `json:"sense"`
	RHS    float64            `json:"rhs"`
}

type Expectation struct {
	Status    string             `json:"status,omitempty"`
	Objective *float64           `json:"objective,omitempty"`
	Values    map[string]float64 `json:"values,omitempty"`
}

type Problem struct {
	Name        string             `json:"name"`
	Direction   string             `json:"direction"`
	Vars        []VarSpec          `json:"vars"`
	Objective   map[string]float64 `json:"objective"`
	Constraints []Constraint       `json:"constraints"`
	Expect      *Expectation       `json:"expect,omitempty"`
}

type Item struct {
	Name                         string                     `json:"name"`
	Status                       string                     `json:"status"`
	Objective                    float64                    `json:"objective"`
	Values                       map[string]float64         `json:"values,omitempty"`
	Iterations                   int                        `json:"iterations,omitempty"`
	Nodes                        int                        `json:"nodes,omitempty"`
	DualBound                    float64                    `json:"dual_bound,omitempty"`
	RelativeGap                  float64                    `json:"relative_gap,omitempty"`
	DualBoundKnown               bool                       `json:"dual_bound_known,omitempty"`
	RelativeGapKnown             bool                       `json:"relative_gap_known,omitempty"`
	NodeSelection                string                     `json:"node_selection,omitempty"`
	CutRounds                    int                        `json:"cut_rounds,omitempty"`
	Cuts                         int                        `json:"cuts,omitempty"`
	CutMS                        float64                    `json:"cut_ms,omitempty"`
	RootSeparation               []RootSeparationRoundStats `json:"root_separation,omitempty"`
	StructureConflicts           int                        `json:"structure_conflicts,omitempty"`
	StructureCovers              int                        `json:"structure_covers,omitempty"`
	StructureImpliedBounds       int                        `json:"structure_implied_bounds,omitempty"`
	StructureFlowCovers          int                        `json:"structure_flow_covers,omitempty"`
	BigMTightenings              int                        `json:"big_m_tightenings,omitempty"`
	OrbitopeFixings              int                        `json:"orbitope_fixings,omitempty"`
	VariableBoundRows            int                        `json:"variable_bound_rows,omitempty"`
	StructureImplications        int                        `json:"structure_implications,omitempty"`
	ProbeSides                   int                        `json:"probe_sides,omitempty"`
	InfeasibleProbeSides         int                        `json:"infeasible_probe_sides,omitempty"`
	GlobalTightenings            int                        `json:"global_tightenings,omitempty"`
	InferenceBranches            int                        `json:"inference_branches,omitempty"`
	RootCleanSolves              int                        `json:"root_clean_solves,omitempty"`
	RootCleanMS                  float64                    `json:"root_clean_ms,omitempty"`
	RootCutTreeRows              int                        `json:"root_cut_tree_rows,omitempty"`
	RootCutTreeNodeSamples       int                        `json:"root_cut_tree_node_samples,omitempty"`
	RootCutTreeActiveRowsTotal   int                        `json:"root_cut_tree_active_rows_total,omitempty"`
	RootCutTreePeakActiveRows    int                        `json:"root_cut_tree_peak_active_rows,omitempty"`
	RootCutTreePoolRetests       int                        `json:"root_cut_tree_pool_retests,omitempty"`
	RootCutTreeRowsDeactivated   int                        `json:"root_cut_tree_rows_deactivated,omitempty"`
	RootCutTreeRowsReactivated   int                        `json:"root_cut_tree_rows_reactivated,omitempty"`
	ReliabilityCandidates        int                        `json:"reliability_candidates,omitempty"`
	ReliabilityLPSolves          int                        `json:"reliability_lp_solves,omitempty"`
	ReliabilityIterations        int                        `json:"reliability_iterations,omitempty"`
	ReliabilityInfeasible        int                        `json:"reliability_infeasible_sides,omitempty"`
	ReliabilitySkipped           int                        `json:"reliability_children_skipped,omitempty"`
	ReliabilityBoundReuses       int                        `json:"reliability_bound_reuses,omitempty"`
	ReliabilityWarmReuses        int                        `json:"reliability_warm_reuses,omitempty"`
	ReliabilityDomainReductions  int                        `json:"reliability_domain_reductions,omitempty"`
	ReliabilitySelectedProbed    int                        `json:"reliability_selected_probed,omitempty"`
	ReliabilityDiscardedProbes   int                        `json:"reliability_discarded_probes,omitempty"`
	ReliabilityAdaptiveLowNodes  int                        `json:"reliability_adaptive_low_nodes,omitempty"`
	ReliabilityAdaptiveHighNodes int                        `json:"reliability_adaptive_high_nodes,omitempty"`
	ReliabilityNativeProbes      int                        `json:"reliability_native_probes,omitempty"`
	ReliabilityFallbackProbes    int                        `json:"reliability_fallback_probes,omitempty"`
	ReliabilityIterationLimits   int                        `json:"reliability_iteration_limits,omitempty"`
	ReliabilityValidLimitBounds  int                        `json:"reliability_valid_limit_bounds,omitempty"`
	ReliabilityUncertInfeasible  int                        `json:"reliability_uncertified_infeasible,omitempty"`
	ReliabilityLookaheadStops    int                        `json:"reliability_lookahead_stops,omitempty"`
	BestEstimateProofSwitches    int                        `json:"best_estimate_proof_switches,omitempty"`
	BestEstimateProofSwitchNode  int                        `json:"best_estimate_proof_switch_node,omitempty"`
	BestEstimateProofSwitchGap   float64                    `json:"best_estimate_proof_switch_gap,omitempty"`
	LPConflictsLearned           int                        `json:"lp_conflicts_learned,omitempty"`
	LPConflictLiterals           int                        `json:"lp_conflict_literals,omitempty"`
	LPConflictPrunes             int                        `json:"lp_conflict_prunes,omitempty"`
	LPConflictPropagations       int                        `json:"lp_conflict_propagations,omitempty"`
	LPConflictLPSolves           int                        `json:"lp_conflict_lp_solves,omitempty"`
	LPConflictIterations         int                        `json:"lp_conflict_iterations,omitempty"`
	LPConflictCertificateTests   int                        `json:"lp_conflict_certificate_tests,omitempty"`
	LPConflictCertificateProofs  int                        `json:"lp_conflict_certificate_proofs,omitempty"`
	ObjectiveCutoffReductions    int                        `json:"objective_cutoff_reductions,omitempty"`
	ObjectiveCutoffPrunes        int                        `json:"objective_cutoff_prunes,omitempty"`
	DualProofsPooled             int                        `json:"dual_proofs_pooled,omitempty"`
	DualProofsRejectedDense      int                        `json:"dual_proofs_rejected_dense,omitempty"`
	DualProofsRejectedInfinite   int                        `json:"dual_proofs_rejected_infinite,omitempty"`
	DualProofsEvicted            int                        `json:"dual_proofs_evicted,omitempty"`
	DualProofPrunes              int                        `json:"dual_proof_prunes,omitempty"`
	DualProofVerifyContradicted  int                        `json:"dual_proof_verify_contradicted,omitempty"`
	ElapsedMS                    float64                    `json:"elapsed_ms"`
	Presolve                     *PresolveStats             `json:"presolve,omitempty"`
	RevisedStats                 *RevisedStats              `json:"revised_stats,omitempty"`
	Incumbents                   []IncumbentEvent           `json:"incumbents,omitempty"`
	Bounds                       []BoundEvent               `json:"bounds,omitempty"`
	Heuristics                   map[string]HeuristicStats  `json:"heuristics,omitempty"`
	Error                        string                     `json:"error,omitempty"`
}

type RootSeparationRoundStats struct {
	Round                     int                            `json:"round"`
	InputRows                 int                            `json:"input_rows"`
	LPStatus                  string                         `json:"lp_status"`
	LPObjective               float64                        `json:"lp_objective,omitempty"`
	LPIterations              int                            `json:"lp_iterations,omitempty"`
	ValidationAttempted       bool                           `json:"validation_attempted,omitempty"`
	ValidationStatus          string                         `json:"validation_status,omitempty"`
	ValidationObjective       float64                        `json:"validation_objective,omitempty"`
	ValidationIterations      int                            `json:"validation_iterations,omitempty"`
	ObjectiveGain             float64                        `json:"objective_gain,omitempty"`
	ElapsedMS                 float64                        `json:"elapsed_ms,omitempty"`
	Generated                 int                            `json:"generated,omitempty"`
	Selected                  int                            `json:"selected,omitempty"`
	Applied                   int                            `json:"applied,omitempty"`
	PoolSize                  int                            `json:"pool_size,omitempty"`
	PoolAdded                 int                            `json:"pool_added,omitempty"`
	PoolDuplicates            int                            `json:"pool_duplicates,omitempty"`
	PoolCandidates            int                            `json:"pool_candidates,omitempty"`
	PoolTested                int                            `json:"pool_tested,omitempty"`
	PoolRetested              int                            `json:"pool_retested,omitempty"`
	PoolReusedCandidates      int                            `json:"pool_reused_candidates,omitempty"`
	PoolSelectedRetained      int                            `json:"pool_selected_retained,omitempty"`
	PoolAgedOut               int                            `json:"pool_aged_out,omitempty"`
	ActiveRows                int                            `json:"active_rows,omitempty"`
	RowprepModified           int                            `json:"rowprep_modified,omitempty"`
	CoefficientsRemoved       int                            `json:"coefficients_removed,omitempty"`
	CoefficientsRounded       int                            `json:"coefficients_rounded,omitempty"`
	NonzerosBefore            int                            `json:"nonzeros_before,omitempty"`
	NonzerosAfter             int                            `json:"nonzeros_after,omitempty"`
	Families                  map[string]CutFamilyRoundStats `json:"families,omitempty"`
	Rejections                map[string]int                 `json:"rejections,omitempty"`
	MaxReferenceLoss          float64                        `json:"max_reference_loss,omitempty"`
	MaxEfficacyBefore         float64                        `json:"max_efficacy_before,omitempty"`
	MaxEfficacy               float64                        `json:"max_efficacy,omitempty"`
	MaxDensity                float64                        `json:"max_density,omitempty"`
	MaxCoefficientBefore      float64                        `json:"max_coefficient_before,omitempty"`
	MaxCoefficient            float64                        `json:"max_coefficient,omitempty"`
	MaxCoefficientRatioBefore float64                        `json:"max_coefficient_ratio_before,omitempty"`
	MaxCoefficientRatio       float64                        `json:"max_coefficient_ratio,omitempty"`
}

type CutFamilyRoundStats struct {
	Generated           int `json:"generated,omitempty"`
	Selected            int `json:"selected,omitempty"`
	Applied             int `json:"applied,omitempty"`
	RowprepModified     int `json:"rowprep_modified,omitempty"`
	CoefficientsRemoved int `json:"coefficients_removed,omitempty"`
	CoefficientsRounded int `json:"coefficients_rounded,omitempty"`
}

type PresolveStats struct {
	OriginalColumns   int `json:"original_columns"`
	OriginalRows      int `json:"original_rows"`
	OriginalNonzeros  int `json:"original_nonzeros"`
	PresolvedColumns  int `json:"presolved_columns"`
	PresolvedRows     int `json:"presolved_rows"`
	PresolvedNonzeros int `json:"presolved_nonzeros"`
	AggregatedColumns int `json:"aggregated_columns,omitempty"`
}

type HeuristicStats struct {
	Attempts  int     `json:"attempts"`
	Successes int     `json:"successes"`
	LPSolves  int     `json:"lp_solves"`
	ElapsedMS float64 `json:"elapsed_ms"`
	Reward    float64 `json:"reward,omitempty"`
}

type BoundEvent struct {
	ElapsedMS  float64 `json:"elapsed_ms"`
	Node       int     `json:"node"`
	Iterations int     `json:"iterations"`
	DualBound  float64 `json:"dual_bound"`
}

type RevisedStats struct {
	LPSolves                   int     `json:"lp_solves,omitempty"`
	Refactors                  int     `json:"refactors,omitempty"`
	FactorReuses               int     `json:"factor_reuses,omitempty"`
	EtaUpdates                 int     `json:"eta_updates,omitempty"`
	EtaRefactorFallbacks       int     `json:"eta_refactor_fallbacks,omitempty"`
	MaxEtaDepth                int     `json:"max_eta_depth,omitempty"`
	MaxBasisResidual           float64 `json:"max_basis_residual,omitempty"`
	MaxBasisRelResidual        float64 `json:"max_basis_rel_residual,omitempty"`
	MinPivot                   float64 `json:"min_pivot,omitempty"`
	MaxGrowth                  float64 `json:"max_growth,omitempty"`
	SparseDenseFallbacks       int     `json:"sparse_dense_fallbacks,omitempty"`
	PhaseIIterations           int     `json:"phase1_iterations,omitempty"`
	PhaseIIIterations          int     `json:"phase2_iterations,omitempty"`
	DualIterations             int     `json:"dual_iterations,omitempty"`
	SparseFactors              int     `json:"sparse_factors,omitempty"`
	DenseFactors               int     `json:"dense_factors,omitempty"`
	ColdStarts                 int     `json:"cold_starts,omitempty"`
	WarmReopts                 int     `json:"warm_reopts,omitempty"`
	WarmFlipRemaps             int     `json:"warm_flip_remaps,omitempty"`
	WarmFailures               int     `json:"warm_failures,omitempty"`
	WarmRejected               int     `json:"warm_rejected,omitempty"`
	WarmInstallFailures        int     `json:"warm_install_failures,omitempty"`
	WarmDualInfeasible         int     `json:"warm_dual_infeasible,omitempty"`
	WarmInfeasibleCandidates   int     `json:"warm_infeasible_candidates,omitempty"`
	WarmInfeasibleCertificates int     `json:"warm_infeasible_certificates,omitempty"`
	WarmInfeasibleChecks       int     `json:"warm_infeasible_checks,omitempty"`
	WarmInfeasibleInconclusive int     `json:"warm_infeasible_inconclusive,omitempty"`
	WarmInfeasibleContradicted int     `json:"warm_infeasible_contradicted,omitempty"`
	WarmCertRejectBasis        int     `json:"warm_cert_reject_basis,omitempty"`
	WarmCertRejectInfinite     int     `json:"warm_cert_reject_infinite,omitempty"`
	WarmCertRejectMargin       int     `json:"warm_cert_reject_margin,omitempty"`
	WarmCertRejectInvalid      int     `json:"warm_cert_reject_invalid,omitempty"`
	WarmDualIterationCaps      int     `json:"warm_dual_iteration_caps,omitempty"`
	WarmArtificialFailure      int     `json:"warm_artificial_failure,omitempty"`
	WarmColdInfeasible         int     `json:"warm_cold_infeasible,omitempty"`
	WarmCertAccepts            int     `json:"warm_cert_accepts,omitempty"`
	WarmCertVerifies           int     `json:"warm_cert_verifies,omitempty"`
	WarmCertVerifyFails        int     `json:"warm_cert_verify_fails,omitempty"`
	WarmCertBoundCorrections   int     `json:"warm_cert_bound_corrections,omitempty"`
	WarmCertVerifyResumes      int     `json:"warm_cert_verify_resumes,omitempty"`
	WarmCertBoundCorrectionMax float64 `json:"warm_cert_bound_correction_max,omitempty"`
	DualStalls                 int     `json:"dual_stalls,omitempty"`
	DualNoEntering             int     `json:"dual_no_entering,omitempty"`
	DualIterationCaps          int     `json:"dual_iteration_caps,omitempty"`
	SessionReopts              int     `json:"session_reopts,omitempty"`
	SessionFallbacks           int     `json:"session_fallbacks,omitempty"`
	SessionAdoptions           int     `json:"session_adoptions,omitempty"`
	SessionInfeasibleRetries   int     `json:"session_infeasible_retries,omitempty"`
	PhaseIMS                   float64 `json:"phase1_ms,omitempty"`
	PhaseIIMS                  float64 `json:"phase2_ms,omitempty"`
	DualMS                     float64 `json:"dual_ms,omitempty"`
	SparseFactorMS             float64 `json:"sparse_factor_ms,omitempty"`
	DenseFactorMS              float64 `json:"dense_factor_ms,omitempty"`
}

type IncumbentEvent struct {
	ElapsedMS float64 `json:"elapsed_ms"`
	Node      int     `json:"node"`
	Objective float64 `json:"objective"`
	Source    string  `json:"source,omitempty"`
}

type Results struct {
	Solver  string `json:"solver"`
	Version string `json:"version,omitempty"`
	Items   []Item `json:"items"`
}

func Load(path string) (*Problem, error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var p Problem
	if err := json.Unmarshal(raw, &p); err != nil {
		return nil, fmt.Errorf("%s: %w", path, err)
	}
	return &p, nil
}

func LoadDir(dir string) ([]*Problem, error) {
	entries, err := os.ReadDir(dir)
	if err != nil {
		return nil, err
	}
	var paths []string
	for _, e := range entries {
		if e.IsDir() || filepath.Ext(e.Name()) != ".json" {
			continue
		}
		paths = append(paths, filepath.Join(dir, e.Name()))
	}
	sort.Strings(paths)
	out := make([]*Problem, 0, len(paths))
	for _, p := range paths {
		prob, err := Load(p)
		if err != nil {
			return nil, err
		}
		out = append(out, prob)
	}
	return out, nil
}
