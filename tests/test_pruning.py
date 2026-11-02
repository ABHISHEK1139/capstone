"""Comprehensive test suite for resource bounds, memory governor, and pruning policies.

Validates:
1. PruningPolicy parameter validation and boundary condition enforcement.
2. Resource budget matrix: (candidates, time, memory) x (raise, degrade, truncate).
3. Memory governor: continuous background sampling (MemorySampler), peak tracking, dynamic memory growth, and level_halted >= 1.
4. Combinatorial rule-generation watchdog on large itemsets (k >= 14, 2^14 - 2 = 16,382 partitions).
5. Search-topology-specific candidate budget semantics across Apriori, FP-Growth, and Eclat.
"""

import time
import pickle
import itertools
import pytest
import numpy as np
import pandas as pd

from skarm.pruning import (
    PruningPolicy,
    MemorySampler,
    CandidateExplosionError,
    BoundaryConditionError,
    DegradedModeWarning,
    get_current_memory_mb,
)
from skarm.estimator import AssociationRulesEstimator
from skarm.rules import association_rules
from skarm.apriori import apriori
from skarm.fpgrowth import fpgrowth
from skarm.eclat import eclat


# =====================================================================
# 1. Parameter Validation & Boundary Conditions
# =====================================================================

def test_pruning_policy_parameter_validation():
    """Verify input validation for PruningPolicy construction."""
    # min_support out of bounds
    with pytest.raises(BoundaryConditionError, match="min_support"):
        PruningPolicy(min_support=-0.1)
    with pytest.raises(BoundaryConditionError, match="min_support"):
        PruningPolicy(min_support=1.5)

    # max_candidates non-positive
    with pytest.raises(BoundaryConditionError, match="max_candidates"):
        PruningPolicy(max_candidates=0)
    with pytest.raises(BoundaryConditionError, match="max_candidates"):
        PruningPolicy(max_candidates=-10)

    # max_len non-positive
    with pytest.raises(BoundaryConditionError, match="max_len"):
        PruningPolicy(max_len=0)

    # max_time_seconds non-positive
    with pytest.raises(BoundaryConditionError, match="max_time_seconds"):
        PruningPolicy(max_time_seconds=0.0)
    with pytest.raises(BoundaryConditionError, match="max_time_seconds"):
        PruningPolicy(max_time_seconds=-1.0)

    # max_memory_mb non-positive
    with pytest.raises(BoundaryConditionError, match="max_memory_mb"):
        PruningPolicy(max_memory_mb=0.0)
    with pytest.raises(BoundaryConditionError, match="max_memory_mb"):
        PruningPolicy(max_memory_mb=-100.0)

    # invalid on_candidate_explosion
    with pytest.raises(ValueError, match="Unknown on_candidate_explosion"):
        PruningPolicy(on_candidate_explosion="invalid_policy")

    # Valid boundary parameters (min_support 0.0 and 1.0 are valid)
    p0 = PruningPolicy(min_support=0.0, max_candidates=10, max_len=1)
    assert p0.min_support == 0.0
    p1 = PruningPolicy(min_support=1.0, max_candidates=10, max_len=1)
    assert p1.min_support == 1.0


# =====================================================================
# 2. Candidate Budget Policy Matrix (raise, degrade, truncate)
# =====================================================================

def test_candidate_budget_policy_matrix():
    """Verify check_candidate_count and check_budget across all three explosion policies."""
    # 1. 'raise' policy: raises immediately on candidate explosion
    policy_raise = PruningPolicy(max_candidates=50, on_candidate_explosion="raise")
    assert policy_raise.check_candidate_count(30, level=2) is True
    assert policy_raise.peak_candidate_count == 30
    with pytest.raises(CandidateExplosionError, match="Candidate explosion detected at level 3"):
        policy_raise.check_candidate_count(51, level=3)
    assert policy_raise.level_halted == 3
    assert policy_raise.candidates_observed_before_halt == 51

    # 2. 'degrade' policy: warns, halts further generation (returns False), marks degraded
    policy_degrade = PruningPolicy(max_candidates=50, on_candidate_explosion="degrade")
    with pytest.warns(DegradedModeWarning, match="Candidate explosion detected"):
        can_proceed = policy_degrade.check_candidate_count(75, level=2)
    assert can_proceed is False
    assert policy_degrade.is_degraded is True
    assert policy_degrade.level_halted == 2
    assert policy_degrade.candidates_observed_before_halt == 75
    assert policy_degrade.excess_candidates_observed == 25
    assert "Candidate explosion detected" in policy_degrade.degradation_reason

    # 3. 'truncate' policy: warns, allows continuing (returns True), marks degraded
    policy_trunc = PruningPolicy(max_candidates=50, on_candidate_explosion="truncate")
    with pytest.warns(DegradedModeWarning, match="Truncating candidates"):
        can_proceed = policy_trunc.check_candidate_count(80, level=2)
    assert can_proceed is True
    assert policy_trunc.is_degraded is True
    assert policy_trunc.candidates_remaining_at_level == 30


# =====================================================================
# 3. Time Budget Enforcement Matrix
# =====================================================================

def test_time_budget_enforcement_matrix():
    """Verify time budget enforcement under 'raise' and 'degrade'."""
    # Under 'raise'
    policy_raise = PruningPolicy(max_time_seconds=0.01, on_candidate_explosion="raise")
    time.sleep(0.02)
    with pytest.raises(CandidateExplosionError, match="Time budget exceeded"):
        policy_raise.check_budget(level=2)

    # Under 'degrade'
    policy_degrade = PruningPolicy(max_time_seconds=0.01, on_candidate_explosion="degrade")
    time.sleep(0.02)
    with pytest.warns(DegradedModeWarning, match="Time budget exceeded"):
        ok = policy_degrade.check_budget(level=3)
    assert ok is False
    assert policy_degrade.is_degraded is True
    assert policy_degrade.level_halted == 3
    assert "Time budget exceeded" in policy_degrade.degradation_reason


# =====================================================================
# 4. Memory Governor & Background Sampler Mechanics
# =====================================================================

def test_memory_sampler_lifecycle_and_serialization():
    """Verify MemorySampler lifecycle, thread safety, and PruningPolicy serialization."""
    policy = PruningPolicy(max_memory_mb=1000.0)
    assert policy.sampler is None

    # Start background sampler
    policy.start_sampler(polling_interval=0.01)
    assert policy.sampler is not None
    assert policy.sampler.is_alive()
    assert policy.sampler.peak_mb >= policy.baseline_memory_mb

    # Serialization while sampler is active (sampler thread omitted safely)
    dumped = pickle.dumps(policy)
    loaded = pickle.loads(dumped)
    assert loaded.max_memory_mb == 1000.0
    assert loaded.sampler is None

    # Stop sampler
    final_peak = policy.stop_sampler()
    assert policy.sampler is None
    assert final_peak >= policy.baseline_memory_mb

    # Serialization after sampler is stopped
    dumped2 = pickle.dumps(policy)
    loaded2 = pickle.loads(dumped2)
    assert loaded2.peak_memory_mb == final_peak


def test_memory_governor_budget_enforcement_direct():
    """Verify check_budget memory limits under raise and degrade directly on policy."""
    base = get_current_memory_mb()

    # Budget respected (generous headroom)
    policy_ok = PruningPolicy(max_memory_mb=base + 1024.0)
    assert policy_ok.check_budget(level=1) is True
    assert policy_ok.is_degraded is False

    # Budget exceeded under 'raise' with simulated effective peak
    policy_raise = PruningPolicy(max_memory_mb=base + 1.0, on_candidate_explosion="raise")
    with pytest.raises(CandidateExplosionError, match="Memory budget exceeded"):
        policy_raise.check_budget(level=2, peak_memory_mb=base + 2.0)

    # Budget exceeded under 'degrade' with simulated effective peak
    policy_degrade = PruningPolicy(max_memory_mb=base + 1.0, on_candidate_explosion="degrade")
    with pytest.warns(DegradedModeWarning, match="Memory budget exceeded"):
        ok = policy_degrade.check_budget(level=2, peak_memory_mb=base + 2.0)
    assert ok is False
    assert policy_degrade.is_degraded is True
    assert policy_degrade.level_halted == 2
    assert "Memory budget exceeded" in policy_degrade.degradation_reason


# =====================================================================
# 5. Dynamic Memory Governor Across Mining Engines (level_halted >= 1)
# =====================================================================

def test_dynamic_memory_growth_governor_enforcement():
    """Verify dynamic memory governor with background MemorySampler asserting level_halted >= 1."""
    # Test across multiple search levels (level >= 1)
    for target_level in [1, 2, 3]:
        # 1. Under degrade policy
        base = get_current_memory_mb()
        policy_degrade = PruningPolicy(
            max_memory_mb=base + 5.0,
            on_candidate_explosion="degrade",
        )
        policy_degrade.start_sampler(polling_interval=0.005)

        # Realistic dynamic memory allocation during search at target_level
        # 30 MB array allocation bypasses pymalloc and requests OS pages
        dynamic_alloc = np.ones((2000, 2000), dtype=np.float64)
        time.sleep(0.02)  # allow sampler to record peak
        peak = policy_degrade.stop_sampler()
        assert peak > base

        with pytest.warns(DegradedModeWarning, match="Memory budget exceeded"):
            can_proceed = policy_degrade.check_budget(level=target_level)

        assert can_proceed is False
        assert policy_degrade.is_degraded is True
        assert policy_degrade.level_halted == target_level
        assert policy_degrade.level_halted >= 1
        assert "Memory budget exceeded" in policy_degrade.degradation_reason
        del dynamic_alloc

        # 2. Under raise policy
        base_raise = get_current_memory_mb()
        policy_raise = PruningPolicy(
            max_memory_mb=base_raise + 5.0,
            on_candidate_explosion="raise",
        )
        policy_raise.start_sampler(polling_interval=0.005)
        dynamic_alloc_raise = np.ones((2000, 2000), dtype=np.float64)
        time.sleep(0.02)
        policy_raise.stop_sampler()

        with pytest.raises(CandidateExplosionError, match="Memory budget exceeded"):
            policy_raise.check_budget(level=target_level)
        assert policy_raise.level_halted == target_level
        assert policy_raise.level_halted >= 1
        del dynamic_alloc_raise


# =====================================================================
# 6. Combinatorial Rule-Generation Watchdog (k >= 14, 2^14 - 2 partitions)
# =====================================================================

def test_combinatorial_rule_generation_watchdog_large_itemset():
    """Verify rule-generation watchdog halts combinatorial explosion on large itemsets (k=14, 16,382 partitions)."""
    k = 14
    items = list(range(k))
    full = frozenset(items)

    # 1-itemsets, frequent pairs, and a large 14-itemset that generates 2^14 - 2 = 16,382 candidate rule partitions
    large_itemsets = {frozenset([i]): 0.8 for i in range(k)}
    for i in range(k):
        for j in range(i + 1, min(k, i + 3)):
            large_itemsets[frozenset([i, j])] = 0.7
    large_itemsets[full] = 0.6

    # 1. Time watchdog with degrade policy: halts during expansion, returns partial rules
    policy_degrade = PruningPolicy(max_time_seconds=0.005, on_candidate_explosion="degrade")
    with pytest.warns(DegradedModeWarning, match="Time budget exceeded"):
        rules_df = association_rules(
            large_itemsets,
            metric="confidence",
            min_threshold=0.5,
            pruning_policy=policy_degrade,
        )
    assert policy_degrade.is_degraded is True
    assert "Time budget exceeded" in policy_degrade.degradation_reason
    assert len(rules_df) < 16382
    assert len(rules_df) > 0  # Produced initial rules before watchdog halted

    # 2. Time watchdog with raise policy: raises CandidateExplosionError during expansion
    policy_raise = PruningPolicy(max_time_seconds=0.005, on_candidate_explosion="raise")
    with pytest.raises(CandidateExplosionError, match="Time budget exceeded"):
        association_rules(
            large_itemsets,
            metric="confidence",
            min_threshold=0.5,
            pruning_policy=policy_raise,
        )

    # 3. Non-vacuous candidate budget enforcement during rule generation
    # Realistic 5-item lattice generating 150 rules unconstrained
    lattice_itemsets = {
        frozenset(c): round(0.9 - 0.15 * len(c), 3)
        for r in range(1, 5)
        for c in itertools.combinations(range(5), r)
    }
    rules_unconstrained = association_rules(lattice_itemsets, metric="confidence", min_threshold=0.1)
    total_unconstrained = len(rules_unconstrained)
    assert total_unconstrained == 150  # Verifies realistic unconstrained baseline (50-150 rules)

    # 3a. Candidate limit with truncate policy: caps rules strictly to max_candidates, emits warning, marks degraded
    policy_trunc = PruningPolicy(max_candidates=20, on_candidate_explosion="truncate")
    with pytest.warns(DegradedModeWarning, match="Truncating candidates to max_candidates limit"):
        rules_trunc = association_rules(
            lattice_itemsets,
            metric="confidence",
            min_threshold=0.1,
            pruning_policy=policy_trunc,
        )
    assert policy_trunc.is_degraded is True
    assert 0 < len(rules_trunc) <= 20
    assert len(rules_trunc) < total_unconstrained

    # 3b. Candidate limit with degrade policy: halts generation, returns partial rules <= max_candidates, marks degraded
    policy_deg = PruningPolicy(max_candidates=20, on_candidate_explosion="degrade")
    with pytest.warns(DegradedModeWarning, match="Entering degraded mode"):
        rules_deg = association_rules(
            lattice_itemsets,
            metric="confidence",
            min_threshold=0.1,
            pruning_policy=policy_deg,
        )
    assert policy_deg.is_degraded is True
    assert 0 < len(rules_deg) <= 20
    assert len(rules_deg) < total_unconstrained

    # 3c. Candidate limit with raise policy: raises CandidateExplosionError
    policy_raise_cand = PruningPolicy(max_candidates=20, on_candidate_explosion="raise")
    with pytest.raises(CandidateExplosionError):
        association_rules(
            lattice_itemsets,
            metric="confidence",
            min_threshold=0.1,
            pruning_policy=policy_raise_cand,
        )

    # 4. Memory watchdog with raise policy: raises CandidateExplosionError during expansion
    base = get_current_memory_mb()
    policy_mem_raise = PruningPolicy(max_memory_mb=base - 1.0, on_candidate_explosion="raise")
    with pytest.raises(CandidateExplosionError, match="Memory budget exceeded"):
        association_rules(
            large_itemsets,
            metric="confidence",
            min_threshold=0.5,
            pruning_policy=policy_mem_raise,
        )

    # 5. Memory watchdog with degrade policy: enters degraded mode during expansion
    policy_mem_deg = PruningPolicy(max_memory_mb=base - 1.0, on_candidate_explosion="degrade")
    with pytest.warns(DegradedModeWarning, match="Memory budget exceeded"):
        rules_mem_deg = association_rules(
            large_itemsets,
            metric="confidence",
            min_threshold=0.5,
            pruning_policy=policy_mem_deg,
        )
    assert policy_mem_deg.is_degraded is True


# =====================================================================
# 7. Topology-Specific Candidate Budget Semantics
# =====================================================================

def test_topology_candidate_budget_semantics():
    """Verify max_candidates budget semantics across Apriori, FP-Growth, and Eclat search topologies."""
    # Dense matrix with 15 items: generates many combinations
    rng = np.random.RandomState(42)
    X = rng.binomial(1, 0.8, size=(100, 15))

    # Apriori: bounds intermediate unverified level-k candidate joins |C_k|
    est_ap = AssociationRulesEstimator(
        algorithm="apriori",
        min_support=0.3,
        max_candidates=100,
        on_candidate_explosion="degrade",
    )
    with pytest.warns(DegradedModeWarning, match="Candidate explosion detected"):
        est_ap.fit(X)
    assert est_ap.is_degraded_ is True
    assert est_ap.metadata_["level_halted"] >= 1
    assert "Candidate explosion detected" in est_ap.termination_reason_

    # FP-Growth: bounds cumulative discovered frequent itemsets |F| from conditional pattern trees
    est_fp = AssociationRulesEstimator(
        algorithm="fpgrowth",
        min_support=0.3,
        max_candidates=100,
        on_candidate_explosion="degrade",
    )
    with pytest.warns(DegradedModeWarning, match="Candidate explosion detected"):
        est_fp.fit(X)
    assert est_fp.is_degraded_ is True
    assert est_fp.metadata_["level_halted"] >= 1
    assert "Candidate explosion detected" in est_fp.termination_reason_

    # Eclat: bounds cumulative discovered frequent itemsets |F| from vertical TID intersections
    est_ec = AssociationRulesEstimator(
        algorithm="eclat",
        min_support=0.3,
        max_candidates=100,
        on_candidate_explosion="degrade",
    )
    with pytest.warns(DegradedModeWarning, match="Candidate explosion detected"):
        est_ec.fit(X)
    assert est_ec.is_degraded_ is True
    assert est_ec.metadata_["level_halted"] >= 1
    assert "Candidate explosion detected" in est_ec.termination_reason_
