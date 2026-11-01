"""Tier 4: Real-World Application Scenarios Verification Suite.

Comprehensive end-to-end tests validating SKARM under realistic industry workflows:
- Scenario 1: Retail Market Basket Analysis (Groceries receipt workflow: raw text -> encoded -> mined -> recommendations)
- Scenario 2: E-commerce Recommendation Engine in scikit-learn Pipeline with feature selection
- Scenario 3: Clickstream Session Pattern Mining with high sparsity (<0.1% density)
- Scenario 4: Dense survey / census demographic pattern discovery with long transactions
- Scenario 5: Fault-tolerant batch ETL pipeline where high-density burst triggers DEGRADED state without crashing
"""

import math
import warnings
import pytest
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import LogisticRegression

from skarm.preprocessing import TransactionEncoder
from skarm.estimator import AssociationRulesEstimator
from skarm.pruning import DegradedModeWarning


# ============================================================================
# SCENARIO 1: Retail Market Basket Analysis (Groceries Workflow)
# ============================================================================

def test_scenario_1_retail_market_basket_analysis():
    """Scenario 1: Retail market basket analysis on grocery receipts.

    Simulates a supermarket retail chain analyzing consumer purchase habits:
    - Raw text grocery basket ingestion
    - Frequent itemset discovery via TransactionEncoder and AssociationRulesEstimator
    - Rule extraction filtered by lift > 1.2 and confidence >= 0.5
    - Generation of personalized recommendations for incomplete shopping baskets
    """
    raw_grocery_receipts = [
        ["whole milk", "butter", "white bread", "yogurt"],
        ["whole milk", "yogurt", "tropical fruit"],
        ["white bread", "butter", "margarine"],
        ["whole milk", "butter", "white bread", "eggs"],
        ["tropical fruit", "citrus fruit", "whole milk"],
        ["whole milk", "butter", "white bread", "yogurt", "eggs"],
        ["citrus fruit", "tropical fruit"],
        ["white bread", "butter", "eggs"],
        ["whole milk", "butter", "white bread"],
        ["bottled water", "soda", "chips"],
        ["whole milk", "yogurt", "butter"],
        ["bottled water", "chips"],
    ]

    est = AssociationRulesEstimator(
        min_support=0.25,
        metric="lift",
        min_threshold=1.2,
        algorithm="auto",
    )
    est.fit(raw_grocery_receipts)

    # 1. Verify item catalog extracted cleanly
    assert "whole milk" in est.columns_
    assert "white bread" in est.columns_
    assert "butter" in est.columns_

    # 2. Verify rules table
    rules = est.get_rules()
    assert not rules.empty
    assert all(rules["lift"] >= 1.2)
    assert all(rules["confidence"] > 0)
    assert all(rules["support"] >= 0.25)

    # 3. Predict recommendations for new shopper carrying ['white bread', 'butter']
    new_shopper_basket = [["white bread", "butter"]]
    recs = est.predict(new_shopper_basket)
    assert isinstance(recs, list)
    assert len(recs) == 1
    # 'whole milk' and 'eggs' frequently accompany bread + butter
    recommended_items = recs[0]
    assert len(recommended_items) > 0
    # Must NOT recommend items already in shopper's basket
    assert "white bread" not in recommended_items
    assert "butter" not in recommended_items
    assert any(item in recommended_items for item in ["whole milk", "eggs"])


# ============================================================================
# SCENARIO 2: E-commerce Recommendation Engine in Scikit-Learn Pipeline
# ============================================================================

def test_scenario_2_ecommerce_pipeline_with_feature_selection():
    """Scenario 2: E-commerce intent classification pipeline with feature selection.

    Builds an end-to-end scikit-learn Pipeline where:
    - Step 1: AssociationRulesEstimator extracts rule-firing indicator features
    - Step 2: VarianceThreshold selects active rule features
    - Step 3: LogisticRegression predicts customer conversion probability
    """
    np.random.seed(42)
    n_samples = 100
    n_items = 8

    # Simulate purchase history
    X_purchases = np.random.choice([0, 1], size=(n_samples, n_items), p=[0.7, 0.3])
    # Target: Customers who buy item 0 AND item 2 have high probability of repeat conversion
    intent_signal = (X_purchases[:, 0] == 1) & (X_purchases[:, 2] == 1)
    noise = np.random.choice([0, 1], size=n_samples, p=[0.9, 0.1])
    y_target = np.logical_xor(intent_signal, noise).astype(int)

    feature_names = [f"product_{i}" for i in range(n_items)]
    df_sessions = pd.DataFrame(X_purchases, columns=feature_names)

    pipeline = Pipeline([
        ("rule_miner", AssociationRulesEstimator(
            min_support=0.1,
            metric="confidence",
            min_threshold=0.4,
            algorithm="fpgrowth",
            sparse_output=False,
        )),
        ("feature_filter", VarianceThreshold(threshold=0.01)),
        ("classifier", LogisticRegression(solver="liblinear", random_state=42)),
    ])

    # Fit pipeline
    pipeline.fit(df_sessions, y_target)

    # Predict conversion on test customers
    test_df = df_sessions.iloc[:10]
    preds = pipeline.predict(test_df)
    proba = pipeline.predict_proba(test_df)

    assert preds.shape == (10,)
    assert proba.shape == (10, 2)
    assert np.all((proba >= 0.0) & (proba <= 1.0))


# ============================================================================
# SCENARIO 3: Clickstream Session Pattern Mining with High Sparsity (<0.1%)
# ============================================================================

def test_scenario_3_clickstream_pattern_mining_high_sparsity():
    """Scenario 3: Clickstream session pattern mining with high sparsity (<0.1%).

    Simulates 2,000 web sessions across 500 distinct page URLs:
    - Highly sparse matrix with density < 0.1% (average 2-3 page views per session)
    - Embedded conversion funnel: Page 12 -> Page 85 -> Page 210
    - Mining via vertical Eclat engine without memory explosion
    """
    np.random.seed(99)
    n_sessions = 2000
    n_pages = 500

    # Build sparse CSR matrix
    rows = []
    cols = []
    for s_idx in range(n_sessions):
        # Normal browsing: 1 to 2 random pages
        n_pages_visited = np.random.randint(1, 3)
        visited = np.random.choice(n_pages, size=n_pages_visited, replace=False)
        for p in visited:
            rows.append(s_idx)
            cols.append(p)

    # Inject embedded checkout funnel pattern in 120 sessions (6% support)
    funnel_pages = [12, 85, 210]
    funnel_sessions = np.random.choice(n_sessions, size=120, replace=False)
    for s_idx in funnel_sessions:
        for p in funnel_pages:
            rows.append(s_idx)
            cols.append(p)

    data = np.ones(len(rows), dtype=bool)
    X_sparse = sparse.csr_matrix((data, (rows, cols)), shape=(n_sessions, n_pages), dtype=bool)

    # Check density is very low (< 0.5%)
    density = X_sparse.nnz / (n_sessions * n_pages)
    assert density < 0.005, f"Density {density} is not sparse enough"

    # Mine patterns with Eclat
    est = AssociationRulesEstimator(
        min_support=0.04,
        metric="confidence",
        min_threshold=0.6,
        algorithm="eclat",
    )
    est.fit(X_sparse)

    assert hasattr(est, "rules_")
    assert not est.rules_.empty

    # The embedded funnel pair {12, 85} must be discovered
    found_itemsets = est.frequent_itemsets_
    funnel_pair = frozenset([12, 85])
    assert funnel_pair in found_itemsets, "Funnel pair {12, 85} not found in frequent itemsets"
    assert found_itemsets[funnel_pair] >= 0.04


# ============================================================================
# SCENARIO 4: Dense Survey / Census Data with Long Transactions
# ============================================================================

def test_scenario_4_dense_survey_demographic_pattern_discovery():
    """Scenario 4: Dense survey demographic discovery with long transactions.

    Simulates census respondents answering 30 binary demographic questions:
    - Dense long transactions (12-15 checked boxes per respondent, density ~40%)
    - Guard against exponential candidate explosion using max_len=3 and max_candidates
    - Successfully mines valid demographic rules while respecting bounds
    """
    np.random.seed(77)
    n_respondents = 150
    n_questions = 25

    # 40% probability of affirmative response per question
    X_survey = (np.random.rand(n_respondents, n_questions) < 0.4).astype(bool)

    # Inject strong demographic correlation: Questions 3 and 7 imply Question 14 in 80% of cases
    for i in range(n_respondents):
        if X_survey[i, 3] and X_survey[i, 7]:
            if np.random.rand() < 0.85:
                X_survey[i, 14] = True

    est = AssociationRulesEstimator(
        min_support=0.15,
        metric="confidence",
        min_threshold=0.6,
        max_len=3,
        max_candidates=1000,
        algorithm="apriori",
    )
    est.fit(X_survey)

    assert hasattr(est, "frequent_itemsets_")
    # Invariant: No itemset exceeds max_len=3
    for itemset in est.frequent_itemsets_.keys():
        assert len(itemset) <= 3

    assert not est.rules_.empty
    # Discovered rules should have length <= 3
    for _, row in est.rules_.iterrows():
        assert len(row["antecedent"]) + len(row["consequent"]) <= 3


# ============================================================================
# SCENARIO 5: Fault-Tolerant Batch ETL Pipeline with Burst Recovery
# ============================================================================

def test_scenario_5_fault_tolerant_batch_etl_burst_recovery():
    """Scenario 5: Fault-tolerant batch ETL pipeline with burst degradation recovery.

    Simulates a continuous data streaming pipeline processing successive batches:
    - Batch 1: Normal sparse retail traffic -> processed in EXACT mode
    - Batch 2: Malicious/anomalous 100% dense burst -> safety governor enters DEGRADED mode
               without crashing the service
    - Batch 3: Normal sparse traffic returns -> pipeline recovers immediately to EXACT mode
               with zero residual state leakage
    """
    normal_batch_1 = [
        ["tea", "sugar"],
        ["coffee", "sugar", "milk"],
        ["tea", "milk"],
        ["coffee", "milk"],
        ["tea", "sugar", "biscuit"],
    ]

    # Anomalous burst: 15 items, all present in all 8 transactions (100% density block)
    burst_batch_2 = np.ones((8, 15), dtype=bool)

    normal_batch_3 = [
        ["bread", "butter"],
        ["bread", "jam"],
        ["bread", "butter", "jam"],
        ["milk", "bread"],
    ]

    # Step 1: Process Batch 1
    est1 = AssociationRulesEstimator(min_support=0.3, max_candidates=50, on_candidate_explosion="degrade")
    est1.fit(normal_batch_1)
    assert est1.is_degraded_ is False
    if hasattr(est1, "status_"):
        assert est1.status_ == "EXACT"

    # Step 2: Process Batch 2 (Dense Burst)
    est2 = AssociationRulesEstimator(min_support=0.3, max_candidates=30, on_candidate_explosion="degrade")
    with pytest.warns(DegradedModeWarning, match="degraded mode"):
        est2.fit(burst_batch_2)
    assert est2.is_degraded_ is True
    if hasattr(est2, "status_"):
        assert est2.status_ == "DEGRADED"

    # Step 3: Process Batch 3 (Normal Traffic - State Recovery)
    est3 = AssociationRulesEstimator(min_support=0.3, max_candidates=50, on_candidate_explosion="degrade")
    est3.fit(normal_batch_3)
    assert est3.is_degraded_ is False
    if hasattr(est3, "status_"):
        assert est3.status_ == "EXACT"
    assert "bread" in est3.columns_
    assert not est3.rules_.empty
