"""skarm: Scalable Association-Rule Mining Estimator with Apriori Baseline.

KL Capstone Project Framework: KLCAP-2026-00340
Specialization: AI for Computational Intelligence
"""

from skarm.preprocessing import TransactionEncoder
from skarm.apriori import apriori
from skarm.fpgrowth import fpgrowth
from skarm.eclat import eclat
from skarm.rules import (
    association_rules,
    compute_leverage,
    compute_conviction,
    compute_zhangs_metric,
)
from skarm.pruning import (
    PruningPolicy,
    CandidateExplosionError,
    BoundaryConditionError,
    DegradedModeWarning,
)
from skarm.telemetry import TelemetryLogger, RecoveryCoordinator
from skarm.estimator import AssociationRulesEstimator
from skarm.explorer import RuleExplorer

__version__ = "0.1.0"

__all__ = [
    "TransactionEncoder",
    "apriori",
    "fpgrowth",
    "eclat",
    "association_rules",
    "compute_leverage",
    "compute_conviction",
    "compute_zhangs_metric",
    "PruningPolicy",
    "CandidateExplosionError",
    "BoundaryConditionError",
    "DegradedModeWarning",
    "TelemetryLogger",
    "RecoveryCoordinator",
    "AssociationRulesEstimator",
    "RuleExplorer",
]
