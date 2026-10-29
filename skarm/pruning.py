"""Pruning policies, candidate bounds, and safety guardrails."""

import warnings
from typing import Optional, Set, FrozenSet, Dict, Any


class CandidateExplosionError(RuntimeError):
    """Raised when candidate generation exceeds the safe bound."""
    pass


class BoundaryConditionError(ValueError):
    """Raised on invalid or out-of-bounds parameters."""
    pass


class DegradedModeWarning(UserWarning):
    """Warns that the estimator is running in degraded or constrained mode."""
    pass


class PruningPolicy:
    """Manages candidate generation constraints and safe degradation.

    Protects memory and computational resources against exponential candidate explosion
    inherent in dense transactional datasets (NT-1, NT-6).

    Parameters
    ----------
    max_candidates : int, default=100000
        Maximum allowed candidates at any single generation step.
    max_len : int or None, default=None
        Maximum itemset length to discover.
    min_support : float, default=0.1
        Minimum support threshold. Must be in (0.0, 1.0].
    on_candidate_explosion : {'raise', 'truncate', 'degrade'}, default='degrade'
        Action when candidate count exceeds max_candidates:
        - 'raise': Raise CandidateExplosionError immediately.
        - 'truncate': Prune low-heuristic candidates and continue.
        - 'degrade': Mark state as degraded, log telemetry, halt further candidate generation.
    """

    def __init__(
        self,
        max_candidates: int = 100000,
        max_len: Optional[int] = None,
        min_support: float = 0.1,
        on_candidate_explosion: str = "degrade",
    ):
        if not (0.0 <= min_support <= 1.0):
            raise BoundaryConditionError(f"min_support must be between 0.0 and 1.0, got {min_support}")
        if max_candidates <= 0:
            raise BoundaryConditionError(f"max_candidates must be positive, got {max_candidates}")
        if max_len is not None and max_len < 1:
            raise BoundaryConditionError(f"max_len must be >= 1, got {max_len}")
        if on_candidate_explosion not in ("raise", "truncate", "degrade"):
            raise ValueError(f"Unknown on_candidate_explosion policy: {on_candidate_explosion}")

        self.max_candidates = max_candidates
        self.max_len = max_len
        self.min_support = min_support
        self.on_candidate_explosion = on_candidate_explosion
        self.is_degraded = False
        self.degradation_reason = None
        self.peak_candidate_count = 0

    def check_candidate_count(self, candidate_count: int, level: int) -> bool:
        """Check if candidate count exceeds safety guardrails.

        Returns True if mining can proceed normally, False if generation should halt.
        """
        if candidate_count > self.peak_candidate_count:
            self.peak_candidate_count = candidate_count

        if candidate_count > self.max_candidates:
            msg = (
                f"Candidate explosion detected at level {level}: {candidate_count} candidates "
                f"exceeds max_candidates={self.max_candidates}."
            )
            if self.on_candidate_explosion == "raise":
                raise CandidateExplosionError(msg)
            elif self.on_candidate_explosion == "degrade":
                self.is_degraded = True
                self.degradation_reason = msg
                warnings.warn(msg + " Entering degraded mode (halting deeper candidate generation).", DegradedModeWarning)
                return False
            elif self.on_candidate_explosion == "truncate":
                warnings.warn(msg + " Truncating candidates to max_candidates limit.", DegradedModeWarning)
                return True

        return True

    def should_continue_length(self, current_len: int) -> bool:
        """Determine whether itemset generation should continue based on max_len."""
        if self.max_len is not None and current_len >= self.max_len:
            return False
        return not self.is_degraded

    def reset(self):
        """Reset degradation and telemetry state."""
        self.is_degraded = False
        self.degradation_reason = None
        self.peak_candidate_count = 0
