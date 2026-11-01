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


import threading


def get_current_memory_mb() -> float:
    """Return the current process RSS memory in megabytes."""
    try:
        import psutil
        import os
        return psutil.Process(os.getpid()).memory_info().rss / (1024.0 * 1024.0)
    except Exception:
        import tracemalloc
        if tracemalloc.is_tracing():
            _, peak = tracemalloc.get_traced_memory()
            return peak / (1024.0 * 1024.0)
        return 0.0


class MemorySampler(threading.Thread):
    """Continuous background RSS memory sampler (10 ms polling frequency = 100 Hz).

    Samples process RSS every `polling_interval` seconds (default 10 ms = 0.01 s)
    to accurately record peak memory consumption during fit operations.
    """

    def __init__(self, polling_interval: float = 0.01):
        super().__init__(daemon=True)
        self.polling_interval = polling_interval
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        self.baseline_mb: float = get_current_memory_mb()
        self._peak_mb: float = self.baseline_mb

    @property
    def peak_mb(self) -> float:
        with self._lock:
            return self._peak_mb

    @peak_mb.setter
    def peak_mb(self, val: float):
        with self._lock:
            if val > self._peak_mb:
                self._peak_mb = val

    def run(self):
        while not self._stop_event.is_set():
            try:
                curr = get_current_memory_mb()
                with self._lock:
                    if curr > self._peak_mb:
                        self._peak_mb = curr
            except Exception:
                pass
            self._stop_event.wait(self.polling_interval)

    def stop(self) -> float:
        """Stop sampling and return final peak RSS in MB."""
        self._stop_event.set()
        try:
            curr = get_current_memory_mb()
            with self._lock:
                if curr > self._peak_mb:
                    self._peak_mb = curr
        except Exception:
            pass
        self.join(timeout=1.0)
        with self._lock:
            return self._peak_mb


class PruningPolicy:
    """Manages candidate generation constraints and safe degradation.

    Protects memory and computational resources against exponential candidate explosion
    inherent in dense transactional datasets (NT-1, NT-6).

    Parameters
    ----------
    max_candidates : int, default=100000
        Maximum allowed candidates at any single generation step (Apriori) or
        discovered pattern budget (FP-Growth / Eclat).
    max_len : int or None, default=None
        Maximum itemset length to discover.
    min_support : float, default=0.1
        Minimum support threshold. Must be in [0.0, 1.0] (0.0 is allowed
        by design and mines every subset; see NT-4).
    on_candidate_explosion : {'raise', 'truncate', 'degrade'}, default='degrade'
        Action when candidate count exceeds max_candidates:
        - 'raise': Raise CandidateExplosionError immediately.
        - 'truncate': Retain first max_candidates via deterministic enumeration and continue.
        - 'degrade': Mark state as degraded, log telemetry, halt further candidate generation.
    max_time_seconds : float or None, default=None
        Maximum total fit time in seconds covering preprocessing, indexing, and mining.
    max_memory_mb : float or None, default=None
        Maximum peak process RSS memory limit in megabytes.
    start_time : float or None, default=None
        Authoritative start timestamp (perf_counter) for the complete fit process.
    baseline_memory_mb : float or None, default=None
        Pre-fit process RSS baseline in megabytes.
    """

    def __init__(
        self,
        max_candidates: int = 100000,
        max_len: Optional[int] = None,
        min_support: float = 0.1,
        on_candidate_explosion: str = "degrade",
        max_time_seconds: Optional[float] = None,
        max_memory_mb: Optional[float] = None,
        start_time: Optional[float] = None,
        baseline_memory_mb: Optional[float] = None,
    ):
        if not (0.0 <= min_support <= 1.0):
            raise BoundaryConditionError(f"min_support must be between 0.0 and 1.0, got {min_support}")
        if max_candidates <= 0:
            raise BoundaryConditionError(f"max_candidates must be positive, got {max_candidates}")
        if max_len is not None and max_len < 1:
            raise BoundaryConditionError(f"max_len must be >= 1, got {max_len}")
        if max_time_seconds is not None and max_time_seconds <= 0:
            raise BoundaryConditionError(f"max_time_seconds must be positive, got {max_time_seconds}")
        if max_memory_mb is not None and max_memory_mb <= 0:
            raise BoundaryConditionError(f"max_memory_mb must be positive, got {max_memory_mb}")
        if on_candidate_explosion not in ("raise", "truncate", "degrade"):
            raise ValueError(f"Unknown on_candidate_explosion policy: {on_candidate_explosion}")

        self.max_candidates = max_candidates
        self.max_len = max_len
        self.min_support = min_support
        self.on_candidate_explosion = on_candidate_explosion
        self.max_time_seconds = max_time_seconds
        self.max_memory_mb = max_memory_mb
        self.is_degraded = False
        self.degradation_reason = None
        self.peak_candidate_count = 0
        self.level_halted: Optional[int] = None

        # Truncation and audit metadata
        self.candidates_observed_before_halt: Optional[int] = None
        self.excess_candidates_observed: Optional[int] = None
        self.unexplored_candidates_count: Optional[int] = None
        self.future_candidates_status: str = "EXHAUSTED"

        # Backwards compatibility aliases
        self.candidates_generated_at_level: Optional[int] = None
        self.candidates_remaining_at_level: Optional[int] = None

        import time as _time
        self.start_time: float = start_time if start_time is not None else _time.perf_counter()
        self.baseline_memory_mb: float = (
            baseline_memory_mb if baseline_memory_mb is not None else get_current_memory_mb()
        )
        self.peak_memory_mb: float = self.baseline_memory_mb
        self.sampler: Optional[MemorySampler] = None

    def start_sampler(self, polling_interval: float = 0.01):
        """Start continuous background 10 ms memory sampling."""
        if self.sampler is None:
            self.sampler = MemorySampler(polling_interval=polling_interval)
            self.sampler.start()

    def stop_sampler(self) -> float:
        """Stop background sampler and capture peak memory."""
        if self.sampler is not None:
            peak = self.sampler.stop()
            self.sampler = None
            if peak > self.peak_memory_mb:
                self.peak_memory_mb = peak
        curr = get_current_memory_mb()
        if curr > self.peak_memory_mb:
            self.peak_memory_mb = curr
        return self.peak_memory_mb

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
            self.level_halted = level
            self.candidates_observed_before_halt = candidate_count
            self.excess_candidates_observed = max(0, candidate_count - self.max_candidates)
            self.unexplored_candidates_count = None
            self.future_candidates_status = "UNKNOWN"

            # Backwards compatibility aliases
            self.candidates_generated_at_level = candidate_count

            if self.on_candidate_explosion == "raise":
                raise CandidateExplosionError(msg)
            elif self.on_candidate_explosion == "degrade":
                self.is_degraded = True
                self.degradation_reason = msg
                self.candidates_remaining_at_level = candidate_count
                warnings.warn(msg + " Entering degraded mode (halting deeper candidate generation).", DegradedModeWarning)
                return False
            elif self.on_candidate_explosion == "truncate":
                self.is_degraded = True
                self.degradation_reason = msg
                self.candidates_remaining_at_level = candidate_count - self.max_candidates
                warnings.warn(msg + " Truncating candidates to max_candidates limit.", DegradedModeWarning)
                return True

        return True

    def check_budget(
        self,
        candidate_count: int = 0,
        level: int = 0,
        elapsed_time: Optional[float] = None,
        peak_memory_mb: Optional[float] = None,
    ) -> bool:
        """Check multi-dimensional resource budget (candidates, time, memory)."""
        import time as _time

        if candidate_count > 0:
            if not self.check_candidate_count(candidate_count, level=level):
                return False

        if elapsed_time is None and self.max_time_seconds is not None:
            elapsed_time = _time.perf_counter() - self.start_time

        if self.max_time_seconds is not None and elapsed_time is not None:
            if elapsed_time > self.max_time_seconds:
                msg = (
                    f"Time budget exceeded: {elapsed_time:.3f}s exceeds "
                    f"max_time_seconds={self.max_time_seconds}s."
                )
                self.level_halted = level
                self.future_candidates_status = "UNKNOWN"
                if self.on_candidate_explosion == "raise":
                    raise CandidateExplosionError(msg)
                if not (self.is_degraded and self.degradation_reason and "Time budget exceeded" in str(self.degradation_reason)):
                    warnings.warn(msg + " Entering degraded mode (time limit reached).", DegradedModeWarning)
                self.is_degraded = True
                self.degradation_reason = msg
                return False

        # Memory Governor: poll sampler or current RSS
        if self.sampler is not None:
            sampled_peak = self.sampler.peak_mb
            if sampled_peak > self.peak_memory_mb:
                self.peak_memory_mb = sampled_peak
        else:
            curr_mem = get_current_memory_mb()
            if curr_mem > self.peak_memory_mb:
                self.peak_memory_mb = curr_mem

        effective_mem = peak_memory_mb if peak_memory_mb is not None else self.peak_memory_mb
        if effective_mem > self.peak_memory_mb:
            self.peak_memory_mb = effective_mem

        if self.max_memory_mb is not None:
            if effective_mem > self.max_memory_mb:
                msg = (
                    f"Memory budget exceeded: {effective_mem:.1f}MB exceeds "
                    f"max_memory_mb={self.max_memory_mb}MB."
                )
                self.level_halted = level
                self.future_candidates_status = "UNKNOWN"
                if self.on_candidate_explosion == "raise":
                    raise CandidateExplosionError(msg)
                if not (self.is_degraded and self.degradation_reason and "Memory budget exceeded" in str(self.degradation_reason)):
                    warnings.warn(msg + " Entering degraded mode (memory limit reached).", DegradedModeWarning)
                self.is_degraded = True
                self.degradation_reason = msg
                return False

        return True

    def should_continue_length(self, current_len: int) -> bool:
        """Determine whether itemset generation should continue based on max_len."""
        if self.max_len is not None and current_len >= self.max_len:
            return False
        return not self.is_degraded

    def __getstate__(self):
        """Serialize PruningPolicy excluding background thread/lock objects."""
        state = self.__dict__.copy()
        state["sampler"] = None
        return state

    def __setstate__(self, state):
        """Restore PruningPolicy state."""
        self.__dict__.update(state)
        self.sampler = None

    def reset(self):
        """Reset degradation and telemetry state."""
        self.stop_sampler()
        self.is_degraded = False
        self.degradation_reason = None
        self.peak_candidate_count = 0
        self.level_halted = None
        self.candidates_observed_before_halt = None
        self.excess_candidates_observed = None
        self.unexplored_candidates_count = None
        self.candidates_generated_at_level = None
        self.candidates_remaining_at_level = None
        self.future_candidates_status = "EXHAUSTED"
        import time as _time
        self.start_time = _time.perf_counter()
        self.baseline_memory_mb = get_current_memory_mb()
        self.peak_memory_mb = self.baseline_memory_mb
