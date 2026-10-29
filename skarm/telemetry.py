"""Telemetry, memory profiling, and recovery coordination."""

import time
import tracemalloc
from typing import Dict, Any, List, Optional


class TelemetryLogger:
    """Collects runtime telemetry, memory metrics, and execution diagnostics."""

    def __init__(self):
        self.metrics: Dict[str, Any] = {}
        self.level_metrics: List[Dict[str, Any]] = []
        self._start_time: Optional[float] = None
        self._start_mem: int = 0

    def start(self):
        """Start recording wall-clock time and peak memory."""
        self.metrics.clear()
        self.level_metrics.clear()
        if not tracemalloc.is_tracing():
            tracemalloc.start()
        tracemalloc.reset_peak()
        self._start_time = time.perf_counter()

    def record_level(self, level: int, candidate_count: int, frequent_count: int, elapsed_level: float):
        """Record diagnostic details for a specific itemset size level."""
        current_mem, peak_mem = tracemalloc.get_traced_memory()
        self.level_metrics.append({
            "level": level,
            "candidates": candidate_count,
            "frequent_itemsets": frequent_count,
            "duration_s": elapsed_level,
            "peak_mem_bytes": peak_mem,
            "peak_mem_mb": peak_mem / (1024 * 1024),
        })

    def stop(self, extra_info: Optional[Dict[str, Any]] = None):
        """Stop tracking and finalize metrics."""
        elapsed = time.perf_counter() - (self._start_time or time.perf_counter())
        current_mem, peak_mem = tracemalloc.get_traced_memory()
        self.metrics.update({
            "elapsed_seconds": elapsed,
            "peak_memory_bytes": peak_mem,
            "peak_memory_mb": peak_mem / (1024 * 1024),
            "level_breakdown": list(self.level_metrics),
        })
        if extra_info:
            self.metrics.update(extra_info)

    def get_summary(self) -> Dict[str, Any]:
        """Return the collected telemetry summary."""
        return dict(self.metrics)


class RecoveryCoordinator:
    """Manages negative testing events, degraded mode recovery, and audit trails."""

    def __init__(self):
        self.audit_log: List[Dict[str, Any]] = []

    def record_failure_or_degradation(
        self,
        test_id: str,
        condition: str,
        observed_behavior: str,
        safe_response: str,
        residual_risk: str,
        recovery_time_s: float = 0.0,
    ):
        """Log a failure, degraded state entry, or negative test trigger."""
        entry = {
            "test_id": test_id,
            "condition": condition,
            "observed_behavior": observed_behavior,
            "safe_response": safe_response,
            "residual_risk": residual_risk,
            "recovery_time_seconds": recovery_time_s,
            "timestamp": time.time(),
        }
        self.audit_log.append(entry)

    def get_audit_trail(self) -> List[Dict[str, Any]]:
        """Return full audit log for NT compliance."""
        return list(self.audit_log)
