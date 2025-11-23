# ABOUTME: Production performance monitoring and alerting system.
# ABOUTME: Provides real-time monitoring, alerting, and performance regression detection.

import logging
import time
import json
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
import statistics
import threading
from collections import deque

from dealgraph.performance.metrics import PerformanceMetrics
from dealgraph.reasoning import deal_reasoner, extract_reasoning_metrics, ReasoningError


@dataclass
class MonitoringAlert:
    """Performance monitoring alert."""
    
    alert_id: str
    timestamp: datetime
    alert_type: str  # performance_degradation, error_rate_spike, low_confidence
    severity: str  # low, medium, high, critical
    message: str
    metric_name: str
    current_value: float
    threshold_value: float
    additional_data: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert alert to dictionary."""
        return {
            "alert_id": self.alert_id,
            "timestamp": self.timestamp.isoformat(),
            "alert_type": self.alert_type,
            "severity": self.severity,
            "message": self.message,
            "metric_name": self.metric_name,
            "current_value": self.current_value,
            "threshold_value": self.threshold_value,
            "additional_data": self.additional_data
        }


@dataclass
class PerformanceThreshold:
    """Performance threshold configuration."""
    
    metric_name: str
    warning_threshold: float
    critical_threshold: float
    direction: str = "below"  # "above" or "below"
    time_window_minutes: int = 5
    min_samples: int = 10
    
    def evaluate(self, values: List[float]) -> Optional[str]:
        """Evaluate if threshold is breached."""
        if len(values) < self.min_samples:
            return None
        
        recent_values = values[-min(len(values), 50):]  # Last 50 values
        current_value = statistics.mean(recent_values)
        
        if self.direction == "below":
            if current_value <= self.critical_threshold:
                return "critical"
            elif current_value <= self.warning_threshold:
                return "warning"
        else:  # direction == "above"
            if current_value >= self.critical_threshold:
                return "critical"
            elif current_value >= self.warning_threshold:
                return "warning"
        
        return None


class PerformanceMonitor:
    """
    Production performance monitoring and alerting system.
    
    Monitors reasoning performance in real-time, detects regressions,
    and sends alerts when thresholds are breached.
    """
    
    def __init__(
        self,
        performance_metrics: Optional[PerformanceMetrics] = None,
        alert_callback: Optional[Callable[[MonitoringAlert], None]] = None,
        monitoring_interval_seconds: int = 30
    ):
        """
        Initialize performance monitor.
        
        Args:
            performance_metrics: Performance measurement utilities
            alert_callback: Function to call when alerts are generated
            monitoring_interval_seconds: Interval for monitoring checks
        """
        self.performance_metrics = performance_metrics or PerformanceMetrics()
        self.alert_callback = alert_callback
        self.monitoring_interval = monitoring_interval_seconds
        self.logger = logging.getLogger(__name__)
        
        # Performance data storage
        self.performance_history: Dict[str, deque] = {
            "composite_score": deque(maxlen=1000),
            "precision_at_3": deque(maxlen=1000),
            "playbook_quality": deque(maxlen=1000),
            "narrative_coherence": deque(maxlen=1000),
            "success_rate": deque(maxlen=1000),
            "response_time": deque(maxlen=1000),
            "error_rate": deque(maxlen=1000)
        }
        
        # Alert management
        self.active_alerts: Dict[str, MonitoringAlert] = {}
        self.alert_history: List[MonitoringAlert] = []
        self.alert_thresholds: List[PerformanceThreshold] = []
        
        # Monitoring control
        self.monitoring_active = False
        self.monitoring_thread = None
        
        # Statistics
        self.total_requests = 0
        self.successful_requests = 0
        self.failed_requests = 0
        self.start_time = datetime.now()
        
        # Initialize default thresholds
        self._setup_default_thresholds()
    
    def _setup_default_thresholds(self):
        """Set up default performance thresholds."""
        self.alert_thresholds = [
            # Composite score thresholds
            PerformanceThreshold(
                metric_name="composite_score",
                warning_threshold=0.6,
                critical_threshold=0.4,
                direction="below",
                time_window_minutes=5,
                min_samples=20
            ),
            # Success rate thresholds
            PerformanceThreshold(
                metric_name="success_rate",
                warning_threshold=0.90,
                critical_threshold=0.80,
                direction="below",
                time_window_minutes=2,
                min_samples=50
            ),
            # Error rate thresholds
            PerformanceThreshold(
                metric_name="error_rate",
                warning_threshold=0.05,
                critical_threshold=0.10,
                direction="above",
                time_window_minutes=2,
                min_samples=50
            ),
            # Response time thresholds
            PerformanceThreshold(
                metric_name="response_time",
                warning_threshold=10.0,
                critical_threshold=30.0,
                direction="above",
                time_window_minutes=1,
                min_samples=10
            ),
            # Precision thresholds
            PerformanceThreshold(
                metric_name="precision_at_3",
                warning_threshold=0.5,
                critical_threshold=0.3,
                direction="below",
                time_window_minutes=5,
                min_samples=20
            )
        ]
    
    def start_monitoring(self):
        """Start continuous performance monitoring."""
        if self.monitoring_active:
            self.logger.warning("Performance monitoring is already active")
            return
        
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        
        self.logger.info("Performance monitoring started")
    
    def stop_monitoring(self):
        """Stop performance monitoring."""
        self.monitoring_active = False
        
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)
        
        self.logger.info("Performance monitoring stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop."""
        while self.monitoring_active:
            try:
                self._check_performance_thresholds()
                time.sleep(self.monitoring_interval)
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                time.sleep(self.monitoring_interval)
    
    def record_request(
        self,
        query: str,
        ranked_deals,
        prompt_version: str = "v1",
        response_time_seconds: float = None,
        user_id: str = None
    ) -> Dict[str, float]:
        """
        Record a reasoning request for monitoring.
        
        Args:
            query: User query
            ranked_deals: Ranked deals
            prompt_version: Prompt version used
            response_time_seconds: Optional response time
            user_id: Optional user identifier
            
        Returns:
            Recorded performance metrics
        """
        start_time = time.time()
        
        try:
            self.total_requests += 1
            
            # Run reasoning
            reasoning_output = deal_reasoner(
                query=query,
                ranked_deals=ranked_deals,
                prompt_version=prompt_version
            )
            
            # Extract metrics
            metrics = self.performance_metrics.evaluate_reasoning_output({
                "precedents": [
                    {"deal_id": p.deal_id, "name": p.name, "similarity_reason": p.similarity_reason}
                    for p in reasoning_output.precedents
                ],
                "playbook_levers": reasoning_output.playbook_levers,
                "risk_themes": reasoning_output.risk_themes,
                "narrative_summary": reasoning_output.narrative_summary
            })
            
            self.successful_requests += 1
            
            # Record performance data
            self._record_performance_metrics(metrics, response_time_seconds)
            
            return metrics
            
        except Exception as e:
            self.failed_requests += 1
            self.logger.error(f"Reasoning request failed: {e}")
            
            # Record error
            error_metrics = self.performance_metrics._get_default_metrics()
            self._record_performance_metrics(error_metrics, response_time_seconds)
            
            # Check for error rate spike
            self._check_error_rate_spike()
            
            return error_metrics
    
    def _record_performance_metrics(
        self,
        metrics: Dict[str, float],
        response_time_seconds: float = None
    ):
        """Record performance metrics to history."""
        # Calculate response time if not provided
        if response_time_seconds is None:
            response_time_seconds = time.time() - self._get_last_request_start_time()
        
        # Record individual metrics
        for metric_name, value in metrics.items():
            if metric_name in self.performance_history:
                self.performance_history[metric_name].append(value)
        
        # Record derived metrics
        self.performance_history["response_time"].append(response_time_seconds)
        
        # Calculate and record success rate
        success_rate = self.successful_requests / self.total_requests if self.total_requests > 0 else 0
        self.performance_history["success_rate"].append(success_rate)
        
        # Calculate and record error rate
        error_rate = self.failed_requests / self.total_requests if self.total_requests > 0 else 0
        self.performance_history["error_rate"].append(error_rate)
    
    def _get_last_request_start_time(self) -> float:
        """Get the start time of the last request (simplified)."""
        # This is a simplified implementation
        # In practice, you'd track this more precisely
        return time.time()
    
    def _check_performance_thresholds(self):
        """Check all performance thresholds."""
        for threshold in self.alert_thresholds:
            self._check_single_threshold(threshold)
    
    def _check_single_threshold(self, threshold: PerformanceThreshold):
        """Check a single performance threshold."""
        try:
            if threshold.metric_name not in self.performance_history:
                return
            
            values = list(self.performance_history[threshold.metric_name])
            breach_level = threshold.evaluate(values)
            
            if breach_level:
                self._trigger_alert(threshold, values, breach_level)
            
        except Exception as e:
            self.logger.warning(f"Error checking threshold {threshold.metric_name}: {e}")
    
    def _trigger_alert(
        self,
        threshold: PerformanceThreshold,
        values: List[float],
        breach_level: str
    ):
        """Trigger an alert for threshold breach."""
        current_value = statistics.mean(values[-min(len(values), 50):])
        
        # Determine severity
        severity_map = {"warning": "medium", "critical": "high"}
        severity = severity_map.get(breach_level, "medium")
        
        # Create alert
        alert_id = f"{threshold.metric_name}_{int(time.time())}"
        alert = MonitoringAlert(
            alert_id=alert_id,
            timestamp=datetime.now(),
            alert_type="performance_degradation",
            severity=severity,
            message=f"{threshold.metric_name} {breach_level}: {current_value:.3f} (threshold: {threshold.warning_threshold:.3f})",
            metric_name=threshold.metric_name,
            current_value=current_value,
            threshold_value=threshold.warning_threshold,
            additional_data={
                "breach_level": breach_level,
                "threshold_type": "performance",
                "data_points": len(values)
            }
        )
        
        # Store alert
        self.active_alerts[alert_id] = alert
        self.alert_history.append(alert)
        
        # Call alert callback if provided
        if self.alert_callback:
            try:
                self.alert_callback(alert)
            except Exception as e:
                self.logger.error(f"Alert callback failed: {e}")
        
        # Clean up old alerts for this metric
        self._cleanup_old_alerts(threshold.metric_name)
        
        self.logger.warning(f"Performance alert triggered: {alert.message}")
    
    def _cleanup_old_alerts(self, metric_name: str):
        """Clean up old alerts for a metric (keep only the most recent)."""
        metric_alerts = [
            alert for alert in self.alert_history
            if alert.metric_name == metric_name and alert.alert_id in self.active_alerts
        ]
        
        # Sort by timestamp, keep only the most recent
        metric_alerts.sort(key=lambda x: x.timestamp, reverse=True)
        
        # Remove all but the most recent
        for alert in metric_alerts[1:]:
            if alert.alert_id in self.active_alerts:
                del self.active_alerts[alert.alert_id]
    
    def _check_error_rate_spike(self):
        """Check for error rate spike."""
        error_rates = list(self.performance_history["error_rate"])
        
        if len(error_rates) < 10:
            return
        
        recent_error_rates = error_rates[-10:]  # Last 10 requests
        current_error_rate = statistics.mean(recent_error_rates)
        
        # Check for spike (more than 20% error rate in recent requests)
        if current_error_rate > 0.2:
            alert_id = f"error_rate_spike_{int(time.time())}"
            alert = MonitoringAlert(
                alert_id=alert_id,
                timestamp=datetime.now(),
                alert_type="error_rate_spike",
                severity="high",
                message=f"High error rate detected: {current_error_rate:.1%}",
                metric_name="error_rate",
                current_value=current_error_rate,
                threshold_value=0.2,
                additional_data={
                    "recent_requests": len(recent_error_rates),
                    "failed_requests": int(current_error_rate * len(recent_error_rates))
                }
            )
            
            self.active_alerts[alert_id] = alert
            self.alert_history.append(alert)
            
            if self.alert_callback:
                try:
                    self.alert_callback(alert)
                except Exception as e:
                    self.logger.error(f"Alert callback failed: {e}")
            
            self.logger.error(f"Error rate spike detected: {current_error_rate:.1%}")
    
    def get_current_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics."""
        metrics = {}
        
        for metric_name, values in self.performance_history.items():
            if values:
                recent_values = list(values)[-min(len(values), 50):]
                metrics[metric_name] = {
                    "current": recent_values[-1] if recent_values else 0,
                    "mean": statistics.mean(recent_values),
                    "median": statistics.median(recent_values),
                    "std": statistics.stdev(recent_values) if len(recent_values) > 1 else 0,
                    "min": min(recent_values),
                    "max": max(recent_values),
                    "sample_size": len(recent_values)
                }
        
        # Add summary statistics
        uptime = (datetime.now() - self.start_time).total_seconds()
        metrics["summary"] = {
            "uptime_seconds": uptime,
            "total_requests": self.total_requests,
            "successful_requests": self.successful_requests,
            "failed_requests": self.failed_requests,
            "overall_success_rate": self.successful_requests / self.total_requests if self.total_requests > 0 else 0,
            "requests_per_minute": (self.total_requests / (uptime / 60)) if uptime > 0 else 0
        }
        
        return metrics
    
    def get_active_alerts(self) -> List[Dict[str, Any]]:
        """Get currently active alerts."""
        return [alert.to_dict() for alert in self.active_alerts.values()]
    
    def get_alert_history(
        self,
        hours: int = 24,
        alert_type: str = None,
        severity: str = None
    ) -> List[Dict[str, Any]]:
        """Get alert history with optional filtering."""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        filtered_alerts = [
            alert for alert in self.alert_history
            if alert.timestamp > cutoff_time
        ]
        
        if alert_type:
            filtered_alerts = [
                alert for alert in filtered_alerts
                if alert.alert_type == alert_type
            ]
        
        if severity:
            filtered_alerts = [
                alert for alert in filtered_alerts
                if alert.severity == severity
            ]
        
        return [alert.to_dict() for alert in filtered_alerts]
    
    def add_custom_threshold(self, threshold: PerformanceThreshold):
        """Add a custom performance threshold."""
        self.alert_thresholds.append(threshold)
        self.logger.info(f"Added custom threshold for {threshold.metric_name}")
    
    def remove_threshold(self, metric_name: str) -> bool:
        """Remove a performance threshold."""
        original_count = len(self.alert_thresholds)
        self.alert_thresholds = [
            t for t in self.alert_thresholds
            if t.metric_name != metric_name
        ]
        
        removed = len(self.alert_thresholds) < original_count
        if removed:
            self.logger.info(f"Removed threshold for {metric_name}")
        
        return removed
    
    def get_performance_trends(self, metric_name: str, hours: int = 24) -> Dict[str, Any]:
        """Get performance trends for a specific metric."""
        if metric_name not in self.performance_history:
            return {"error": f"Metric {metric_name} not found"}
        
        cutoff_time = datetime.now() - timedelta(hours=hours)
        cutoff_timestamp = time.mktime(cutoff_time.timetuple())
        
        # Get values within time window
        # Note: This is simplified - in practice you'd track timestamps with values
        values = list(self.performance_history[metric_name])
        
        if not values:
            return {"error": "No data available"}
        
        # Calculate trend statistics
        recent_values = values[-min(len(values), 100):]  # Last 100 values
        
        return {
            "metric_name": metric_name,
            "time_window_hours": hours,
            "current_value": recent_values[-1],
            "mean": statistics.mean(recent_values),
            "trend": self._calculate_trend(recent_values),
            "sample_size": len(recent_values)
        }
    
    def _calculate_trend(self, values: List[float]) -> str:
        """Calculate simple trend direction."""
        if len(values) < 10:
            return "insufficient_data"
        
        # Simple linear trend
        n = len(values)
        x_values = list(range(n))
        
        x_mean = sum(x_values) / n
        y_mean = sum(values) / n
        
        numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(x_values, values))
        denominator = sum((x - x_mean) ** 2 for x in x_values)
        
        slope = numerator / denominator if denominator != 0 else 0
        
        if slope > 0.001:
            return "improving"
        elif slope < -0.001:
            return "declining"
        else:
            return "stable"
    
    def save_monitoring_report(self, output_path: str) -> str:
        """Save monitoring report to file."""
        report_data = {
            "report_generated": datetime.now().isoformat(),
            "monitoring_duration": (datetime.now() - self.start_time).total_seconds(),
            "current_metrics": self.get_current_metrics(),
            "active_alerts": self.get_active_alerts(),
            "recent_alerts": self.get_alert_history(hours=24),
            "thresholds": [
                {
                    "metric_name": t.metric_name,
                    "warning_threshold": t.warning_threshold,
                    "critical_threshold": t.critical_threshold,
                    "direction": t.direction
                }
                for t in self.alert_thresholds
            ]
        }
        
        # Create output directory
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Save report
        with open(output_path, 'w') as f:
            json.dump(report_data, f, indent=2)
        
        self.logger.info(f"Monitoring report saved to: {output_path}")
        return output_path
