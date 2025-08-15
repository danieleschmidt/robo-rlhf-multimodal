"""
Anomaly Detector: Advanced failure detection and prediction for pipeline components.

Implements machine learning-based anomaly detection, failure prediction, and pattern analysis.
"""

import asyncio
import logging
import time
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from collections import deque, defaultdict
from enum import Enum
import statistics

logger = logging.getLogger(__name__)


class AnomalyType(Enum):
    """Types of anomalies that can be detected."""
    PERFORMANCE_DEGRADATION = "performance_degradation"
    RESOURCE_EXHAUSTION = "resource_exhaustion"
    ERROR_SPIKE = "error_spike"
    AVAILABILITY_DROP = "availability_drop"
    LATENCY_INCREASE = "latency_increase"
    PATTERN_DEVIATION = "pattern_deviation"
    THRESHOLD_BREACH = "threshold_breach"


class Severity(Enum):
    """Severity levels for anomalies."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class Anomaly:
    """Represents a detected anomaly."""
    component: str
    type: AnomalyType
    severity: Severity
    confidence: float  # 0.0 to 1.0
    timestamp: float
    description: str
    metrics: Dict[str, float]
    prediction_horizon: Optional[float] = None  # For predictive anomalies
    recommendations: List[str] = field(default_factory=list)


@dataclass
class ThresholdRule:
    """Threshold-based detection rule."""
    metric_name: str
    threshold: float
    operator: str  # 'gt', 'lt', 'eq'
    severity: Severity
    window_size: int = 5  # Number of consecutive violations needed


class AnomalyDetector:
    """
    Advanced anomaly detection system with multiple detection algorithms.
    
    Features:
    - Statistical anomaly detection
    - Threshold-based rules
    - Pattern deviation detection
    - Machine learning models
    - Predictive failure analysis
    """
    
    def __init__(
        self,
        component_name: str,
        history_size: int = 1000,
        detection_interval: int = 60,
        enable_ml_detection: bool = True
    ):
        self.component_name = component_name
        self.history_size = history_size
        self.detection_interval = detection_interval
        self.enable_ml_detection = enable_ml_detection
        
        # Metric storage
        self.metrics_history: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=history_size)
        )
        
        # Detection rules
        self.threshold_rules: List[ThresholdRule] = []
        self.statistical_thresholds: Dict[str, Dict[str, float]] = {}
        
        # Anomaly tracking
        self.detected_anomalies: deque = deque(maxlen=500)
        self.active_anomalies: Dict[str, Anomaly] = {}
        
        # Detection state
        self.baseline_established = False
        self.min_samples_for_baseline = 30
        
        # ML models (lazy loaded)
        self._isolation_forest = None
        self._lstm_model = None
        
        # Quantum integration if available
        try:
            from robo_rlhf.quantum import PredictiveAnalytics
            self.predictive_analytics = PredictiveAnalytics()
            self.quantum_enabled = True
            logger.info(f"Quantum-enhanced anomaly detection for {component_name}")
        except ImportError:
            self.quantum_enabled = False
            logger.info(f"Standard anomaly detection for {component_name}")
    
    def add_threshold_rule(
        self,
        metric_name: str,
        threshold: float,
        operator: str,
        severity: Severity,
        window_size: int = 5
    ) -> None:
        """Add a threshold-based detection rule."""
        rule = ThresholdRule(
            metric_name=metric_name,
            threshold=threshold,
            operator=operator,
            severity=severity,
            window_size=window_size
        )
        self.threshold_rules.append(rule)
        logger.info(f"Added threshold rule: {metric_name} {operator} {threshold}")
    
    def update_metric(self, metric_name: str, value: float, timestamp: float) -> None:
        """Update metric value and trigger detection if needed."""
        self.metrics_history[metric_name].append({
            "value": value,
            "timestamp": timestamp
        })
        
        # Update statistical thresholds if baseline established
        if self.baseline_established:
            self._update_statistical_thresholds(metric_name)
    
    async def detect_anomalies(self) -> List[Anomaly]:
        """Perform comprehensive anomaly detection."""
        detected = []
        
        # Skip detection if insufficient data
        if not self._has_sufficient_data():
            return detected
        
        # Ensure baseline is established
        if not self.baseline_established:
            self._establish_baseline()
        
        # Multiple detection methods
        detected.extend(await self._threshold_detection())
        detected.extend(await self._statistical_detection())
        detected.extend(await self._pattern_detection())
        
        if self.enable_ml_detection:
            detected.extend(await self._ml_detection())
        
        if self.quantum_enabled:
            detected.extend(await self._quantum_detection())
        
        # Deduplicate and validate anomalies
        unique_anomalies = self._deduplicate_anomalies(detected)
        
        # Store and track anomalies
        for anomaly in unique_anomalies:
            self.detected_anomalies.append(anomaly)
            self.active_anomalies[f"{anomaly.type.value}_{anomaly.timestamp}"] = anomaly
        
        return unique_anomalies
    
    def _has_sufficient_data(self) -> bool:
        """Check if we have enough data for detection."""
        return any(
            len(history) >= self.min_samples_for_baseline
            for history in self.metrics_history.values()
        )
    
    def _establish_baseline(self) -> None:
        """Establish statistical baselines for metrics."""
        for metric_name, history in self.metrics_history.items():
            if len(history) < self.min_samples_for_baseline:
                continue
            
            values = [point["value"] for point in history]
            
            if len(values) >= self.min_samples_for_baseline:
                mean = statistics.mean(values)
                std_dev = statistics.stdev(values) if len(values) > 1 else 0
                
                # Calculate dynamic thresholds (2 and 3 sigma)
                self.statistical_thresholds[metric_name] = {
                    "mean": mean,
                    "std_dev": std_dev,
                    "upper_2sigma": mean + (2 * std_dev),
                    "lower_2sigma": mean - (2 * std_dev),
                    "upper_3sigma": mean + (3 * std_dev),
                    "lower_3sigma": mean - (3 * std_dev)
                }
        
        self.baseline_established = len(self.statistical_thresholds) > 0
        if self.baseline_established:
            logger.info(f"Baseline established for {self.component_name}")
    
    def _update_statistical_thresholds(self, metric_name: str) -> None:
        """Update statistical thresholds with new data."""
        history = self.metrics_history[metric_name]
        if len(history) < self.min_samples_for_baseline:
            return
        
        # Use recent history for adaptive thresholds
        recent_values = [point["value"] for point in list(history)[-100:]]
        
        if len(recent_values) >= 20:  # Minimum for stable statistics
            mean = statistics.mean(recent_values)
            std_dev = statistics.stdev(recent_values)
            
            self.statistical_thresholds[metric_name] = {
                "mean": mean,
                "std_dev": std_dev,
                "upper_2sigma": mean + (2 * std_dev),
                "lower_2sigma": mean - (2 * std_dev),
                "upper_3sigma": mean + (3 * std_dev),
                "lower_3sigma": mean - (3 * std_dev)
            }
    
    async def _threshold_detection(self) -> List[Anomaly]:
        """Detect anomalies using threshold rules."""
        anomalies = []
        
        for rule in self.threshold_rules:
            metric_history = self.metrics_history.get(rule.metric_name)
            if not metric_history or len(metric_history) < rule.window_size:
                continue
            
            # Check recent values against threshold
            recent_values = list(metric_history)[-rule.window_size:]
            violations = 0
            
            for point in recent_values:
                value = point["value"]
                
                if rule.operator == "gt" and value > rule.threshold:
                    violations += 1
                elif rule.operator == "lt" and value < rule.threshold:
                    violations += 1
                elif rule.operator == "eq" and abs(value - rule.threshold) < 0.001:
                    violations += 1
            
            # Trigger anomaly if all recent values violate threshold
            if violations == rule.window_size:
                current_value = recent_values[-1]["value"]
                
                anomaly = Anomaly(
                    component=self.component_name,
                    type=AnomalyType.THRESHOLD_BREACH,
                    severity=rule.severity,
                    confidence=1.0,  # High confidence for threshold violations
                    timestamp=time.time(),
                    description=f"{rule.metric_name} {rule.operator} {rule.threshold} (current: {current_value:.2f})",
                    metrics={rule.metric_name: current_value},
                    recommendations=[f"Investigate {rule.metric_name} spike", "Check system resources"]
                )
                anomalies.append(anomaly)
        
        return anomalies
    
    async def _statistical_detection(self) -> List[Anomaly]:
        """Detect anomalies using statistical methods."""
        anomalies = []
        
        for metric_name, thresholds in self.statistical_thresholds.items():
            history = self.metrics_history[metric_name]
            if not history:
                continue
            
            current_point = history[-1]
            current_value = current_point["value"]
            
            # Check for statistical anomalies
            if current_value > thresholds["upper_3sigma"]:
                anomaly = Anomaly(
                    component=self.component_name,
                    type=AnomalyType.PATTERN_DEVIATION,
                    severity=Severity.HIGH,
                    confidence=0.95,
                    timestamp=current_point["timestamp"],
                    description=f"{metric_name} significantly above normal (3σ+)",
                    metrics={metric_name: current_value, "mean": thresholds["mean"]},
                    recommendations=["Investigate unusual high values", "Check for resource leaks"]
                )
                anomalies.append(anomaly)
            
            elif current_value < thresholds["lower_3sigma"]:
                anomaly = Anomaly(
                    component=self.component_name,
                    type=AnomalyType.PATTERN_DEVIATION,
                    severity=Severity.HIGH,
                    confidence=0.95,
                    timestamp=current_point["timestamp"],
                    description=f"{metric_name} significantly below normal (3σ-)",
                    metrics={metric_name: current_value, "mean": thresholds["mean"]},
                    recommendations=["Investigate unusual low values", "Check component health"]
                )
                anomalies.append(anomaly)
            
            elif current_value > thresholds["upper_2sigma"]:
                anomaly = Anomaly(
                    component=self.component_name,
                    type=AnomalyType.PERFORMANCE_DEGRADATION,
                    severity=Severity.MEDIUM,
                    confidence=0.85,
                    timestamp=current_point["timestamp"],
                    description=f"{metric_name} above normal range (2σ+)",
                    metrics={metric_name: current_value, "mean": thresholds["mean"]},
                    recommendations=["Monitor closely", "Consider preventive actions"]
                )
                anomalies.append(anomaly)
        
        return anomalies
    
    async def _pattern_detection(self) -> List[Anomaly]:
        """Detect anomalies based on pattern analysis."""
        anomalies = []
        
        for metric_name, history in self.metrics_history.items():
            if len(history) < 20:  # Need sufficient data for pattern analysis
                continue
            
            recent_values = [point["value"] for point in list(history)[-20:]]
            
            # Detect trends
            trend_anomaly = self._detect_trend_anomaly(metric_name, recent_values)
            if trend_anomaly:
                anomalies.append(trend_anomaly)
            
            # Detect periodic anomalies
            periodic_anomaly = self._detect_periodic_anomaly(metric_name, recent_values)
            if periodic_anomaly:
                anomalies.append(periodic_anomaly)
        
        return anomalies
    
    def _detect_trend_anomaly(self, metric_name: str, values: List[float]) -> Optional[Anomaly]:
        """Detect strong trending patterns that might indicate issues."""
        if len(values) < 10:
            return None
        
        # Calculate trend using linear regression
        x = list(range(len(values)))
        n = len(values)
        
        sum_x = sum(x)
        sum_y = sum(values)
        sum_xy = sum(x[i] * values[i] for i in range(n))
        sum_x2 = sum(x[i] ** 2 for i in range(n))
        
        # Calculate slope
        slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x ** 2)
        
        # Check for strong positive trend (potential resource leak)
        if slope > 0 and abs(slope) > (max(values) - min(values)) / (2 * len(values)):
            return Anomaly(
                component=self.component_name,
                type=AnomalyType.RESOURCE_EXHAUSTION,
                severity=Severity.MEDIUM,
                confidence=0.8,
                timestamp=time.time(),
                description=f"{metric_name} showing strong upward trend (slope: {slope:.3f})",
                metrics={metric_name: values[-1], "trend_slope": slope},
                recommendations=["Investigate resource usage growth", "Check for memory leaks"]
            )
        
        return None
    
    def _detect_periodic_anomaly(self, metric_name: str, values: List[float]) -> Optional[Anomaly]:
        """Detect unusual periodic patterns."""
        if len(values) < 15:
            return None
        
        # Simple oscillation detection
        changes = [values[i+1] - values[i] for i in range(len(values)-1)]
        sign_changes = sum(1 for i in range(len(changes)-1) if changes[i] * changes[i+1] < 0)
        
        # High frequency oscillation might indicate instability
        oscillation_ratio = sign_changes / len(changes)
        
        if oscillation_ratio > 0.7:  # More than 70% sign changes
            return Anomaly(
                component=self.component_name,
                type=AnomalyType.PATTERN_DEVIATION,
                severity=Severity.MEDIUM,
                confidence=0.75,
                timestamp=time.time(),
                description=f"{metric_name} showing high frequency oscillation",
                metrics={metric_name: values[-1], "oscillation_ratio": oscillation_ratio},
                recommendations=["Check system stability", "Investigate control loops"]
            )
        
        return None
    
    async def _ml_detection(self) -> List[Anomaly]:
        """Machine learning-based anomaly detection."""
        try:
            # Lazy load ML models
            if self._isolation_forest is None:
                self._initialize_ml_models()
            
            anomalies = []
            
            # Prepare feature matrix
            features = self._prepare_features()
            if features is None or len(features) < 10:
                return anomalies
            
            # Isolation Forest detection
            if self._isolation_forest is not None:
                outlier_scores = self._isolation_forest.decision_function([features[-1]])
                
                if outlier_scores[0] < -0.1:  # Threshold for anomaly
                    anomaly = Anomaly(
                        component=self.component_name,
                        type=AnomalyType.PATTERN_DEVIATION,
                        severity=Severity.MEDIUM,
                        confidence=abs(outlier_scores[0]),
                        timestamp=time.time(),
                        description="ML model detected anomalous pattern",
                        metrics={"outlier_score": outlier_scores[0]},
                        recommendations=["ML analysis suggests investigation", "Check recent changes"]
                    )
                    anomalies.append(anomaly)
            
            return anomalies
            
        except Exception as e:
            logger.error(f"ML detection failed: {e}")
            return []
    
    def _initialize_ml_models(self) -> None:
        """Initialize machine learning models."""
        try:
            from sklearn.ensemble import IsolationForest
            
            # Initialize Isolation Forest
            self._isolation_forest = IsolationForest(
                contamination=0.1,  # Expect 10% anomalies
                random_state=42
            )
            
            # Train on existing data if available
            features = self._prepare_features()
            if features is not None and len(features) >= 20:
                self._isolation_forest.fit(features)
                logger.info("Isolation Forest model trained")
                
        except ImportError:
            logger.warning("scikit-learn not available, ML detection disabled")
        except Exception as e:
            logger.error(f"Failed to initialize ML models: {e}")
    
    def _prepare_features(self) -> Optional[np.ndarray]:
        """Prepare feature matrix for ML models."""
        try:
            # Get metrics with sufficient data
            metrics_with_data = {
                name: history for name, history in self.metrics_history.items()
                if len(history) >= 20
            }
            
            if not metrics_with_data:
                return None
            
            # Create feature matrix
            min_length = min(len(history) for history in metrics_with_data.values())
            features = []
            
            for i in range(min_length):
                feature_vector = []
                for metric_name in sorted(metrics_with_data.keys()):
                    history = metrics_with_data[metric_name]
                    feature_vector.append(history[i]["value"])
                features.append(feature_vector)
            
            return np.array(features)
            
        except Exception as e:
            logger.error(f"Feature preparation failed: {e}")
            return None
    
    async def _quantum_detection(self) -> List[Anomaly]:
        """Quantum-enhanced anomaly detection."""
        try:
            # Prepare metrics data for quantum analysis
            metrics_data = {}
            for metric_name, history in self.metrics_history.items():
                if len(history) >= 10:
                    metrics_data[metric_name] = [
                        point["value"] for point in list(history)[-50:]
                    ]
            
            if not metrics_data:
                return []
            
            # Get quantum predictions
            quantum_results = await self.predictive_analytics.detect_anomalies(
                component=self.component_name,
                metrics_data=metrics_data
            )
            
            anomalies = []
            for result in quantum_results.get("anomalies", []):
                anomaly = Anomaly(
                    component=self.component_name,
                    type=AnomalyType(result.get("type", "pattern_deviation")),
                    severity=Severity(result.get("severity", "medium")),
                    confidence=result.get("confidence", 0.7),
                    timestamp=time.time(),
                    description=result.get("description", "Quantum-detected anomaly"),
                    metrics=result.get("metrics", {}),
                    prediction_horizon=result.get("prediction_horizon"),
                    recommendations=result.get("recommendations", [])
                )
                anomalies.append(anomaly)
            
            return anomalies
            
        except Exception as e:
            logger.error(f"Quantum detection failed: {e}")
            return []
    
    def _deduplicate_anomalies(self, anomalies: List[Anomaly]) -> List[Anomaly]:
        """Remove duplicate anomalies based on type and timestamp proximity."""
        unique_anomalies = []
        
        for anomaly in anomalies:
            is_duplicate = False
            
            for existing in unique_anomalies:
                # Check if similar anomaly exists within time window
                if (existing.type == anomaly.type and 
                    existing.component == anomaly.component and
                    abs(existing.timestamp - anomaly.timestamp) < 60):  # 1 minute window
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique_anomalies.append(anomaly)
        
        return unique_anomalies
    
    def get_active_anomalies(self) -> List[Anomaly]:
        """Get currently active anomalies."""
        current_time = time.time()
        active = []
        
        for anomaly in self.active_anomalies.values():
            # Consider anomaly active if detected within last 5 minutes
            if current_time - anomaly.timestamp < 300:
                active.append(anomaly)
        
        return active
    
    def get_detection_stats(self) -> Dict[str, Any]:
        """Get detection statistics."""
        total_anomalies = len(self.detected_anomalies)
        active_count = len(self.get_active_anomalies())
        
        # Count by type
        type_counts = defaultdict(int)
        severity_counts = defaultdict(int)
        
        for anomaly in self.detected_anomalies:
            type_counts[anomaly.type.value] += 1
            severity_counts[anomaly.severity.value] += 1
        
        return {
            "component": self.component_name,
            "total_anomalies_detected": total_anomalies,
            "active_anomalies": active_count,
            "baseline_established": self.baseline_established,
            "anomalies_by_type": dict(type_counts),
            "anomalies_by_severity": dict(severity_counts),
            "threshold_rules": len(self.threshold_rules),
            "metrics_tracked": len(self.metrics_history),
            "quantum_enabled": self.quantum_enabled,
            "ml_enabled": self.enable_ml_detection and self._isolation_forest is not None
        }


class FailurePredictor:
    """
    Predictive failure analysis system.
    
    Uses historical data and machine learning to predict potential failures
    before they occur, enabling proactive intervention.
    """
    
    def __init__(self, prediction_horizon: float = 3600.0):  # 1 hour default
        self.prediction_horizon = prediction_horizon
        self.failure_patterns: Dict[str, List[Dict[str, Any]]] = {}
        self.prediction_models: Dict[str, Any] = {}
        
        # Quantum integration if available
        try:
            from robo_rlhf.quantum import PredictiveAnalytics
            self.predictive_analytics = PredictiveAnalytics()
            self.quantum_enabled = True
            logger.info("Quantum-enhanced failure prediction initialized")
        except ImportError:
            self.quantum_enabled = False
            logger.info("Standard failure prediction initialized")
    
    def record_failure(
        self,
        component: str,
        failure_type: str,
        preceding_metrics: Dict[str, List[float]],
        context: Dict[str, Any]
    ) -> None:
        """Record a failure with its preceding conditions."""
        failure_record = {
            "timestamp": time.time(),
            "failure_type": failure_type,
            "preceding_metrics": preceding_metrics,
            "context": context
        }
        
        if component not in self.failure_patterns:
            self.failure_patterns[component] = []
        
        self.failure_patterns[component].append(failure_record)
        
        # Keep only recent patterns (last 100 failures)
        if len(self.failure_patterns[component]) > 100:
            self.failure_patterns[component] = self.failure_patterns[component][-100:]
        
        logger.info(f"Recorded failure pattern for {component}: {failure_type}")
    
    async def predict_failure_risk(
        self,
        component: str,
        current_metrics: Dict[str, float]
    ) -> Dict[str, Any]:
        """Predict failure risk for a component."""
        if self.quantum_enabled:
            return await self._quantum_prediction(component, current_metrics)
        else:
            return await self._statistical_prediction(component, current_metrics)
    
    async def _quantum_prediction(
        self,
        component: str,
        current_metrics: Dict[str, float]
    ) -> Dict[str, Any]:
        """Use quantum analytics for failure prediction."""
        try:
            prediction = await self.predictive_analytics.predict_failure_risk(
                component=component,
                current_metrics=current_metrics,
                failure_history=self.failure_patterns.get(component, []),
                prediction_horizon=self.prediction_horizon
            )
            
            return prediction
            
        except Exception as e:
            logger.error(f"Quantum prediction failed: {e}")
            return await self._statistical_prediction(component, current_metrics)
    
    async def _statistical_prediction(
        self,
        component: str,
        current_metrics: Dict[str, float]
    ) -> Dict[str, Any]:
        """Statistical failure prediction."""
        if component not in self.failure_patterns:
            return {
                "risk_score": 0.0,
                "confidence": 0.0,
                "predicted_time_to_failure": None,
                "risk_factors": [],
                "recommendations": []
            }
        
        patterns = self.failure_patterns[component]
        if not patterns:
            return {
                "risk_score": 0.0,
                "confidence": 0.0,
                "predicted_time_to_failure": None,
                "risk_factors": [],
                "recommendations": []
            }
        
        # Analyze patterns to find risk indicators
        risk_factors = []
        risk_score = 0.0
        
        for pattern in patterns[-10:]:  # Analyze recent patterns
            preceding_metrics = pattern["preceding_metrics"]
            
            # Check for similar metric patterns
            for metric_name, metric_value in current_metrics.items():
                if metric_name in preceding_metrics:
                    historical_values = preceding_metrics[metric_name]
                    if historical_values:
                        avg_before_failure = statistics.mean(historical_values)
                        
                        # Check if current value is similar to pre-failure values
                        similarity = 1.0 - abs(metric_value - avg_before_failure) / max(avg_before_failure, 1.0)
                        
                        if similarity > 0.8:  # High similarity
                            risk_factors.append(f"{metric_name} similar to pre-failure patterns")
                            risk_score += 0.1
        
        # Cap risk score at 1.0
        risk_score = min(risk_score, 1.0)
        
        recommendations = []
        if risk_score > 0.7:
            recommendations.extend([
                "Consider preventive restart",
                "Scale resources proactively",
                "Enable circuit breaker"
            ])
        elif risk_score > 0.4:
            recommendations.extend([
                "Monitor closely",
                "Prepare recovery plans"
            ])
        
        return {
            "risk_score": risk_score,
            "confidence": 0.6 if len(patterns) >= 5 else 0.3,
            "predicted_time_to_failure": self.prediction_horizon if risk_score > 0.8 else None,
            "risk_factors": risk_factors,
            "recommendations": recommendations
        }