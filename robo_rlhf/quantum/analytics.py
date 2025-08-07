"""
Predictive analytics and autonomous resource management for quantum-inspired SDLC.

Implements machine learning models for predicting system behavior, resource usage,
performance bottlenecks, and autonomous optimization recommendations.
"""

import asyncio
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import time
import json
from collections import deque, defaultdict
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

from robo_rlhf.core import get_logger, get_config
from robo_rlhf.core.exceptions import RoboRLHFError


class PredictionType(Enum):
    """Types of predictions."""
    RESOURCE_USAGE = "resource_usage"
    PERFORMANCE = "performance" 
    FAILURE_PROBABILITY = "failure_probability"
    COMPLETION_TIME = "completion_time"
    QUALITY_SCORE = "quality_score"
    OPTIMIZATION_IMPACT = "optimization_impact"


class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


@dataclass
class MetricSample:
    """Time-series metric sample."""
    timestamp: float
    value: float
    source: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PredictionResult:
    """Result of a prediction."""
    prediction_type: PredictionType
    predicted_value: float
    confidence: float
    prediction_horizon: float
    model_accuracy: float
    timestamp: float = field(default_factory=time.time)
    features_used: List[str] = field(default_factory=list)
    explanation: str = ""


@dataclass
class ResourcePrediction:
    """Resource usage prediction."""
    resource_type: str
    current_usage: float
    predicted_usage: float
    peak_prediction: float
    time_to_peak: float
    confidence: float
    recommendations: List[str] = field(default_factory=list)


@dataclass
class Alert:
    """System alert."""
    id: str
    severity: AlertSeverity
    title: str
    description: str
    source: str
    timestamp: float = field(default_factory=time.time)
    resolved: bool = False
    resolution_time: Optional[float] = None
    actions_taken: List[str] = field(default_factory=list)


class PredictiveAnalytics:
    """Predictive analytics engine for autonomous SDLC."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.logger = get_logger(__name__)
        self.config = config or get_config().to_dict()
        
        # Analytics parameters
        self.window_size = self.config.get("analytics", {}).get("window_size", 100)
        self.prediction_horizon = self.config.get("analytics", {}).get("prediction_horizon", 300.0)  # 5 minutes
        self.model_retrain_interval = self.config.get("analytics", {}).get("retrain_interval", 3600.0)  # 1 hour
        self.anomaly_threshold = self.config.get("analytics", {}).get("anomaly_threshold", 0.05)
        
        # Time-series data storage
        self.metrics_buffer: Dict[str, deque] = defaultdict(lambda: deque(maxlen=self.window_size))
        self.prediction_history: List[PredictionResult] = []
        self.alerts: List[Alert] = []
        
        # Machine learning models
        self.models: Dict[str, Any] = {}
        self.scalers: Dict[str, StandardScaler] = {}
        self.model_performance: Dict[str, List[float]] = defaultdict(list)
        
        # Anomaly detection
        self.anomaly_detector = IsolationForest(contamination=self.anomaly_threshold, random_state=42)
        self.anomaly_fitted = False
        
        # Pattern recognition
        self.pattern_clusters = {}
        self.cluster_models: Dict[str, KMeans] = {}
        
        self.logger.info("PredictiveAnalytics initialized with ML-based forecasting")
    
    async def ingest_metrics(self, metrics: Dict[str, float], source: str = "system") -> None:
        """Ingest real-time metrics for analysis."""
        timestamp = time.time()
        
        for metric_name, value in metrics.items():
            sample = MetricSample(
                timestamp=timestamp,
                value=value,
                source=source,
                metadata={"ingestion_time": timestamp}
            )
            
            self.metrics_buffer[metric_name].append(sample)
        
        # Trigger analysis if buffer is full
        if any(len(buffer) >= self.window_size for buffer in self.metrics_buffer.values()):
            await self._trigger_analysis()
    
    async def predict(self, 
                     prediction_type: PredictionType,
                     target_metric: str,
                     horizon: Optional[float] = None) -> PredictionResult:
        """Generate prediction for specified metric and type."""
        horizon = horizon or self.prediction_horizon
        
        if target_metric not in self.metrics_buffer or len(self.metrics_buffer[target_metric]) < 10:
            return PredictionResult(
                prediction_type=prediction_type,
                predicted_value=0.0,
                confidence=0.0,
                prediction_horizon=horizon,
                model_accuracy=0.0,
                explanation="Insufficient data for prediction"
            )
        
        # Prepare features
        features, target_values = self._prepare_features(target_metric)
        
        if len(features) == 0:
            return PredictionResult(
                prediction_type=prediction_type,
                predicted_value=0.0,
                confidence=0.0,
                prediction_horizon=horizon,
                model_accuracy=0.0,
                explanation="No features available"
            )
        
        # Get or create model
        model = await self._get_or_create_model(prediction_type, target_metric, features, target_values)
        
        # Make prediction
        latest_features = features[-1:] if len(features) > 0 else [[0]]
        predicted_value = model.predict(latest_features)[0]
        
        # Calculate confidence based on model performance
        model_key = f"{prediction_type.value}_{target_metric}"
        confidence = self._calculate_prediction_confidence(model_key, predicted_value, target_values)
        
        # Get model accuracy
        accuracy = np.mean(self.model_performance[model_key][-10:]) if self.model_performance[model_key] else 0.5
        
        # Generate explanation
        explanation = self._generate_prediction_explanation(prediction_type, target_metric, predicted_value, confidence)
        
        result = PredictionResult(
            prediction_type=prediction_type,
            predicted_value=predicted_value,
            confidence=confidence,
            prediction_horizon=horizon,
            model_accuracy=accuracy,
            features_used=self._get_feature_names(target_metric),
            explanation=explanation
        )
        
        self.prediction_history.append(result)
        
        # Keep only recent predictions
        if len(self.prediction_history) > 1000:
            self.prediction_history = self.prediction_history[-500:]
        
        return result
    
    async def detect_anomalies(self, metric_name: str) -> List[Dict[str, Any]]:
        """Detect anomalies in metric data."""
        if metric_name not in self.metrics_buffer or len(self.metrics_buffer[metric_name]) < 20:
            return []
        
        # Prepare data for anomaly detection
        samples = list(self.metrics_buffer[metric_name])
        values = np.array([[sample.value] for sample in samples])
        timestamps = [sample.timestamp for sample in samples]
        
        # Fit anomaly detector if not already done
        if not self.anomaly_fitted:
            self.anomaly_detector.fit(values)
            self.anomaly_fitted = True
        
        # Detect anomalies
        anomaly_scores = self.anomaly_detector.decision_function(values)
        anomalies = self.anomaly_detector.predict(values)
        
        detected_anomalies = []
        for i, (is_anomaly, score, timestamp, sample) in enumerate(zip(anomalies, anomaly_scores, timestamps, samples)):
            if is_anomaly == -1:  # Anomaly detected
                anomaly_data = {
                    "timestamp": timestamp,
                    "value": sample.value,
                    "anomaly_score": score,
                    "severity": self._calculate_anomaly_severity(score),
                    "context": self._get_anomaly_context(metric_name, i, samples)
                }
                detected_anomalies.append(anomaly_data)
        
        # Generate alerts for critical anomalies
        for anomaly in detected_anomalies:
            if anomaly["severity"] == AlertSeverity.CRITICAL:
                await self._create_anomaly_alert(metric_name, anomaly)
        
        return detected_anomalies
    
    async def identify_patterns(self, metric_name: str) -> Dict[str, Any]:
        """Identify patterns in metric data using clustering."""
        if metric_name not in self.metrics_buffer or len(self.metrics_buffer[metric_name]) < 30:
            return {"patterns": [], "clusters": 0}
        
        # Prepare feature matrix
        samples = list(self.metrics_buffer[metric_name])
        features = []
        
        for i in range(len(samples) - 5):  # Use sliding window of 5 samples
            window_values = [samples[j].value for j in range(i, i + 5)]
            window_features = [
                np.mean(window_values),
                np.std(window_values),
                np.max(window_values) - np.min(window_values),  # Range
                len([v for v in window_values if v > np.mean(window_values)]),  # Count above mean
                np.sum(np.diff(window_values) > 0)  # Trend direction
            ]
            features.append(window_features)
        
        if len(features) < 10:
            return {"patterns": [], "clusters": 0}
        
        features = np.array(features)
        
        # Determine optimal number of clusters
        n_clusters = min(5, len(features) // 5)
        
        if n_clusters < 2:
            return {"patterns": [], "clusters": 0}
        
        # Apply clustering
        if metric_name not in self.cluster_models:
            self.cluster_models[metric_name] = KMeans(n_clusters=n_clusters, random_state=42)
        
        cluster_labels = self.cluster_models[metric_name].fit_predict(features)
        
        # Analyze patterns in each cluster
        patterns = []
        for cluster_id in range(n_clusters):
            cluster_features = features[cluster_labels == cluster_id]
            if len(cluster_features) > 0:
                pattern = {
                    "cluster_id": cluster_id,
                    "size": len(cluster_features),
                    "centroid": self.cluster_models[metric_name].cluster_centers_[cluster_id].tolist(),
                    "characteristics": self._describe_pattern(cluster_features),
                    "frequency": len(cluster_features) / len(features)
                }
                patterns.append(pattern)
        
        # Store pattern information
        self.pattern_clusters[metric_name] = {
            "model": self.cluster_models[metric_name],
            "patterns": patterns,
            "last_updated": time.time()
        }
        
        return {"patterns": patterns, "clusters": n_clusters}
    
    async def generate_insights(self, time_window: float = 3600.0) -> List[Dict[str, Any]]:
        """Generate actionable insights from analytics."""
        insights = []
        current_time = time.time()
        
        # Analyze each metric
        for metric_name, buffer in self.metrics_buffer.items():
            if len(buffer) < 10:
                continue
            
            # Recent samples within time window
            recent_samples = [s for s in buffer if current_time - s.timestamp <= time_window]
            
            if len(recent_samples) < 5:
                continue
            
            # Generate insights for this metric
            metric_insights = await self._generate_metric_insights(metric_name, recent_samples)
            insights.extend(metric_insights)
        
        # Cross-metric insights
        correlation_insights = await self._generate_correlation_insights(time_window)
        insights.extend(correlation_insights)
        
        # Performance insights
        performance_insights = await self._generate_performance_insights()
        insights.extend(performance_insights)
        
        # Prioritize insights
        insights.sort(key=lambda x: x.get("priority", 0.5), reverse=True)
        
        return insights[:20]  # Return top 20 insights
    
    async def get_resource_forecast(self, resource_type: str, forecast_horizon: float = 1800.0) -> ResourcePrediction:
        """Get resource usage forecast."""
        # Predict resource usage
        usage_prediction = await self.predict(
            PredictionType.RESOURCE_USAGE,
            f"{resource_type}_usage",
            forecast_horizon
        )
        
        # Predict peak usage
        peak_prediction = await self.predict(
            PredictionType.RESOURCE_USAGE,
            f"{resource_type}_peak",
            forecast_horizon
        )
        
        # Current usage
        current_usage = 0.0
        if f"{resource_type}_usage" in self.metrics_buffer:
            recent_samples = list(self.metrics_buffer[f"{resource_type}_usage"])[-5:]
            if recent_samples:
                current_usage = np.mean([s.value for s in recent_samples])
        
        # Time to peak (simplified calculation)
        time_to_peak = forecast_horizon * 0.7  # Assume peak occurs at 70% of forecast horizon
        
        # Generate recommendations
        recommendations = self._generate_resource_recommendations(
            resource_type, current_usage, usage_prediction.predicted_value, peak_prediction.predicted_value
        )
        
        return ResourcePrediction(
            resource_type=resource_type,
            current_usage=current_usage,
            predicted_usage=usage_prediction.predicted_value,
            peak_prediction=peak_prediction.predicted_value,
            time_to_peak=time_to_peak,
            confidence=min(usage_prediction.confidence, peak_prediction.confidence),
            recommendations=recommendations
        )
    
    def _prepare_features(self, target_metric: str) -> Tuple[List[List[float]], List[float]]:
        """Prepare features and targets for ML models."""
        samples = list(self.metrics_buffer[target_metric])
        
        if len(samples) < 10:
            return [], []
        
        features = []
        targets = []
        
        # Create time-series features
        for i in range(5, len(samples)):  # Use window of 5 previous samples
            # Features: previous 5 values, their statistics
            window_values = [samples[j].value for j in range(i-5, i)]
            feature_vector = [
                np.mean(window_values),
                np.std(window_values),
                np.min(window_values),
                np.max(window_values),
                window_values[-1],  # Last value
                np.mean(np.diff(window_values)),  # Trend
                len([v for v in window_values if v > np.mean(window_values)])  # Count above mean
            ]
            
            # Add time-based features
            sample = samples[i-1]
            hour_of_day = time.localtime(sample.timestamp).tm_hour
            feature_vector.extend([
                np.sin(2 * np.pi * hour_of_day / 24),  # Hour sine
                np.cos(2 * np.pi * hour_of_day / 24)   # Hour cosine
            ])
            
            features.append(feature_vector)
            targets.append(samples[i].value)
        
        return features, targets
    
    def _get_feature_names(self, target_metric: str) -> List[str]:
        """Get names of features used for prediction."""
        return [
            f"{target_metric}_mean",
            f"{target_metric}_std", 
            f"{target_metric}_min",
            f"{target_metric}_max",
            f"{target_metric}_last",
            f"{target_metric}_trend",
            f"{target_metric}_above_mean_count",
            "hour_sin",
            "hour_cos"
        ]
    
    async def _get_or_create_model(self, 
                                   prediction_type: PredictionType,
                                   target_metric: str,
                                   features: List[List[float]],
                                   targets: List[float]) -> Any:
        """Get existing model or create new one."""
        model_key = f"{prediction_type.value}_{target_metric}"
        
        if model_key not in self.models or self._should_retrain_model(model_key):
            # Create and train new model
            if prediction_type in [PredictionType.RESOURCE_USAGE, PredictionType.PERFORMANCE]:
                model = RandomForestRegressor(n_estimators=50, random_state=42)
            else:
                model = LinearRegression()
            
            # Scale features
            scaler = StandardScaler()
            scaled_features = scaler.fit_transform(features)
            
            # Train model
            model.fit(scaled_features, targets)
            
            # Store model and scaler
            self.models[model_key] = model
            self.scalers[model_key] = scaler
            
            # Evaluate model performance
            predictions = model.predict(scaled_features)
            mse = np.mean((predictions - targets) ** 2)
            mae = np.mean(np.abs(predictions - targets))
            
            # Store performance metrics
            self.model_performance[model_key].append(1.0 / (1.0 + mae))  # Accuracy score
            
            self.logger.debug(f"Trained new model for {model_key}: MAE={mae:.4f}, MSE={mse:.4f}")
        
        return self.models[model_key]
    
    def _should_retrain_model(self, model_key: str) -> bool:
        """Determine if model should be retrained."""
        if model_key not in self.models:
            return True
        
        # Check if enough time has passed
        if hasattr(self.models[model_key], '_last_trained'):
            time_since_training = time.time() - self.models[model_key]._last_trained
            if time_since_training > self.model_retrain_interval:
                return True
        
        # Check if performance has degraded
        if model_key in self.model_performance and len(self.model_performance[model_key]) > 10:
            recent_performance = np.mean(self.model_performance[model_key][-5:])
            historical_performance = np.mean(self.model_performance[model_key][-15:-5])
            
            if recent_performance < historical_performance * 0.9:  # 10% degradation
                return True
        
        return False
    
    def _calculate_prediction_confidence(self, 
                                       model_key: str,
                                       predicted_value: float,
                                       historical_values: List[float]) -> float:
        """Calculate confidence in prediction."""
        if not historical_values:
            return 0.5
        
        # Base confidence on historical model performance
        base_confidence = 0.5
        if model_key in self.model_performance and self.model_performance[model_key]:
            base_confidence = np.mean(self.model_performance[model_key][-5:])
        
        # Adjust based on prediction stability
        if len(historical_values) > 5:
            recent_values = historical_values[-5:]
            volatility = np.std(recent_values) / (np.mean(recent_values) + 1e-8)
            stability_factor = 1.0 / (1.0 + volatility)
            base_confidence *= stability_factor
        
        # Adjust based on prediction reasonableness
        if len(historical_values) > 0:
            mean_historical = np.mean(historical_values)
            if mean_historical > 0:
                deviation_factor = abs(predicted_value - mean_historical) / mean_historical
                reasonableness_factor = 1.0 / (1.0 + deviation_factor)
                base_confidence *= reasonableness_factor
        
        return min(0.95, max(0.05, base_confidence))
    
    def _generate_prediction_explanation(self,
                                       prediction_type: PredictionType,
                                       metric_name: str,
                                       predicted_value: float,
                                       confidence: float) -> str:
        """Generate human-readable explanation for prediction."""
        confidence_text = "high" if confidence > 0.8 else "medium" if confidence > 0.6 else "low"
        
        explanations = {
            PredictionType.RESOURCE_USAGE: f"Based on recent usage patterns, {metric_name} is predicted to reach {predicted_value:.2f} with {confidence_text} confidence.",
            PredictionType.PERFORMANCE: f"Performance metric {metric_name} is forecasted to be {predicted_value:.2f} based on historical trends ({confidence_text} confidence).",
            PredictionType.FAILURE_PROBABILITY: f"Failure probability for {metric_name} is estimated at {predicted_value:.1%} with {confidence_text} confidence.",
            PredictionType.COMPLETION_TIME: f"Estimated completion time for {metric_name} is {predicted_value:.0f} seconds with {confidence_text} confidence.",
            PredictionType.QUALITY_SCORE: f"Quality score for {metric_name} is predicted to be {predicted_value:.2f} with {confidence_text} confidence."
        }
        
        return explanations.get(prediction_type, f"Prediction for {metric_name}: {predicted_value:.2f} ({confidence_text} confidence)")
    
    async def _trigger_analysis(self) -> None:
        """Trigger automated analysis when buffer is full."""
        # Detect anomalies in all metrics
        for metric_name in self.metrics_buffer.keys():
            if len(self.metrics_buffer[metric_name]) >= 20:
                anomalies = await self.detect_anomalies(metric_name)
                if anomalies:
                    self.logger.info(f"Detected {len(anomalies)} anomalies in {metric_name}")
        
        # Identify patterns
        for metric_name in self.metrics_buffer.keys():
            if len(self.metrics_buffer[metric_name]) >= 30:
                patterns = await self.identify_patterns(metric_name)
                if patterns["clusters"] > 0:
                    self.logger.debug(f"Identified {patterns['clusters']} patterns in {metric_name}")
    
    def _calculate_anomaly_severity(self, anomaly_score: float) -> AlertSeverity:
        """Calculate anomaly severity based on score."""
        if anomaly_score < -0.5:
            return AlertSeverity.CRITICAL
        elif anomaly_score < -0.3:
            return AlertSeverity.WARNING
        else:
            return AlertSeverity.INFO
    
    def _get_anomaly_context(self, metric_name: str, index: int, samples: List[MetricSample]) -> Dict[str, Any]:
        """Get context information for anomaly."""
        context_window = 5
        start_idx = max(0, index - context_window)
        end_idx = min(len(samples), index + context_window + 1)
        
        context_values = [samples[i].value for i in range(start_idx, end_idx)]
        
        return {
            "metric_name": metric_name,
            "context_window": context_values,
            "position_in_sequence": index,
            "mean_context": np.mean(context_values),
            "std_context": np.std(context_values)
        }
    
    async def _create_anomaly_alert(self, metric_name: str, anomaly: Dict[str, Any]) -> None:
        """Create alert for detected anomaly."""
        alert_id = f"anomaly_{metric_name}_{int(anomaly['timestamp'])}"
        
        alert = Alert(
            id=alert_id,
            severity=anomaly["severity"],
            title=f"Anomaly detected in {metric_name}",
            description=f"Unusual value {anomaly['value']:.2f} detected with anomaly score {anomaly['anomaly_score']:.3f}",
            source="predictive_analytics"
        )
        
        self.alerts.append(alert)
        
        # Keep only recent alerts
        if len(self.alerts) > 100:
            self.alerts = self.alerts[-50:]
        
        self.logger.warning(f"Created anomaly alert: {alert.title}")
    
    def _describe_pattern(self, cluster_features: np.ndarray) -> Dict[str, Any]:
        """Describe characteristics of a pattern cluster."""
        if len(cluster_features) == 0:
            return {}
        
        mean_features = np.mean(cluster_features, axis=0)
        
        return {
            "mean_value": mean_features[0],
            "volatility": mean_features[1],
            "range": mean_features[2],
            "above_mean_tendency": mean_features[3] / 5.0,  # Normalized
            "trend_direction": mean_features[4] / 4.0,  # Normalized
            "stability": 1.0 / (1.0 + mean_features[1])  # Inverse of volatility
        }
    
    async def _generate_metric_insights(self, metric_name: str, samples: List[MetricSample]) -> List[Dict[str, Any]]:
        """Generate insights for a specific metric."""
        insights = []
        
        values = [s.value for s in samples]
        timestamps = [s.timestamp for s in samples]
        
        # Trend analysis
        if len(values) >= 5:
            trend = np.polyfit(range(len(values)), values, 1)[0]
            if abs(trend) > 0.1:
                direction = "increasing" if trend > 0 else "decreasing"
                insights.append({
                    "type": "trend",
                    "metric": metric_name,
                    "description": f"{metric_name} is {direction} with trend {trend:.3f}",
                    "priority": min(0.9, abs(trend)),
                    "actionable": True
                })
        
        # Volatility analysis
        volatility = np.std(values) / (np.mean(values) + 1e-8)
        if volatility > 0.3:
            insights.append({
                "type": "volatility",
                "metric": metric_name,
                "description": f"{metric_name} shows high volatility ({volatility:.2f})",
                "priority": min(0.8, volatility),
                "actionable": True
            })
        
        # Performance degradation
        if len(values) >= 10:
            recent_avg = np.mean(values[-5:])
            historical_avg = np.mean(values[:-5])
            if recent_avg < historical_avg * 0.9:  # 10% degradation
                degradation = (historical_avg - recent_avg) / historical_avg
                insights.append({
                    "type": "degradation",
                    "metric": metric_name,
                    "description": f"{metric_name} performance degraded by {degradation:.1%}",
                    "priority": 0.9,
                    "actionable": True
                })
        
        return insights
    
    async def _generate_correlation_insights(self, time_window: float) -> List[Dict[str, Any]]:
        """Generate insights from cross-metric correlations."""
        insights = []
        
        # Get metrics with sufficient data
        valid_metrics = {
            name: [s for s in buffer if time.time() - s.timestamp <= time_window]
            for name, buffer in self.metrics_buffer.items()
            if len(buffer) >= 10
        }
        
        if len(valid_metrics) < 2:
            return insights
        
        # Calculate correlations
        metric_names = list(valid_metrics.keys())
        for i, metric1 in enumerate(metric_names):
            for j, metric2 in enumerate(metric_names):
                if i < j:
                    values1 = [s.value for s in valid_metrics[metric1]]
                    values2 = [s.value for s in valid_metrics[metric2]]
                    
                    # Align by timestamp (simplified)
                    min_len = min(len(values1), len(values2))
                    values1 = values1[-min_len:]
                    values2 = values2[-min_len:]
                    
                    if min_len >= 5:
                        correlation = np.corrcoef(values1, values2)[0, 1]
                        
                        if abs(correlation) > 0.7:
                            relationship = "strong positive" if correlation > 0 else "strong negative"
                            insights.append({
                                "type": "correlation",
                                "metrics": [metric1, metric2],
                                "description": f"{relationship} correlation ({correlation:.3f}) between {metric1} and {metric2}",
                                "priority": abs(correlation) * 0.6,
                                "actionable": True
                            })
        
        return insights
    
    async def _generate_performance_insights(self) -> List[Dict[str, Any]]:
        """Generate performance-related insights."""
        insights = []
        
        # Model performance insights
        for model_key, performance_history in self.model_performance.items():
            if len(performance_history) >= 10:
                recent_perf = np.mean(performance_history[-5:])
                historical_perf = np.mean(performance_history[:-5])
                
                if recent_perf < historical_perf * 0.9:
                    insights.append({
                        "type": "model_degradation",
                        "model": model_key,
                        "description": f"Model {model_key} performance degraded from {historical_perf:.3f} to {recent_perf:.3f}",
                        "priority": 0.7,
                        "actionable": True
                    })
        
        # Prediction accuracy insights
        if len(self.prediction_history) >= 20:
            recent_predictions = self.prediction_history[-20:]
            avg_confidence = np.mean([p.confidence for p in recent_predictions])
            
            if avg_confidence < 0.6:
                insights.append({
                    "type": "low_prediction_confidence",
                    "description": f"Average prediction confidence is low ({avg_confidence:.2f})",
                    "priority": 0.8,
                    "actionable": True
                })
        
        return insights
    
    def _generate_resource_recommendations(self,
                                         resource_type: str,
                                         current_usage: float,
                                         predicted_usage: float,
                                         peak_prediction: float) -> List[str]:
        """Generate resource optimization recommendations."""
        recommendations = []
        
        # Usage increase recommendations
        if predicted_usage > current_usage * 1.2:
            increase_pct = (predicted_usage - current_usage) / current_usage * 100
            recommendations.append(f"Consider scaling up {resource_type} resources - predicted {increase_pct:.1f}% increase")
        
        # Peak usage recommendations
        if peak_prediction > current_usage * 1.5:
            recommendations.append(f"Prepare for peak {resource_type} usage of {peak_prediction:.2f} - consider auto-scaling")
        
        # Efficiency recommendations
        if resource_type == "cpu" and current_usage > 0.8:
            recommendations.append("High CPU usage detected - consider optimizing algorithms or adding parallel processing")
        
        elif resource_type == "memory" and current_usage > 0.9:
            recommendations.append("High memory usage - consider implementing caching strategies or memory optimization")
        
        elif resource_type == "storage" and predicted_usage > current_usage * 1.1:
            recommendations.append("Storage growth predicted - plan for capacity expansion or data archival")
        
        return recommendations


class ResourcePredictor:
    """Specialized predictor for resource management."""
    
    def __init__(self, analytics_engine: PredictiveAnalytics):
        self.logger = get_logger(__name__)
        self.analytics = analytics_engine
        
        # Resource types to monitor
        self.resource_types = ["cpu", "memory", "storage", "network", "gpu"]
        
        # Thresholds for alerts
        self.alert_thresholds = {
            "cpu": 0.8,
            "memory": 0.9,
            "storage": 0.85,
            "network": 0.7,
            "gpu": 0.85
        }
    
    async def predict_resource_demand(self, 
                                    time_horizon: float = 3600.0) -> Dict[str, ResourcePrediction]:
        """Predict resource demand for all resource types."""
        predictions = {}
        
        for resource_type in self.resource_types:
            try:
                prediction = await self.analytics.get_resource_forecast(resource_type, time_horizon)
                predictions[resource_type] = prediction
                
                # Check if intervention is needed
                if prediction.predicted_usage > self.alert_thresholds.get(resource_type, 0.8):
                    await self._create_resource_alert(resource_type, prediction)
                    
            except Exception as e:
                self.logger.error(f"Failed to predict {resource_type} demand: {str(e)}")
        
        return predictions
    
    async def optimize_resource_allocation(self, 
                                         current_allocation: Dict[str, float],
                                         predictions: Dict[str, ResourcePrediction]) -> Dict[str, float]:
        """Optimize resource allocation based on predictions."""
        optimized_allocation = current_allocation.copy()
        
        for resource_type, prediction in predictions.items():
            current = current_allocation.get(resource_type, 0.5)
            predicted = prediction.predicted_usage
            
            # Simple optimization logic
            if predicted > current * 1.2:  # Predicted 20% increase
                new_allocation = min(1.0, predicted * 1.1)  # Add 10% buffer
                optimized_allocation[resource_type] = new_allocation
                
                self.logger.info(f"Optimized {resource_type} allocation: {current:.2f} -> {new_allocation:.2f}")
            
            elif predicted < current * 0.7:  # Predicted 30% decrease
                new_allocation = max(0.1, predicted * 0.9)  # Remove some allocation but keep minimum
                optimized_allocation[resource_type] = new_allocation
                
                self.logger.info(f"Reduced {resource_type} allocation: {current:.2f} -> {new_allocation:.2f}")
        
        return optimized_allocation
    
    async def _create_resource_alert(self, resource_type: str, prediction: ResourcePrediction) -> None:
        """Create alert for resource threshold breach."""
        severity = AlertSeverity.WARNING
        if prediction.predicted_usage > 0.95:
            severity = AlertSeverity.CRITICAL
        elif prediction.predicted_usage > 0.9:
            severity = AlertSeverity.WARNING
        
        alert = Alert(
            id=f"resource_{resource_type}_{int(time.time())}",
            severity=severity,
            title=f"{resource_type.upper()} usage alert",
            description=f"Predicted {resource_type} usage ({prediction.predicted_usage:.1%}) exceeds threshold",
            source="resource_predictor"
        )
        
        self.analytics.alerts.append(alert)
        self.logger.warning(f"Created resource alert: {alert.title}")
    
    def get_resource_health_score(self, predictions: Dict[str, ResourcePrediction]) -> float:
        """Calculate overall resource health score."""
        if not predictions:
            return 0.5
        
        health_scores = []
        
        for resource_type, prediction in predictions.items():
            threshold = self.alert_thresholds.get(resource_type, 0.8)
            
            # Score based on how close to threshold
            if prediction.predicted_usage <= threshold:
                score = 1.0
            else:
                score = max(0.0, 1.0 - (prediction.predicted_usage - threshold) / (1.0 - threshold))
            
            # Weight by confidence
            weighted_score = score * prediction.confidence + 0.5 * (1 - prediction.confidence)
            health_scores.append(weighted_score)
        
        return np.mean(health_scores)
    
    def generate_capacity_plan(self, 
                              predictions: Dict[str, ResourcePrediction],
                              planning_horizon: float = 86400.0) -> Dict[str, Any]:
        """Generate capacity planning recommendations."""
        plan = {
            "planning_horizon": planning_horizon,
            "recommendations": [],
            "cost_impact": "medium",
            "urgency": "normal"
        }
        
        urgent_resources = []
        
        for resource_type, prediction in predictions.items():
            if prediction.predicted_usage > 0.9:
                urgent_resources.append(resource_type)
                plan["recommendations"].append({
                    "resource": resource_type,
                    "action": "immediate_scaling",
                    "current": prediction.current_usage,
                    "predicted": prediction.predicted_usage,
                    "recommended_capacity": prediction.predicted_usage * 1.2
                })
            
            elif prediction.predicted_usage > 0.8:
                plan["recommendations"].append({
                    "resource": resource_type,
                    "action": "planned_scaling",
                    "current": prediction.current_usage,
                    "predicted": prediction.predicted_usage,
                    "recommended_capacity": prediction.predicted_usage * 1.15
                })
        
        if urgent_resources:
            plan["urgency"] = "high"
            plan["cost_impact"] = "high"
        
        return plan