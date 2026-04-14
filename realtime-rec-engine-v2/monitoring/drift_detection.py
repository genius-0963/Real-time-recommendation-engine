"""
Production-grade model drift detection for Netflix/Meta scale recommendation engine.
Implements PSI, KL divergence, statistical testing, and automated retraining triggers.
"""

import logging
import time
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from concurrent.futures import ThreadPoolExecutor
import json
import pickle
from scipy import stats
from scipy.stats import chi2_contingency
import asyncio
from collections import defaultdict, deque
import warnings

warnings.filterwarnings("ignore")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class DriftDetectionConfig:
    """Configuration for drift detection system."""
    psi_threshold: float = 0.1
    kl_threshold: float = 0.1
    ks_threshold: float = 0.05
    window_size: int = 10000
    reference_window_size: int = 50000
    min_samples: int = 1000
    detection_interval_seconds: int = 300  # 5 minutes
    alert_cooldown_seconds: int = 1800  # 30 minutes
    feature_importance_threshold: float = 0.01
    drift_persistence_threshold: int = 3  # Number of consecutive detections
    false_positive_mitigation: bool = True
    statistical_significance: float = 0.05
    enable_explainability: bool = True


@dataclass
class DriftMetrics:
    """Drift detection metrics."""
    psi_score: float = 0.0
    kl_divergence: float = 0.0
    ks_statistic: float = 0.0
    ks_p_value: float = 0.0
    wasserstein_distance: float = 0.0
    population_stability_index: float = 0.0
    feature_drift_scores: Dict[str, float] = field(default_factory=dict)
    drift_detected: bool = False
    confidence: float = 0.0
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class DriftAlert:
    """Drift alert information."""
    alert_id: str
    model_name: str
    model_version: str
    drift_type: str
    severity: str  # 'low', 'medium', 'high', 'critical'
    metrics: DriftMetrics
    affected_features: List[str]
    business_impact: Dict[str, Any]
    recommended_action: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class PopulationStabilityIndex:
    """PSI calculation for drift detection."""
    
    @staticmethod
    def calculate_psi(expected: np.ndarray, actual: np.ndarray, 
                     bucket_type: str = 'quantiles', 
                     num_buckets: int = 10) -> Tuple[float, np.ndarray, np.ndarray]:
        """Calculate Population Stability Index."""
        # Create buckets
        if bucket_type == 'quantiles':
            _, bins = pd.qcut(expected, q=num_buckets, retbins=True, duplicates='drop')
        else:
            _, bins = pd.cut(expected, bins=num_bins, retbins=True, duplicates='drop')
        
        # Calculate frequencies
        expected_counts = np.histogram(expected, bins=bins)[0]
        actual_counts = np.histogram(actual, bins=bins)[0]
        
        # Convert to percentages
        expected_perc = expected_counts / len(expected)
        actual_perc = actual_counts / len(actual)
        
        # Avoid division by zero
        expected_perc = np.where(expected_perc == 0, 0.0001, expected_perc)
        actual_perc = np.where(actual_perc == 0, 0.0001, actual_perc)
        
        # Calculate PSI
        psi_values = (actual_perc - expected_perc) * np.log(actual_perc / expected_perc)
        psi_total = np.sum(psi_values)
        
        return psi_total, expected_perc, actual_perc


class KLDivergenceCalculator:
    """KL divergence calculation for drift detection."""
    
    @staticmethod
    def calculate_kl_divergence(p: np.ndarray, q: np.ndarray, 
                              epsilon: float = 1e-10) -> float:
        """Calculate KL divergence D(P||Q)."""
        # Add small epsilon to avoid log(0)
        p = p + epsilon
        q = q + epsilon
        
        # Normalize
        p = p / np.sum(p)
        q = q / np.sum(q)
        
        # Calculate KL divergence
        kl_div = np.sum(p * np.log(p / q))
        
        return kl_div


class StatisticalTests:
    """Statistical tests for drift detection."""
    
    @staticmethod
    def ks_test(reference: np.ndarray, current: np.ndarray) -> Tuple[float, float]:
        """Two-sample Kolmogorov-Smirnov test."""
        ks_stat, p_value = stats.ks_2samp(reference, current)
        return ks_stat, p_value
    
    @staticmethod
    def chi_square_test(reference: np.ndarray, current: np.ndarray, 
                        bins: int = 10) -> Tuple[float, float]:
        """Chi-square test for categorical distributions."""
        # Create bins for both distributions
        all_data = np.concatenate([reference, current])
        _, bin_edges = pd.cut(all_data, bins=bins, retbins=True, duplicates='drop')
        
        # Calculate frequencies
        ref_counts = np.histogram(reference, bins=bin_edges)[0]
        cur_counts = np.histogram(current, bins=bin_edges)[0]
        
        # Chi-square test
        chi2_stat, p_value, _, _ = chi2_contingency([ref_counts, cur_counts])
        
        return chi2_stat, p_value
    
    @staticmethod
    def wasserstein_distance(reference: np.ndarray, current: np.ndarray) -> float:
        """Wasserstein distance between distributions."""
        return stats.wasserstein_distance(reference, current)


class FeatureDriftDetector:
    """Detect drift for individual features."""
    
    def __init__(self, config: DriftDetectionConfig):
        self.config = config
        self.psi_calculator = PopulationStabilityIndex()
        self.kl_calculator = KLDivergenceCalculator()
        self.statistical_tests = StatisticalTests()
    
    def detect_feature_drift(self, feature_name: str, 
                            reference_data: np.ndarray, 
                            current_data: np.ndarray) -> DriftMetrics:
        """Detect drift for a single feature."""
        if len(reference_data) < self.config.min_samples or len(current_data) < self.config.min_samples:
            return DriftMetrics()
        
        try:
            # Calculate PSI
            psi_score, _, _ = self.psi_calculator.calculate_psi(
                reference_data, current_data, num_buckets=10
            )
            
            # Calculate KL divergence (for continuous features)
            if len(np.unique(reference_data)) > 10:  # Continuous feature
                # Create histograms
                hist_ref, bin_edges = np.histogram(reference_data, bins=50, density=True)
                hist_cur, _ = np.histogram(current_data, bins=bin_edges, density=True)
                
                kl_divergence = self.kl_calculator.calculate_kl_divergence(hist_ref, hist_cur)
            else:
                kl_divergence = 0.0
            
            # Statistical tests
            ks_stat, ks_p_value = self.statistical_tests.ks_test(reference_data, current_data)
            wasserstein_dist = self.statistical_tests.wasserstein_distance(reference_data, current_data)
            
            # Determine if drift is detected
            drift_detected = (
                psi_score > self.config.psi_threshold or
                kl_divergence > self.config.kl_threshold or
                ks_p_value < self.config.ks_threshold
            )
            
            # Calculate confidence based on multiple metrics
            confidence = self._calculate_confidence(
                psi_score, kl_divergence, ks_p_value, wasserstein_dist
            )
            
            return DriftMetrics(
                psi_score=psi_score,
                kl_divergence=kl_divergence,
                ks_statistic=ks_stat,
                ks_p_value=ks_p_value,
                wasserstein_distance=wasserstein_dist,
                population_stability_index=psi_score,
                drift_detected=drift_detected,
                confidence=confidence
            )
            
        except Exception as e:
            logger.error(f"Error detecting drift for feature {feature_name}: {e}")
            return DriftMetrics()
    
    def _calculate_confidence(self, psi_score: float, kl_divergence: float, 
                            ks_p_value: float, wasserstein_dist: float) -> float:
        """Calculate confidence score for drift detection."""
        # Normalize each metric to [0, 1]
        psi_normalized = min(psi_score / self.config.psi_threshold, 1.0)
        kl_normalized = min(kl_divergence / self.config.kl_threshold, 1.0)
        ks_normalized = 1.0 - ks_p_value  # Lower p-value means more significant
        
        # Weighted average (adjust weights based on domain knowledge)
        confidence = (0.4 * psi_normalized + 0.3 * kl_normalized + 0.3 * ks_normalized)
        
        return min(confidence, 1.0)


class ModelDriftDetector:
    """Main drift detection system for recommendation models."""
    
    def __init__(self, config: DriftDetectionConfig):
        self.config = config
        self.feature_detector = FeatureDriftDetector(config)
        
        # State management
        self.reference_data: Dict[str, np.ndarray] = {}
        self.current_data: Dict[str, deque] = defaultdict(lambda: deque(maxlen=config.window_size))
        self.drift_history: Dict[str, List[DriftMetrics]] = defaultdict(list)
        self.alert_history: List[DriftAlert] = []
        self.last_alert_time: Dict[str, datetime] = {}
        
        # Performance tracking
        self.performance_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        
        # Thread pool for parallel processing
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # Drift persistence tracking
        self.consecutive_detections: Dict[str, int] = defaultdict(int)
        
    def set_reference_data(self, model_name: str, features: Dict[str, np.ndarray]):
        """Set reference data for drift comparison."""
        self.reference_data[model_name] = features
        logger.info(f"Set reference data for model {model_name} with {len(features)} features")
    
    def add_current_data(self, model_name: str, features: Dict[str, np.ndarray]):
        """Add current data for drift detection."""
        for feature_name, values in features.items():
            self.current_data[model_name][feature_name].extend(values)
    
    def add_performance_data(self, model_name: str, metric_name: str, value: float):
        """Add performance metrics for business impact analysis."""
        self.performance_history[model_name][metric_name].append(value)
    
    def detect_drift(self, model_name: str, model_version: str) -> Optional[DriftAlert]:
        """Detect drift for a specific model."""
        if model_name not in self.reference_data:
            logger.warning(f"No reference data available for model {model_name}")
            return None
        
        # Check if we have enough current data
        current_features = {}
        for feature_name in self.reference_data[model_name].keys():
            current_data = list(self.current_data[model_name][feature_name])
            if len(current_data) < self.config.min_samples:
                continue
            current_features[feature_name] = np.array(current_data)
        
        if not current_features:
            logger.warning(f"Insufficient current data for drift detection")
            return None
        
        # Check alert cooldown
        if self._is_in_cooldown(model_name):
            return None
        
        # Detect drift for each feature in parallel
        feature_drift_results = {}
        futures = {}
        
        for feature_name, reference_values in self.reference_data[model_name].items():
            if feature_name in current_features:
                future = self.executor.submit(
                    self.feature_detector.detect_feature_drift,
                    feature_name,
                    reference_values,
                    current_features[feature_name]
                )
                futures[feature_name] = future
        
        # Collect results
        for feature_name, future in futures.items():
            try:
                feature_drift_results[feature_name] = future.result()
            except Exception as e:
                logger.error(f"Error processing feature {feature_name}: {e}")
        
        # Aggregate results
        overall_drift = self._aggregate_drift_results(feature_drift_results)
        
        # Check if drift is detected
        if overall_drift.drift_detected:
            # Apply false positive mitigation
            if self.config.false_positive_mitigation:
                if not self._confirm_drift_persistence(model_name, overall_drift):
                    return None
            
            # Create alert
            alert = self._create_alert(model_name, model_version, overall_drift, feature_drift_results)
            
            # Update state
            self.alert_history.append(alert)
            self.last_alert_time[model_name] = datetime.now(timezone.utc)
            self.consecutive_detections[model_name] += 1
            
            logger.warning(f"Drift detected for model {model_name}: PSI={overall_drift.psi_score:.4f}, "
                         f"KL={overall_drift.kl_divergence:.4f}")
            
            return alert
        else:
            # Reset consecutive detections
            self.consecutive_detections[model_name] = 0
        
        return None
    
    def _aggregate_drift_results(self, feature_results: Dict[str, DriftMetrics]) -> DriftMetrics:
        """Aggregate drift results across features."""
        if not feature_results:
            return DriftMetrics()
        
        # Weight features by importance (simplified - could use feature importance from model)
        num_features = len(feature_results)
        
        # Aggregate metrics
        psi_scores = [r.psi_score for r in feature_results.values()]
        kl_divergences = [r.kl_divergence for r in feature_results.values()]
        ks_statistics = [r.ks_statistic for r in feature_results.values()]
        ks_p_values = [r.ks_p_value for r in feature_results.values()]
        wasserstein_distances = [r.wasserstein_distance for r in feature_results.values()]
        
        # Calculate weighted averages
        overall_psi = np.mean(psi_scores)
        overall_kl = np.mean(kl_divergences)
        overall_ks = np.mean(ks_statistics)
        overall_ks_p = np.mean(ks_p_values)
        overall_wasserstein = np.mean(wasserstein_distances)
        
        # Determine if overall drift is detected
        drift_detected = (
            overall_psi > self.config.psi_threshold or
            overall_kl > self.config.kl_threshold or
            overall_ks_p < self.config.ks_threshold
        )
        
        # Calculate overall confidence
        confidences = [r.confidence for r in feature_results.values()]
        overall_confidence = np.mean(confidences)
        
        return DriftMetrics(
            psi_score=overall_psi,
            kl_divergence=overall_kl,
            ks_statistic=overall_ks,
            ks_p_value=overall_ks_p,
            wasserstein_distance=overall_wasserstein,
            population_stability_index=overall_psi,
            feature_drift_scores={name: r.psi_score for name, r in feature_results.items()},
            drift_detected=drift_detected,
            confidence=overall_confidence
        )
    
    def _confirm_drift_persistence(self, model_name: str, drift_metrics: DriftMetrics) -> bool:
        """Confirm drift persistence to reduce false positives."""
        # Check if we have consecutive detections
        if self.consecutive_detections[model_name] >= self.config.drift_persistence_threshold:
            return True
        
        # Check if drift is severe enough to bypass persistence check
        if drift_metrics.psi_score > self.config.psi_threshold * 2:
            return True
        
        # Check if confidence is high
        if drift_metrics.confidence > 0.8:
            return True
        
        return False
    
    def _is_in_cooldown(self, model_name: str) -> bool:
        """Check if model is in alert cooldown period."""
        if model_name not in self.last_alert_time:
            return False
        
        time_since_last = datetime.now(timezone.utc) - self.last_alert_time[model_name]
        return time_since_last.total_seconds() < self.config.alert_cooldown_seconds
    
    def _create_alert(self, model_name: str, model_version: str, 
                    drift_metrics: DriftMetrics, 
                    feature_results: Dict[str, DriftMetrics]) -> DriftAlert:
        """Create drift alert."""
        # Determine severity
        severity = self._determine_severity(drift_metrics)
        
        # Identify affected features
        affected_features = [
            name for name, result in feature_results.items() 
            if result.drift_detected
        ]
        
        # Calculate business impact
        business_impact = self._calculate_business_impact(model_name, drift_metrics)
        
        # Recommend action
        recommended_action = self._recommend_action(severity, drift_metrics, business_impact)
        
        return DriftAlert(
            alert_id=f"drift_{model_name}_{int(time.time())}",
            model_name=model_name,
            model_version=model_version,
            drift_type="feature_drift",
            severity=severity,
            metrics=drift_metrics,
            affected_features=affected_features,
            business_impact=business_impact,
            recommended_action=recommended_action
        )
    
    def _determine_severity(self, drift_metrics: DriftMetrics) -> str:
        """Determine alert severity based on drift metrics."""
        psi_score = drift_metrics.psi_score
        confidence = drift_metrics.confidence
        
        if psi_score > 0.5 and confidence > 0.8:
            return "critical"
        elif psi_score > 0.25 and confidence > 0.6:
            return "high"
        elif psi_score > 0.1 and confidence > 0.4:
            return "medium"
        else:
            return "low"
    
    def _calculate_business_impact(self, model_name: str, drift_metrics: DriftMetrics) -> Dict[str, Any]:
        """Calculate business impact of drift."""
        impact = {
            "estimated_ctr_drop": 0.0,
            "estimated_engagement_drop": 0.0,
            "estimated_revenue_impact": 0.0,
            "confidence": drift_metrics.confidence
        }
        
        # Get recent performance data
        if model_name in self.performance_history:
            recent_ctr = list(self.performance_history[model_name].get("ctr", []))
            recent_engagement = list(self.performance_history[model_name].get("engagement_rate", []))
            
            if recent_ctr:
                # Estimate drop based on drift severity
                estimated_drop = drift_metrics.psi_score * 0.1  # 10% drop at max PSI
                impact["estimated_ctr_drop"] = estimated_drop
            
            if recent_engagement:
                estimated_drop = drift_metrics.psi_score * 0.08
                impact["estimated_engagement_drop"] = estimated_drop
        
        return impact
    
    def _recommend_action(self, severity: str, drift_metrics: DriftMetrics, 
                         business_impact: Dict[str, Any]) -> str:
        """Recommend action based on drift severity and impact."""
        if severity == "critical":
            return "immediate_retraining"
        elif severity == "high":
            return "scheduled_retraining"
        elif severity == "medium":
            return "monitor_closely"
        else:
            return "continue_monitoring"
    
    def get_drift_summary(self, model_name: str) -> Dict[str, Any]:
        """Get drift detection summary for a model."""
        recent_alerts = [
            alert for alert in self.alert_history 
            if alert.model_name == model_name and 
            (datetime.now(timezone.utc) - alert.timestamp).days <= 7
        ]
        
        return {
            "model_name": model_name,
            "total_alerts": len(self.alert_history),
            "recent_alerts": len(recent_alerts),
            "last_alert": self.last_alert_time.get(model_name),
            "consecutive_detections": self.consecutive_detections.get(model_name, 0),
            "current_drift_status": self._get_current_drift_status(model_name)
        }
    
    def _get_current_drift_status(self, model_name: str) -> str:
        """Get current drift status for a model."""
        if model_name not in self.consecutive_detections:
            return "no_data"
        
        detections = self.consecutive_detections[model_name]
        if detections >= 3:
            return "critical_drift"
        elif detections >= 2:
            return "moderate_drift"
        elif detections >= 1:
            return "minor_drift"
        else:
            return "stable"
    
    def cleanup(self):
        """Cleanup resources."""
        self.executor.shutdown(wait=True)


class AutomatedRetrainingTrigger:
    """Automated retraining trigger based on drift detection."""
    
    def __init__(self, drift_detector: ModelDriftDetector):
        self.drift_detector = drift_detector
        self.retraining_queue = asyncio.Queue()
        self.is_running = False
    
    async def start_monitoring(self):
        """Start automated retraining monitoring."""
        self.is_running = True
        
        while self.is_running:
            try:
                # Check for models that need retraining
                await self._check_retraining_triggers()
                
                # Wait before next check
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                logger.error(f"Error in retraining monitoring: {e}")
                await asyncio.sleep(60)
    
    async def _check_retraining_triggers(self):
        """Check for models that need retraining."""
        # This would integrate with your model training pipeline
        # For now, just log the alerts that would trigger retraining
        
        for alert in self.drift_detector.alert_history[-10:]:  # Check last 10 alerts
            if alert.recommended_action == "immediate_retraining":
                logger.info(f"Triggering immediate retraining for model {alert.model_name}")
                await self.retraining_queue.put({
                    "model_name": alert.model_name,
                    "model_version": alert.model_version,
                    "reason": "drift_detection",
                    "severity": alert.severity,
                    "metrics": alert.metrics
                })
    
    async def get_retraining_jobs(self) -> List[Dict[str, Any]]:
        """Get queued retraining jobs."""
        jobs = []
        while not self.retraining_queue.empty():
            try:
                job = self.retraining_queue.get_nowait()
                jobs.append(job)
            except asyncio.QueueEmpty:
                break
        return jobs
    
    def stop(self):
        """Stop monitoring."""
        self.is_running = False


# Example usage
def main():
    """Example usage of drift detection system."""
    # Configuration
    config = DriftDetectionConfig(
        psi_threshold=0.1,
        kl_threshold=0.1,
        window_size=1000,
        detection_interval_seconds=60
    )
    
    # Initialize drift detector
    drift_detector = ModelDriftDetector(config)
    
    # Generate sample reference data
    np.random.seed(42)
    reference_features = {
        "user_age": np.random.normal(35, 10, 5000),
        "item_price": np.random.lognormal(3, 0.5, 5000),
        "user_rating": np.random.uniform(1, 5, 5000)
    }
    
    # Set reference data
    drift_detector.set_reference_data("recommendation_model_v1", reference_features)
    
    # Simulate current data with drift
    current_features = {
        "user_age": np.random.normal(40, 12, 1000),  # Drifted mean
        "item_price": np.random.lognormal(3.2, 0.6, 1000),  # Drifted parameters
        "user_rating": np.random.uniform(1.2, 5.2, 1000)  # Slight drift
    }
    
    # Add current data
    drift_detector.add_current_data("recommendation_model_v1", current_features)
    
    # Detect drift
    alert = drift_detector.detect_drift("recommendation_model_v1", "v1.0.0")
    
    if alert:
        print(f"Drift Alert: {alert.severity} - {alert.recommended_action}")
        print(f"PSI Score: {alert.metrics.psi_score:.4f}")
        print(f"Affected Features: {alert.affected_features}")
    else:
        print("No drift detected")
    
    # Get summary
    summary = drift_detector.get_drift_summary("recommendation_model_v1")
    print(f"Drift Summary: {summary}")
    
    # Cleanup
    drift_detector.cleanup()


if __name__ == "__main__":
    main()
