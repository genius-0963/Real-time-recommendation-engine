"""
Chaos testing for Netflix/Meta scale recommendation engine
Simulates real-world failures and tests system resilience
"""

import time
import random
import logging
import asyncio
import subprocess
import json
import yaml
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
import requests
import kubernetes
from kubernetes import client, config
import psycopg2
import redis
from kafka import KafkaProducer, KafkaConsumer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ChaosExperiment(Enum):
    """Types of chaos experiments."""
    POD_DELETION = "pod_deletion"
    NETWORK_PARTITION = "network_partition"
    CPU_PRESSURE = "cpu_pressure"
    MEMORY_PRESSURE = "memory_pressure"
    DISK_PRESSURE = "disk_pressure"
    KAFKA_BROKER_FAILURE = "kafka_broker_failure"
    REDIS_NODE_FAILURE = "redis_node_failure"
    DATABASE_CONNECTION_FAILURE = "database_connection_failure"
    DNS_FAILURE = "dns_failure"
    LOAD_BALANCER_FAILURE = "load_balancer_failure"


@dataclass
class ChaosConfig:
    """Configuration for chaos testing."""
    namespace: str = "rec-engine-prod"
    kubeconfig_path: Optional[str] = None
    experiment_duration: int = 300  # 5 minutes
    recovery_time: int = 600  # 10 minutes
    monitoring_interval: int = 10  # seconds
    baseline_duration: int = 60  # 1 minute
    max_failure_rate: float = 0.05  # 5%
    max_latency_increase: float = 2.0  # 2x increase
    max_qps_drop: float = 0.3  # 30% drop
    
    # Service endpoints
    api_endpoint: str = "https://api.rec-engine.company.com"
    prometheus_endpoint: str = "http://prometheus.rec-engine-prod.svc.cluster.local:9090"
    grafana_endpoint: str = "http://grafana.rec-engine-prod.svc.cluster.local:3000"
    
    # Database connection
    postgres_host: str = "postgres.rec-engine-prod.svc.cluster.local"
    postgres_port: int = 5432
    postgres_database: str = "rec_engine_prod"
    
    # Redis connection
    redis_host: str = "redis-cluster.rec-engine-prod.svc.cluster.local"
    redis_port: int = 6379
    
    # Kafka connection
    kafka_bootstrap_servers: List[str] = field(default_factory=lambda: [
        "kafka-broker-1.rec-engine-prod.svc.cluster.local:9092",
        "kafka-broker-2.rec-engine-prod.svc.cluster.local:9092",
        "kafka-broker-3.rec-engine-prod.svc.cluster.local:9092"
    ])


@dataclass
class ExperimentResult:
    """Results of a chaos experiment."""
    experiment_type: ChaosExperiment
    start_time: datetime
    end_time: datetime
    duration: int
    success: bool
    baseline_metrics: Dict[str, float]
    chaos_metrics: Dict[str, float]
    recovery_metrics: Dict[str, float]
    impact_assessment: Dict[str, Any]
    failure_details: Optional[str] = None


class MetricsCollector:
    """Collects metrics during chaos experiments."""
    
    def __init__(self, config: ChaosConfig):
        self.config = config
        self.prometheus_url = f"{config.prometheus_endpoint}/api/v1/query"
    
    def query_metric(self, query: str) -> float:
        """Query a single metric from Prometheus."""
        try:
            response = requests.get(f"{self.prometheus_url}?query={query}")
            response.raise_for_status()
            data = response.json()
            
            if data['data']['result']:
                return float(data['data']['result'][0]['value'][1])
            return 0.0
        except Exception as e:
            logger.error(f"Failed to query metric {query}: {e}")
            return 0.0
    
    def collect_baseline_metrics(self) -> Dict[str, float]:
        """Collect baseline metrics before chaos."""
        return {
            'qps': self.query_metric('rate(recommendation_requests_total[1m])'),
            'p95_latency': self.query_metric('histogram_quantile(0.95, rate(request_latency_seconds_bucket[1m]))'),
            'p99_latency': self.query_metric('histogram_quantile(0.99, rate(request_latency_seconds_bucket[1m]))'),
            'error_rate': self.query_metric('rate(http_requests_total{status=~"5.."}[1m]) / rate(http_requests_total[1m])'),
            'cache_hit_rate': self.query_metric('rate(cache_hits_total[1m]) / (rate(cache_hits_total[1m]) + rate(cache_misses_total[1m]))'),
            'cpu_usage': self.query_metric('avg by (pod) (rate(container_cpu_usage_seconds_total[1m]))'),
            'memory_usage': self.query_metric('avg by (pod) (container_memory_usage_bytes[1m]) / 1024/1024/1024)'),
            'kafka_lag': self.query_metric('sum(kafka_consumer_lag_sum)'),
            'redis_connections': self.query_metric('sum(redis_connected_clients)')
        }
    
    def collect_chaos_metrics(self) -> Dict[str, float]:
        """Collect metrics during chaos."""
        return self.collect_baseline_metrics()  # Same metrics, different values
    
    def collect_recovery_metrics(self) -> Dict[str, float]:
        """Collect metrics after chaos recovery."""
        return self.collect_baseline_metrics()  # Same metrics, different values


class KubernetesChaos:
    """Kubernetes-based chaos experiments."""
    
    def __init__(self, config: ChaosConfig):
        self.config = config
        try:
            if config.kubeconfig_path:
                config.load_kube_config(config_file=config.kubeconfig_path)
            else:
                config.load_incluster_config()
            self.v1 = client.CoreV1Api()
            self.apps_v1 = client.AppsV1Api()
            self.networking_v1 = client.NetworkingV1Api()
        except Exception as e:
            logger.error(f"Failed to initialize Kubernetes client: {e}")
            raise
    
    def delete_random_pod(self, app_label: str, count: int = 1) -> List[str]:
        """Delete random pods."""
        deleted_pods = []
        
        try:
            pods = self.v1.list_namespaced_pod(
                namespace=self.config.namespace,
                label_selector=f"app={app_label}"
            )
            
            if len(pods.items) < count:
                logger.warning(f"Only {len(pods.items)} pods available, requested {count}")
                count = len(pods.items)
            
            selected_pods = random.sample(pods.items, count)
            
            for pod in selected_pods:
                logger.info(f"Deleting pod: {pod.metadata.name}")
                self.v1.delete_namespaced_pod(
                    name=pod.metadata.name,
                    namespace=self.config.namespace
                )
                deleted_pods.append(pod.metadata.name)
            
            return deleted_pods
            
        except Exception as e:
            logger.error(f"Failed to delete pods: {e}")
            return []
    
    def create_network_partition(self, source_pod: str, target_pod: str) -> bool:
        """Create network partition between pods."""
        try:
            # Use network policy to create partition
            network_policy = {
                "apiVersion": "networking.k8s.io/v1",
                "kind": "NetworkPolicy",
                "metadata": {
                    "name": f"chaos-partition-{int(time.time())}",
                    "namespace": self.config.namespace
                },
                "spec": {
                    "podSelector": {
                        "matchLabels": {
                            "app": source_pod.split('-')[0]  # Extract app name
                        }
                    },
                    "policyTypes": ["Egress"],
                    "egress": [
                        {
                            "to": [
                                {
                                    "podSelector": {
                                        "matchLabels": {
                                            "app": target_pod.split('-')[0]
                                        }
                                    }
                                }
                            ],
                            "action": "Deny"
                        }
                    ]
                }
            }
            
            self.networking_v1.create_namespaced_network_policy(
                namespace=self.config.namespace,
                body=network_policy
            )
            
            logger.info(f"Created network partition between {source_pod} and {target_pod}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create network partition: {e}")
            return False
    
    def cleanup_network_policy(self, policy_name: str) -> bool:
        """Clean up network policy."""
        try:
            self.networking_v1.delete_namespaced_network_policy(
                name=policy_name,
                namespace=self.config.namespace
            )
            return True
        except Exception as e:
            logger.error(f"Failed to cleanup network policy: {e}")
            return False
    
    def apply_cpu_pressure(self, pod_name: str, cpu_load: float) -> bool:
        """Apply CPU pressure to a pod."""
        try:
            # Use stress-ng to apply CPU pressure
            command = [
                "kubectl", "exec", pod_name, "-n", self.config.namespace, "--",
                "sh", "-c", f"stress-ng --cpu {int(cpu_load)} --timeout 300s &"
            ]
            
            subprocess.run(command, check=True)
            logger.info(f"Applied CPU pressure {cpu_load} to pod {pod_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to apply CPU pressure: {e}")
            return False
    
    def apply_memory_pressure(self, pod_name: str, memory_mb: int) -> bool:
        """Apply memory pressure to a pod."""
        try:
            # Use stress-ng to apply memory pressure
            command = [
                "kubectl", "exec", pod_name, "-n", self.config.namespace, "--",
                "sh", "-c", f"stress-ng --vm {int(memory_mb/100)} --vm-bytes {memory_mb}M --timeout 300s &"
            ]
            
            subprocess.run(command, check=True)
            logger.info(f"Applied memory pressure {memory_mb}MB to pod {pod_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to apply memory pressure: {e}")
            return False


class DatabaseChaos:
    """Database chaos experiments."""
    
    def __init__(self, config: ChaosConfig):
        self.config = config
    
    def simulate_connection_failure(self, duration: int) -> bool:
        """Simulate database connection failure."""
        try:
            # This would typically involve network policies or iptables rules
            # For demonstration, we'll simulate by blocking connections
            logger.info(f"Simulating database connection failure for {duration}s")
            
            # In a real implementation, you might:
            # 1. Add iptables rules to block connections
            # 2. Scale down the database deployment
            # 3. Modify service endpoints
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to simulate database connection failure: {e}")
            return False


class RedisChaos:
    """Redis chaos experiments."""
    
    def __init__(self, config: ChaosConfig):
        self.config = config
        self.redis_client = None
    
    def connect(self):
        """Connect to Redis."""
        try:
            self.redis_client = redis.Redis(
                host=self.config.redis_host,
                port=self.config.redis_port,
                decode_responses=True
            )
            self.redis_client.ping()
            return True
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            return False
    
    def simulate_node_failure(self, node_id: int) -> bool:
        """Simulate Redis node failure."""
        try:
            # This would typically involve stopping a Redis node
            logger.info(f"Simulating Redis node {node_id} failure")
            
            # In a real implementation:
            # 1. Scale down a specific Redis pod
            # 2. Block network access to the node
            # 3. Stop the Redis process
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to simulate Redis node failure: {e}")
            return False
    
    def flush_cache(self) -> bool:
        """Flush Redis cache."""
        try:
            if self.redis_client:
                self.redis_client.flushall()
                logger.info("Flushed Redis cache")
                return True
            return False
        except Exception as e:
            logger.error(f"Failed to flush Redis cache: {e}")
            return False


class KafkaChaos:
    """Kafka chaos experiments."""
    
    def __init__(self, config: ChaosConfig):
        self.config = config
    
    def simulate_broker_failure(self, broker_id: int) -> bool:
        """Simulate Kafka broker failure."""
        try:
            broker_host = f"kafka-broker-{broker_id}.{self.config.namespace}.svc.cluster.local"
            
            # This would typically involve stopping the broker
            logger.info(f"Simulating Kafka broker {broker_id} failure ({broker_host})")
            
            # In a real implementation:
            # 1. Scale down the broker deployment
            # 2. Block network access
            # 3. Stop the Kafka process
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to simulate Kafka broker failure: {e}")
            return False
    
    def produce_test_messages(self, topic: str, count: int = 100) -> bool:
        """Produce test messages to verify Kafka functionality."""
        try:
            producer = KafkaProducer(
                bootstrap_servers=self.config.kafka_bootstrap_servers,
                value_serializer=lambda v: json.dumps(v).encode('utf-8')
            )
            
            for i in range(count):
                message = {
                    "test_id": f"chaos_test_{int(time.time())}",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "message_id": i
                }
                
                producer.send(topic, message)
            
            producer.flush()
            producer.close()
            
            logger.info(f"Produced {count} test messages to topic {topic}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to produce test messages: {e}")
            return False


class ChaosTester:
    """Main chaos testing orchestrator."""
    
    def __init__(self, config: ChaosConfig):
        self.config = config
        self.metrics_collector = MetricsCollector(config)
        self.k8s_chaos = KubernetesChaos(config)
        self.db_chaos = DatabaseChaos(config)
        self.redis_chaos = RedisChaos(config)
        self.kafka_chaos = KafkaChaos(config)
        self.results: List[ExperimentResult] = []
    
    def run_experiment(self, experiment_type: ChaosExperiment) -> ExperimentResult:
        """Run a chaos experiment."""
        logger.info(f"Starting chaos experiment: {experiment_type.value}")
        
        start_time = datetime.now(timezone.utc)
        
        try:
            # Collect baseline metrics
            logger.info("Collecting baseline metrics...")
            baseline_metrics = self.metrics_collector.collect_baseline_metrics()
            
            # Wait for baseline to stabilize
            time.sleep(self.config.baseline_duration)
            
            # Execute chaos experiment
            logger.info(f"Executing {experiment_type.value} experiment...")
            success = self._execute_experiment(experiment_type)
            
            # Monitor during chaos
            logger.info("Monitoring during chaos...")
            chaos_metrics = self._monitor_during_chaos(self.config.experiment_duration)
            
            # Stop chaos and allow recovery
            logger.info("Stopping chaos and allowing recovery...")
            self._stop_chaos(experiment_type)
            
            # Monitor recovery
            logger.info("Monitoring recovery...")
            time.sleep(self.config.recovery_time)
            recovery_metrics = self.metrics_collector.collect_recovery_metrics()
            
            # Assess impact
            impact_assessment = self._assess_impact(baseline_metrics, chaos_metrics, recovery_metrics)
            
            end_time = datetime.now(timezone.utc)
            duration = int((end_time - start_time).total_seconds())
            
            result = ExperimentResult(
                experiment_type=experiment_type,
                start_time=start_time,
                end_time=end_time,
                duration=duration,
                success=success,
                baseline_metrics=baseline_metrics,
                chaos_metrics=chaos_metrics,
                recovery_metrics=recovery_metrics,
                impact_assessment=impact_assessment
            )
            
            self.results.append(result)
            
            logger.info(f"Chaos experiment {experiment_type.value} completed")
            return result
            
        except Exception as e:
            logger.error(f"Chaos experiment {experiment_type.value} failed: {e}")
            
            # Clean up any remaining chaos
            self._stop_chaos(experiment_type)
            
            end_time = datetime.now(timezone.utc)
            duration = int((end_time - start_time).total_seconds())
            
            result = ExperimentResult(
                experiment_type=experiment_type,
                start_time=start_time,
                end_time=end_time,
                duration=duration,
                success=False,
                baseline_metrics={},
                chaos_metrics={},
                recovery_metrics={},
                impact_assessment={},
                failure_details=str(e)
            )
            
            self.results.append(result)
            return result
    
    def _execute_experiment(self, experiment_type: ChaosExperiment) -> bool:
        """Execute the specific chaos experiment."""
        if experiment_type == ChaosExperiment.POD_DELETION:
            deleted_pods = self.k8s_chaos.delete_random_pod("rec-engine-api", count=2)
            return len(deleted_pods) > 0
        
        elif experiment_type == ChaosExperiment.NETWORK_PARTITION:
            return self.k8s_chaos.create_network_partition("rec-engine-api", "rec-engine-redis")
        
        elif experiment_type == ChaosExperiment.CPU_PRESSURE:
            pods = self.k8s_chaos.v1.list_namespaced_pod(
                namespace=self.config.namespace,
                label_selector="app=rec-engine-api"
            )
            if pods.items:
                return self.k8s_chaos.apply_cpu_pressure(pods.items[0].metadata.name, 4)
            return False
        
        elif experiment_type == ChaosExperiment.MEMORY_PRESSURE:
            pods = self.k8s_chaos.v1.list_namespaced_pod(
                namespace=self.config.namespace,
                label_selector="app=rec-engine-api"
            )
            if pods.items:
                return self.k8s_chaos.apply_memory_pressure(pods.items[0].metadata.name, 512)
            return False
        
        elif experiment_type == ChaosExperiment.REDIS_NODE_FAILURE:
            return self.redis_chaos.simulate_node_failure(1)
        
        elif experiment_type == ChaosExperiment.KAFKA_BROKER_FAILURE:
            return self.kafka_chaos.simulate_broker_failure(1)
        
        elif experiment_type == ChaosExperiment.DATABASE_CONNECTION_FAILURE:
            return self.db_chaos.simulate_connection_failure(300)
        
        else:
            logger.warning(f"Experiment type {experiment_type} not implemented")
            return False
    
    def _monitor_during_chaos(self, duration: int) -> Dict[str, float]:
        """Monitor metrics during chaos."""
        metrics_history = []
        
        for _ in range(duration // self.config.monitoring_interval):
            metrics = self.metrics_collector.collect_chaos_metrics()
            metrics_history.append(metrics)
            time.sleep(self.config.monitoring_interval)
        
        # Calculate average metrics during chaos
        if metrics_history:
            avg_metrics = {}
            for key in metrics_history[0].keys():
                avg_metrics[key] = sum(m[key] for m in metrics_history) / len(metrics_history)
            return avg_metrics
        
        return {}
    
    def _stop_chaos(self, experiment_type: ChaosExperiment):
        """Stop chaos experiment."""
        if experiment_type == ChaosExperiment.NETWORK_PARTITION:
            # Clean up network policies
            network_policies = self.k8s_chaos.networking_v1.list_namespaced_network_policy(
                namespace=self.config.namespace
            )
            
            for policy in network_policies.items:
                if "chaos-partition" in policy.metadata.name:
                    self.k8s_chaos.cleanup_network_policy(policy.metadata.name)
        
        # Other cleanup would go here depending on the experiment type
        logger.info(f"Stopped chaos experiment: {experiment_type.value}")
    
    def _assess_impact(self, baseline: Dict[str, float], chaos: Dict[str, float], 
                      recovery: Dict[str, float]) -> Dict[str, Any]:
        """Assess the impact of the chaos experiment."""
        impact = {
            'qps_drop': 0.0,
            'latency_increase': 0.0,
            'error_rate_increase': 0.0,
            'cache_hit_rate_drop': 0.0,
            'recovery_time': 0,
            'service_degraded': False,
            'service_failed': False
        }
        
        # Calculate QPS drop
        if baseline.get('qps', 0) > 0:
            impact['qps_drop'] = (baseline['qps'] - chaos.get('qps', 0)) / baseline['qps']
        
        # Calculate latency increase
        if baseline.get('p95_latency', 0) > 0:
            impact['latency_increase'] = chaos.get('p95_latency', 0) / baseline['p95_latency']
        
        # Calculate error rate increase
        baseline_error_rate = baseline.get('error_rate', 0)
        chaos_error_rate = chaos.get('error_rate', 0)
        impact['error_rate_increase'] = max(0, chaos_error_rate - baseline_error_rate)
        
        # Calculate cache hit rate drop
        if baseline.get('cache_hit_rate', 0) > 0:
            impact['cache_hit_rate_drop'] = (baseline['cache_hit_rate'] - chaos.get('cache_hit_rate', 0)) / baseline['cache_hit_rate']
        
        # Determine service status
        if impact['error_rate_increase'] > 0.5:  # 50% error rate increase
            impact['service_failed'] = True
        elif impact['qps_drop'] > 0.5 or impact['latency_increase'] > 3.0:
            impact['service_degraded'] = True
        
        return impact
    
    def run_all_experiments(self) -> List[ExperimentResult]:
        """Run all chaos experiments."""
        experiments = [
            ChaosExperiment.POD_DELETION,
            ChaosExperiment.NETWORK_PARTITION,
            ChaosExperiment.CPU_PRESSURE,
            ChaosExperiment.MEMORY_PRESSURE,
            ChaosExperiment.REDIS_NODE_FAILURE,
            ChaosExperiment.KAFKA_BROKER_FAILURE
        ]
        
        results = []
        
        for experiment in experiments:
            try:
                result = self.run_experiment(experiment)
                results.append(result)
                
                # Wait between experiments
                time.sleep(60)
                
            except Exception as e:
                logger.error(f"Failed to run experiment {experiment}: {e}")
        
        return results
    
    def generate_report(self) -> str:
        """Generate chaos testing report."""
        report = []
        report.append("# Chaos Testing Report\n")
        report.append(f"Generated: {datetime.now(timezone.utc).isoformat()}\n")
        
        for result in self.results:
            report.append(f"## {result.experiment_type.value}\n")
            report.append(f"- **Duration**: {result.duration}s")
            report.append(f"- **Success**: {result.success}")
            report.append(f"- **QPS Drop**: {result.impact_assessment.get('qps_drop', 0):.2%}")
            report.append(f"- **Latency Increase**: {result.impact_assessment.get('latency_increase', 0):.2f}x")
            report.append(f"- **Error Rate Increase**: {result.impact_assessment.get('error_rate_increase', 0):.2%}")
            report.append(f"- **Service Degraded**: {result.impact_assessment.get('service_degraded', False)}")
            report.append(f"- **Service Failed**: {result.impact_assessment.get('service_failed', False)}")
            report.append("")
        
        return "\n".join(report)


# Example usage
def main():
    """Example usage of chaos testing."""
    config = ChaosConfig(
        namespace="rec-engine-prod",
        experiment_duration=300,
        recovery_time=600
    )
    
    tester = ChaosTester(config)
    
    # Run a single experiment
    result = tester.run_experiment(ChaosExperiment.POD_DELETION)
    print(f"Experiment result: {result.success}")
    
    # Run all experiments
    results = tester.run_all_experiments()
    
    # Generate report
    report = tester.generate_report()
    print(report)


if __name__ == "__main__":
    main()
