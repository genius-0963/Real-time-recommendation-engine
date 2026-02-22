"""
Configuration management for the real-time recommendation engine.
Supports environment-specific configs and feature flags.
"""

import os
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from pathlib import Path
import yaml
from omegaconf import OmegaConf


@dataclass
class KafkaConfig:
    """Kafka configuration for streaming pipeline."""
    bootstrap_servers: List[str] = field(default_factory=lambda: ["localhost:9092"])
    topic_prefix: str = "rec-engine"
    consumer_group: str = "rec-engine-consumer"
    auto_offset_reset: str = "latest"
    enable_auto_commit: bool = False
    session_timeout_ms: int = 30000
    heartbeat_interval_ms: int = 3000
    max_poll_records: int = 1000
    compression_type: str = "snappy"
    
    # Topic configurations
    user_events_topic: str = "user-events"
    interaction_events_topic: str = "interaction-events"
    feature_updates_topic: str = "feature-updates"
    model_updates_topic: str = "model-updates"
    
    # Schema registry
    schema_registry_url: str = "http://localhost:8081"
    
    # Security
    security_protocol: str = "PLAINTEXT"
    sasl_mechanism: Optional[str] = None
    sasl_username: Optional[str] = None
    sasl_password: Optional[str] = None


@dataclass
class RedisConfig:
    """Redis configuration for caching and online feature store."""
    host: str = "localhost"
    port: int = 6379
    db: int = 0
    password: Optional[str] = None
    max_connections: int = 100
    socket_timeout: int = 5
    socket_connect_timeout: int = 5
    retry_on_timeout: bool = True
    health_check_interval: int = 30
    
    # Cluster configuration
    cluster_enabled: bool = False
    cluster_nodes: List[Dict[str, str]] = field(default_factory=list)
    
    # TTL configurations (seconds)
    user_features_ttl: int = 3600  # 1 hour
    item_features_ttl: int = 7200  # 2 hours
    recommendation_cache_ttl: int = 300  # 5 minutes
    model_cache_ttl: int = 1800  # 30 minutes


@dataclass
class DatabaseConfig:
    """PostgreSQL configuration for offline feature store."""
    host: str = "localhost"
    port: int = 5432
    database: str = "rec_engine"
    username: str = "postgres"
    password: str = "postgres"
    pool_size: int = 20
    max_overflow: int = 30
    pool_timeout: int = 30
    pool_recycle: int = 3600
    
    # Connection settings
    connect_timeout: int = 10
    command_timeout: int = 30
    
    # SSL settings
    ssl_mode: str = "prefer"
    ssl_cert: Optional[str] = None
    ssl_key: Optional[str] = None
    ssl_root_cert: Optional[str] = None


@dataclass
class ModelConfig:
    """Model configuration and hyperparameters."""
    embedding_dim: int = 128
    hidden_layers: List[int] = field(default_factory=lambda: [512, 256, 128])
    dropout_rate: float = 0.2
    learning_rate: float = 0.001
    batch_size: int = 2048
    num_epochs: int = 100
    early_stopping_patience: int = 10
    
    # Negative sampling
    num_negatives: int = 5
    negative_sampling_strategy: str = "uniform"  # uniform, popular, adaptive
    
    # Regularization
    l2_regularization: float = 1e-5
    embedding_l2_regularization: float = 1e-6
    
    # Model paths
    model_registry_path: str = "/models"
    checkpoint_dir: str = "/checkpoints"
    
    # ANN index configuration
    index_type: str = "scann"  # scann, faiss
    num_candidates: int = 100
    final_k: int = 10
    
    # ScaNN specific
    scann_num_leaves: int = 100
    scann_num_leaves_to_search: int = 10
    scann_quantization_bits: int = 8
    
    # FAISS specific
    faiss_index_type: str = "IVF_PQ"
    faiss_nlist: int = 1000
    faiss_m: int = 64
    faiss_nbits: int = 8


@dataclass
class TrainingConfig:
    """Distributed training configuration."""
    backend: str = "nccl"
    world_size: int = 1
    rank: int = 0
    local_rank: int = 0
    master_addr: str = "localhost"
    master_port: str = "12355"
    
    # Mixed precision training
    use_amp: bool = True
    amp_opt_level: str = "O1"
    
    # Gradient accumulation
    gradient_accumulation_steps: int = 1
    
    # Checkpointing
    save_every_n_steps: int = 1000
    save_top_k: int = 3
    monitor_metric: str = "val_auc"
    
    # Data loading
    num_workers: int = 4
    pin_memory: bool = True
    persistent_workers: bool = True
    
    # Profiling
    profile_memory: bool = False
    profile_steps: int = 100


@dataclass
class APIConfig:
    """FastAPI server configuration."""
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 4
    reload: bool = False
    log_level: str = "info"
    
    # Rate limiting
    rate_limit_requests: int = 1000
    rate_limit_window: int = 60  # seconds
    
    # Request timeouts
    request_timeout: int = 30
    read_timeout: int = 60
    write_timeout: int = 60
    
    # Circuit breaker
    circuit_breaker_threshold: int = 5
    circuit_breaker_timeout: int = 60
    
    # Authentication
    jwt_secret_key: str = "your-secret-key"
    jwt_algorithm: str = "HS256"
    jwt_expiration: int = 3600  # seconds
    
    # CORS
    cors_origins: List[str] = field(default_factory=lambda: ["*"])
    cors_methods: List[str] = field(default_factory=lambda: ["GET", "POST"])
    cors_headers: List[str] = field(default_factory=lambda: ["*"])


@dataclass
class MonitoringConfig:
    """Monitoring and observability configuration."""
    prometheus_port: int = 9090
    grafana_port: int = 3000
    
    # Metrics collection
    metrics_interval: int = 15  # seconds
    histogram_buckets: List[float] = field(default_factory=lambda: [
        0.001, 0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0, 7.5, 10.0
    ])
    
    # Logging
    log_level: str = "INFO"
    log_format: str = "json"
    log_file: Optional[str] = None
    
    # Tracing
    jaeger_endpoint: Optional[str] = None
    tracing_sample_rate: float = 0.1
    
    # Alerting
    alert_webhook_url: Optional[str] = None
    alert_thresholds: Dict[str, float] = field(default_factory=lambda: {
        "latency_p95_ms": 100.0,
        "error_rate": 0.01,
        "qps": 10000.0,
        "memory_usage": 0.8,
        "cpu_usage": 0.8
    })
    
    # Drift detection
    drift_detection_interval: int = 300  # seconds
    psi_threshold: float = 0.1
    kl_threshold: float = 0.1
    ctr_window_size: int = 10000  # number of interactions


@dataclass
class FeatureStoreConfig:
    """Feature store configuration."""
    online_store: str = "redis"
    offline_store: str = "postgres"
    
    # Feature definitions
    feature_definitions_path: str = "features/"
    
    # Sync settings
    sync_interval: int = 300  # seconds
    batch_size: int = 1000
    
    # Point-in-time correctness
    enable_point_in_time: bool = True
    max_lookback_days: int = 30
    
    # Feature versioning
    enable_feature_versioning: bool = True
    max_feature_versions: int = 10


@dataclass
class ExperimentConfig:
    """A/B testing and experimentation configuration."""
    experiment_registry_path: str = "/experiments"
    
    # Traffic splitting
    default_traffic_split: Dict[str, float] = field(default_factory=lambda: {
        "control": 0.5,
        "treatment": 0.5
    })
    
    # Statistical significance
    min_sample_size: int = 1000
    confidence_level: float = 0.95
    statistical_power: float = 0.8
    
    # Metrics
    primary_metric: str = "ctr"
    secondary_metrics: List[str] = field(default_factory=lambda: [
        "conversion_rate", "engagement_time", "return_rate"
    ])


@dataclass
class Config:
    """Main configuration class."""
    environment: str = os.getenv("ENVIRONMENT", "development")
    debug: bool = os.getenv("DEBUG", "false").lower() == "true"
    
    kafka: KafkaConfig = field(default_factory=KafkaConfig)
    redis: RedisConfig = field(default_factory=RedisConfig)
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    api: APIConfig = field(default_factory=APIConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)
    feature_store: FeatureStoreConfig = field(default_factory=FeatureStoreConfig)
    experiments: ExperimentConfig = field(default_factory=ExperimentConfig)
    
    @classmethod
    def from_file(cls, config_path: str) -> "Config":
        """Load configuration from YAML file."""
        if not Path(config_path).exists():
            return cls()
        
        with open(config_path, "r") as f:
            config_dict = yaml.safe_load(f)
        
        return cls(**config_dict)
    
    @classmethod
    def from_env(cls) -> "Config":
        """Load configuration from environment variables."""
        config = cls()
        
        # Override with environment variables
        if os.getenv("KAFKA_BOOTSTRAP_SERVERS"):
            config.kafka.bootstrap_servers = os.getenv("KAFKA_BOOTSTRAP_SERVERS").split(",")
        
        if os.getenv("REDIS_HOST"):
            config.redis.host = os.getenv("REDIS_HOST")
        
        if os.getenv("REDIS_PORT"):
            config.redis.port = int(os.getenv("REDIS_PORT"))
        
        if os.getenv("DATABASE_URL"):
            # Parse database URL
            import re
            match = re.match(r"postgresql://([^:]+):([^@]+)@([^:]+):(\d+)/(.+)", os.getenv("DATABASE_URL"))
            if match:
                config.database.username, config.database.password, config.database.host, config.database.port, config.database.database = match.groups()
                config.database.port = int(config.database.port)
        
        return config
    
    def to_dict(self) -> Dict:
        """Convert configuration to dictionary."""
        return OmegaConf.to_container(OmegaConf.structured(self), resolve=True)
    
    def save(self, path: str) -> None:
        """Save configuration to YAML file."""
        with open(path, "w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)


# Global configuration instance
config = Config.from_env()
