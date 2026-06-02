<div align="center">

# ⚡ Real-Time Recommendation Engine

### Production-Grade Personalization at Netflix/Meta Scale

[![Build Status](https://img.shields.io/badge/build-passing-brightgreen?style=for-the-badge&logo=github-actions&logoColor=white)](https://github.com/company/rec-engine/actions)
[![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104.1-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1.0-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)
[![Kubernetes](https://img.shields.io/badge/Kubernetes-1.24+-326CE5?style=for-the-badge&logo=kubernetes&logoColor=white)](https://kubernetes.io)
[![Kafka](https://img.shields.io/badge/Apache_Kafka-231F20?style=for-the-badge&logo=apache-kafka&logoColor=white)](https://kafka.apache.org)
[![Redis](https://img.shields.io/badge/Redis-5.0+-DC382D?style=for-the-badge&logo=redis&logoColor=white)](https://redis.io)
[![License](https://img.shields.io/badge/License-MIT-F7DF1E?style=for-the-badge)](LICENSE)

---

**Sub-100ms latency** · **100K+ QPS** · **99.99% uptime** · **Two-Tower neural retrieval** · **Multi-model A/B testing**

A horizontally scalable, real-time recommendation engine delivering personalized suggestions with sub-100ms P95 latency while serving millions of requests per second. Built on a **Two-Tower neural network** with multi-head attention, **ScaNN/FAISS** approximate nearest neighbor search, **Kafka** event streaming with exactly-once semantics, **Istio** service mesh traffic routing, and **Argo Rollouts** canary deployments on Kubernetes.

[Getting Started](#-quick-start) · [Architecture](#-architecture) · [API Docs](#-api-reference) · [Deployment](#-deployment) · [Contributing](#-contributing)

</div>

---

## 📑 Table of Contents

- [Architecture](#-architecture)
- [Project Structure](#-project-structure)
- [Core Services](#-core-services)
- [Technology Stack](#-technology-stack)
- [Performance Benchmarks](#-performance-benchmarks)
- [Quick Start](#-quick-start)
- [API Reference](#-api-reference)
- [Configuration](#-configuration)
- [Monitoring & Observability](#-monitoring--observability)
- [Testing](#-testing)
- [Deployment](#-deployment)
- [Roadmap](#-roadmap)
- [Contributing](#-contributing)
- [License](#-license)

---

## 🏗 Architecture

### High-Level System Design

```mermaid
graph TB
    subgraph Clients["🖥️ Client Applications"]
        Mobile["📱 Mobile App"]
        Web["🌐 Web App"]
        TV["📺 TV App"]
        IoT["📡 IoT Devices"]
    end

    subgraph Gateway["🚪 API Gateway Layer"]
        LB["Load Balancer<br/><i>NGINX / HAProxy</i>"]
        RL["Rate Limiter<br/><i>Redis Token Bucket</i>"]
    end

    subgraph API["⚙️ Recommendation Service (K8s + Istio)"]
        Pod1["Pod 1"]
        Pod2["Pod 2"]
        Pod3["Pod 3"]
        PodN["Pod N"]
    end

    subgraph DataLayer["💾 Data & Model Layer"]
        FS["Feature Store<br/><i>Redis (online) + PostgreSQL (offline)</i>"]
        MS["Model Store<br/><i>Two-Tower PyTorch + ScaNN/FAISS</i>"]
        Cache["Cache Layer<br/><i>Redis TTL</i>"]
    end

    subgraph Streaming["📡 Kafka Streaming Pipeline"]
        K1["user-events"]
        K2["item-events"]
        K3["interaction-events"]
        K4["feature-updates"]
    end

    subgraph ML["🧠 Distributed Training"]
        DDP["PyTorch DDP<br/><i>Multi-GPU Workers</i>"]
        MLflow["Experiment Tracking<br/><i>MLflow + W&B</i>"]
        Tune["Hyperparameter Tuning<br/><i>Ray Tune + ASHA</i>"]
    end

    subgraph Monitoring["📊 Observability Stack"]
        Prom["Prometheus<br/><i>30+ Alert Rules</i>"]
        Graf["Grafana<br/><i>Dashboards</i>"]
        OTel["OpenTelemetry<br/><i>Traces</i>"]
        Drift["Drift Detection<br/><i>PSI + KL + KS</i>"]
    end

    Clients --> Gateway
    LB --> RL
    RL --> API
    API --> FS
    API --> MS
    API --> Cache
    API --> Streaming
    Streaming --> ML
    ML --> MS
    API -.-> Monitoring
    Streaming -.-> Monitoring
```

### Request Lifecycle

```mermaid
sequenceDiagram
    participant C as Client
    participant GW as API Gateway
    participant RL as Rate Limiter
    participant API as FastAPI Service
    participant FS as Feature Store
    participant Cache as Redis Cache
    participant Model as Two-Tower Model
    participant VX as ANN Index (ScaNN/FAISS)
    participant K as Kafka

    C->>GW: POST /recommend
    GW->>RL: Check rate limit (token bucket)
    RL-->>GW: ✅ Allowed

    GW->>API: Forward request
    API->>Cache: Check cache (user_id + context)
    
    alt Cache Hit
        Cache-->>API: Return cached results
    else Cache Miss
        API->>FS: Fetch user & item features (Redis online store)
        FS-->>API: Feature vectors
        API->>Model: Generate embeddings (User Tower + Item Tower)
        Model->>VX: ANN search (Top-K candidates)
        VX-->>Model: Candidate items with scores
        Model-->>API: Ranked recommendations
        API->>Cache: Store results (TTL)
    end

    API->>K: Publish interaction event (Avro serialization)
    API-->>C: JSON response (< 100ms)
```

### Data Flow Architecture

```mermaid
graph LR
    subgraph Ingestion["📥 Event Ingestion"]
        Click["Click Events"]
        View["View Events"]
        Purchase["Purchase Events"]
    end

    subgraph StreamProc["🔄 Stream Processing"]
        KP["Kafka Producer<br/><i>Avro + DLQ</i>"]
        KC["Kafka Consumer<br/><i>Exactly-Once</i>"]
        SP["Stream Processor<br/><i>Real-time Aggregation</i>"]
    end

    subgraph Features["📊 Feature Engineering"]
        SW["Sliding Windows<br/><i>5min / 1h / 1day</i>"]
        Agg["Aggregations<br/><i>count/sum/mean/std</i>"]
        Pop["Popularity Tracking<br/><i>Item + Category</i>"]
    end

    subgraph Stores["💾 Feature Stores"]
        Redis["Redis Online Store<br/><i>Sub-ms Lookups</i>"]
        PG["PostgreSQL Offline Store<br/><i>Point-in-Time Queries</i>"]
        Sync["Bidirectional Sync<br/><i>3 Conflict Strategies</i>"]
    end

    Ingestion --> KP --> KC --> SP
    SP --> Features
    Features --> Redis
    PG --> Sync --> Redis
```

---

## 📂 Project Structure

```
realtime-rec-engine-v2/
│
├── 📦 app/                              # Core API application
│   ├── config.py                        # Centralized config (9 dataclasses, env-based)
│   └── api/
│       └── main.py                      # FastAPI app: 15 endpoints, CORS, GZip, OTel, lifespan
│
├── 🧠 training/                         # Distributed ML training
│   └── distributed/
│       ├── model.py                     # Two-Tower neural network (multi-head attention, contrastive loss)
│       ├── dataset.py                   # Dataset utilities
│       ├── launcher.py                  # Training job launcher
│       └── train_ddp.py                 # PyTorch DDP: multi-GPU, mixed precision, MLflow + W&B
│
├── 🔍 index/                            # ANN vector search engine
│   ├── build_index.py                   # ScaNN/FAISS index builder + native FAISS merge
│   ├── benchmark.py                     # Recall@K, latency percentiles, QPS benchmarking
│   └── incremental_update.py            # Real-time add/remove, auto-rebuild on >20% drift
│
├── 📡 streaming/                        # Kafka event streaming pipeline
│   ├── kafka_producer.py                # Avro serialization, batching, dead letter queue
│   ├── kafka_consumer.py                # Consumer groups, exactly-once, EventProcessor logic
│   ├── schema_registry.py              # Schema management
│   └── stream_processor.py             # Sliding windows, real-time aggregations, session tracking
│
├── 🗄️ feature_store/                    # Dual-store feature management
│   ├── online_store.py                  # Redis: TTL, JSON serialization, connection pooling
│   ├── offline_store.py                 # PostgreSQL: versioning, point-in-time joins, time-travel
│   └── sync_pipeline.py                # Bidirectional sync with real DB queries + 3 conflict strategies
│
├── 📊 monitoring/                       # Observability stack configuration
│   ├── prometheus.yaml                  # Scrape configs: API, Redis, Kafka, Node exporters
│   ├── alert_rules.yaml                 # 30+ alert rules across 8 groups
│   ├── otel_config.yaml                 # OpenTelemetry Collector: OTLP, Jaeger, tail sampling
│   ├── drift_detection.py              # Statistical drift: PSI, KL Divergence, KS Test + auto-retrain
│   ├── grafana_dashboard.json           # 12-panel production Grafana dashboard
│   ├── grafana-datasource.yml           # Auto-provisioned Prometheus datasource
│   └── grafana-dashboard-provider.yml   # Auto-provisioned dashboard loader
│
├── 🏗️ infrastructure/                   # Production deployment configs
│   ├── kubernetes/
│   │   ├── api-deployment.yaml          # Deployment + HPA (3–50 pods) + PDB + NetworkPolicy
│   │   ├── kafka-cluster.yaml           # Strimzi: 3 brokers, 3 ZK, SASL/SCRAM, TLS
│   │   ├── redis-cluster.yaml           # Redis Operator: 6 nodes, 12GB, LRU eviction
│   │   ├── training-job.yaml            # DDP Job: 4 GPU workers (V100/A100)
│   │   └── ingress.yaml                 # NGINX Ingress with TLS + rate limiting
│   └── helm/rec-engine/                 # Helm chart (8 templates)
│       ├── Chart.yaml                   # Chart metadata (v1.0.0)
│       ├── values.yaml                  # Default values (replicas, resources, HPA, env)
│       └── templates/                   # deployment, service, hpa, ingress, configmap
│
├── 🧪 tests/                            # Comprehensive test suite
│   ├── conftest.py                      # 8 shared fixtures (mocked Redis, PostgreSQL, FastAPI)
│   ├── pytest.ini                       # Markers: unit, integration, slow
│   ├── unit/
│   │   ├── test_api_routes.py           # 12 API endpoint tests
│   │   ├── test_model.py                # 8 Two-Tower model tests
│   │   ├── test_feature_store.py        # 9 feature store tests
│   │   ├── test_index.py                # 6 ANN index tests
│   │   ├── test_streaming.py            # 5 streaming pipeline tests
│   │   └── test_config.py               # 6 config tests
│   └── integration/
│       ├── test_api_integration.py       # API flow integration tests
│       ├── test_feature_pipeline.py      # Feature sync integration tests
│       └── test_streaming_pipeline.py    # Kafka pipeline integration tests
│
├── 🧪 load_testing/                     # Performance & chaos testing
│   ├── locustfile.py                    # 4 user types: recommend, batch, events, feature store
│   ├── k6_script.js                     # Staged ramp (10→50→100 VUs), custom thresholds
│   └── chaos_testing.py                 # 10 experiment types, Prometheus-based impact assessment
│
├── 🔄 ci-cd/                            # Continuous integration & deployment
│   ├── github-actions.yml               # 8-job pipeline: lint → test → build → scan → deploy
│   ├── canary_deploy.yml                # Argo Rollouts + Istio canary deployments
│   └── build_and_push.sh              # Docker multi-arch build & ECR push script
│
├── 🗃️ alembic/                          # Database migrations
│   ├── env.py                           # Migration environment (reads DATABASE_URL)
│   ├── script.py.mako                   # Migration template
│   └── versions/
│       └── 001_initial_schema.py        # Initial: features, interactions, model_versions tables
│
├── 🐳 Docker Compose
│   ├── docker-compose.yml               # Full stack: Redis, PostgreSQL, Kafka, Prometheus, Grafana, Jaeger
│   ├── docker-compose.test.yml          # Minimal stack for integration tests
│   └── docker-compose.monitoring.yml    # Full observability: Prometheus, Grafana, Jaeger, OTel, exporters
│
├── ⚙️ Project Configuration
│   ├── .env.example                     # All environment variables with descriptions
│   ├── .gitignore                       # Python, ML, IDE, Docker, OS patterns
│   ├── .pre-commit-config.yaml          # 6 hook repos: black, isort, flake8, mypy, bandit
│   ├── config.yaml                      # Application config (all sections)
│   ├── pyproject.toml                   # Project metadata + tool configs
│   ├── alembic.ini                      # Alembic database migration config
│   ├── pytest.ini                       # Test runner configuration
│   ├── Makefile                         # 11 development workflow targets
│   ├── requirements.txt                 # 94 Python dependencies
│   ├── LICENSE                          # MIT License
│   ├── CONTRIBUTING.md                  # Contributing guide
│   └── README.md                        # This file
└──
```

---

## ⚙️ Core Services

### 1. Recommendation API

A high-performance FastAPI application with **15 endpoints**, circuit breaker patterns, response caching, and full OpenTelemetry instrumentation.

| Capability | Implementation |
|---|---|
| Framework | FastAPI 0.104.1 + Uvicorn (ASGI) |
| Validation | Pydantic v2 request/response schemas |
| Rate Limiting | Redis-backed token bucket middleware |
| Authentication | JWT (python-jose) + bcrypt password hashing |
| Serialization | `orjson` for high-performance JSON |
| Compression | GZip middleware |
| Tracing | OpenTelemetry → Jaeger |
| Resilience | `tenacity` circuit breaker pattern |
| Lifecycle | Async lifespan events (startup/shutdown) |

### 2. Two-Tower Neural Network

The core ML model is a **Two-Tower architecture** — separate `UserTower` and `ItemTower` encoder networks that learn embeddings in a shared vector space, enabling efficient approximate nearest neighbor retrieval at serving time.

```
                 ┌──────────────────┐          ┌──────────────────┐
                 │    User Tower    │          │    Item Tower    │
                 │                  │          │                  │
User Features →  │  [512] → [256]  │          │  [512] → [256]  │  ← Item Features
                 │  → [128-d emb]  │          │  → [128-d emb]  │
                 │                  │          │                  │
                 │  + Multi-Head   │          │  + Multi-Head   │
                 │    Attention    │          │    Attention    │
                 │    (8 heads)    │          │    (8 heads)    │
                 └────────┬───────┘          └────────┬───────┘
                          │                           │
                          └─────────┬─────────────────┘
                                    │
                              Dot Product
                                    │
                          ┌─────────▼─────────┐
                          │ Contrastive Loss   │
                          │ (τ = 0.1)          │
                          │ + Hard Negatives   │
                          │   (5 per sample)   │
                          └───────────────────┘
```

| Feature | Detail |
|---|---|
| Architecture | Two-Tower (UserTower + ItemTower) with shared embedding space |
| Embedding Dim | 128 dimensions |
| Hidden Layers | [512, 256, 128] (configurable) |
| Attention | Multi-head attention with 8 heads |
| Loss Function | Contrastive loss with temperature scaling (τ = 0.1) |
| Negative Mining | Hard negative mining (5 negatives per positive) |
| Methods | `forward()`, `predict()`, `generate_embeddings()` |

### 3. ANN Vector Search Index

Dual-engine vector search with **ScaNN** and **FAISS**, supporting hot-swapping, incremental updates, and comprehensive benchmarking.

| Backend | Use Case | GPU Support |
|---|---|---|
| **ScaNN** | Production default — Google's Scalable Nearest Neighbors | ❌ |
| **FAISS** | Alternative — Meta's vector similarity library | ✅ (`faiss-gpu`) |
| **Brute-force** | Development/fallback — exact KNN | ❌ |

| Component | Capabilities |
|---|---|
| `IndexManager` | Build, save, load, query with configurable parameters |
| `IncrementalIndexUpdater` | Real-time add/remove, auto-rebuild when drift > 20% |
| `AnnBenchmark` | Recall@K, latency P50/P95/P99, QPS measurement, visualization |

### 4. Feature Store

A **dual-store architecture** with bidirectional synchronization and three conflict resolution strategies. Integrated with **Feast** for formal feature management.

| Layer | Store | Capabilities |
|---|---|---|
| **Online** | Redis Cluster | Sub-ms lookups, TTL management, connection pooling, batch get/set |
| **Offline** | PostgreSQL | Point-in-time joins, time-travel queries, feature versioning |
| **Sync** | Bidirectional Pipeline | 3 strategies: `timestamp_wins`, `online_wins`, `offline_wins` |
| **Definitions** | Feast | FeatureViews, Entities, FeatureServices |

### 5. Streaming Pipeline

An **Apache Kafka** event streaming system with exactly-once semantics, Avro schema serialization, dead letter queues, and real-time feature engineering.

| Component | Capabilities |
|---|---|
| **Producer** | Avro serialization, delivery callbacks, batching, DLQ, error handling |
| **Consumer** | Consumer groups, exactly-once semantics, batch processing, DLQ |
| **Stream Processor** | Sliding time windows (5min/1h/1day), real-time aggregations (count/sum/mean/std/min/max/median), session tracking, popularity tracking, event weighting (view=1 → purchase=10) |

**Kafka Topics (Strimzi-managed):**

| Topic | Purpose | Config |
|---|---|---|
| `user-events` | Real-time user interactions | SASL/SCRAM + TLS |
| `item-events` | Item catalog changes | SASL/SCRAM + TLS |
| `interaction-events` | Click, view, purchase events | SASL/SCRAM + TLS |
| `feature-updates` | Feature value changes | SASL/SCRAM + TLS |

### 6. Distributed Training

**PyTorch DistributedDataParallel (DDP)** training with NCCL backend, mixed precision, and dual experiment tracking (MLflow + Weights & Biases).

| Capability | Implementation |
|---|---|
| Parallelism | PyTorch DDP with NCCL backend |
| Precision | Mixed precision training (AMP) |
| Optimization | Gradient clipping, LR warmup + cosine annealing |
| Reliability | Checkpoint save/resume, early stopping |
| Tracking | MLflow + Weights & Biases (dual logging) |
| Tuning | Ray Tune + ASHA scheduler (Bayesian, grid, random search) |
| Data Pipeline | PostgreSQL → Pandas → PyTorch DataLoader |
| Hardware | V100/A100 GPU node affinity, 8GB shared memory (NCCL) |

---

## 🛠 Technology Stack

<table>
<tr>
<td>

### Core & API
| Technology | Version |
|---|---|
| Python | 3.9+ |
| FastAPI | 0.104.1 |
| Uvicorn | 0.24.0 |
| Pydantic | 2.4.2+ |
| orjson | 3.9.10 |
| httpx | 0.25.1 |

</td>
<td>

### ML & AI
| Technology | Version |
|---|---|
| PyTorch | 2.1.0 |
| TensorFlow | 2.13.0 |
| ScaNN | 1.9.0 |
| FAISS | 1.7.4 |
| Ray / Ray Tune | 2.8.0 |
| MLflow | 2.7.1 |
| Scikit-learn | 1.3.0 |
| Feast | 0.38.0 |
| Numba | 0.58.1 |

</td>
</tr>
<tr>
<td>

### Data & Streaming
| Technology | Version |
|---|---|
| Apache Kafka | confluent 2.2.0 |
| Kafka (Strimzi) | K8s Operator |
| Redis | 5.0.1 |
| PostgreSQL | psycopg2 2.9.7 |
| SQLAlchemy | 2.0.23 |
| Avro | 1.11.3 |
| PyArrow | 13.0.0 |

</td>
<td>

### Observability
| Technology | Purpose |
|---|---|
| Prometheus | Metrics (30+ alert rules) |
| Grafana | Dashboarding |
| OpenTelemetry | Distributed tracing (tail sampling) |
| Jaeger | Trace visualization |
| Sentry | Error tracking |
| Whylogs | Data drift detection |
| structlog | Structured JSON logging |
| Fluent Bit | Log shipping |

</td>
</tr>
<tr>
<td>

### Infrastructure
| Technology | Purpose |
|---|---|
| Docker | Multi-stage, multi-arch builds |
| Kubernetes | Orchestration (HPA, PDB) |
| Istio | Service mesh + traffic routing |
| Argo Rollouts | Canary deployments |
| Strimzi | Kafka on K8s |
| Redis Operator | Redis Cluster on K8s |
| GitHub Actions | 8-job CI/CD pipeline |
| Trivy / Grype | Container security scanning |
| Bandit / Semgrep | Code security analysis |
| SonarCloud | Code quality |

</td>
<td>

### Testing & Performance
| Technology | Purpose |
|---|---|
| pytest | Unit & integration tests |
| Locust | Python-based load testing |
| k6 | High-performance load testing |
| Chaos testing | 10 resilience experiments |
| Hydra + OmegaConf | Config management |
| black + flake8 + mypy | Code quality |

</td>
</tr>
<tr>
<td>

### Cloud & Storage
| Technology | Purpose |
|---|---|
| AWS (boto3) | S3, ECR, EBS, EFS |
| Azure (Blob) | Storage support |
| GCP (GCS) | Storage support |
| Alembic | Database migrations |
| Celery | Task queue |
| Dask | Distributed computing |

</td>
<td>

### Security & Auth
| Technology | Purpose |
|---|---|
| python-jose | JWT tokens |
| passlib + bcrypt | Password hashing |
| lz4 + xxhash | Compression & hashing |
| SASL/SCRAM | Kafka authentication |
| TLS | Encryption in transit |
| NetworkPolicy | K8s network isolation |

</td>
</tr>
</table>

---

## 📈 Performance Benchmarks

### Latency Targets

```
┌────────────────────┬──────────┬──────────┐
│     Component      │  P95     │  P99     │
├────────────────────┼──────────┼──────────┤
│ API Response       │ < 100ms  │ < 200ms  │
│ Feature Lookup     │ < 50ms   │ < 80ms   │
│ Model Inference    │ < 30ms   │ < 50ms   │
│ ANN Vector Search  │ < 20ms   │ < 35ms   │
└────────────────────┴──────────┴──────────┘
```

### Throughput & Reliability

| Metric | Target |
|---|---|
| API QPS | **100,000+** requests/second |
| Kafka Throughput | **1M+** events/second |
| Redis Operations | **500K+** ops/second |
| Database Queries | **50K+** queries/second |
| Uptime SLA | **99.99%** (99.9% SLO target in alert rules) |
| Recovery Time (MTTR) | **< 5 minutes** |
| Data Consistency | Strong consistency for features |
| Disaster Recovery | Multi-region deployment |

---

## 🚀 Quick Start

### Prerequisites

| Requirement | Minimum |
|---|---|
| Python | 3.9+ (tested on 3.9, 3.10, 3.11) |
| Docker & Docker Compose | Latest |
| RAM | 16 GB+ |
| CPU | 8+ cores |
| GPU | NVIDIA V100/A100 *(training only)* |
| Kubernetes | 1.24+ *(production only)* |

### Option A: Quick Start with Make

```bash
# 1. Clone the repository
git clone https://github.com/genius-0963/Real-time-recommendation-engine.git
cd Real-time-recommendation-engine/realtime-rec-engine-v2

# 2. Copy and configure environment variables
cp .env.example .env
# Edit .env with your settings (database, redis, kafka, etc.)

# 3. Install dependencies + start services + run server
make install
make dev          # starts docker-compose + uvicorn with hot-reload
```

### Option B: Manual Setup

```bash
# 1. Clone the repository
git clone https://github.com/genius-0963/Real-time-recommendation-engine.git
cd Real-time-recommendation-engine/realtime-rec-engine-v2

# 2. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate          # macOS/Linux
# venv\Scripts\activate           # Windows

# 3. Install dependencies (94 packages)
pip install -r requirements.txt

# 4. Copy environment config
cp .env.example .env

# 5. Start backing services (Redis, PostgreSQL, Kafka, etc.)
docker-compose up -d

# 6. Run database migrations
alembic upgrade head

# 7. Start the API server (with hot-reload)
uvicorn app.api.main:app --host 0.0.0.0 --port 8000 --reload
```

### Makefile Targets

| Target | Description |
|---|---|
| `make install` | Create venv and install dependencies |
| `make dev` | Start docker-compose + uvicorn with hot-reload |
| `make test` | Run unit tests with coverage |
| `make test-integration` | Run integration tests (starts test services) |
| `make lint` | Run black, flake8, mypy checks |
| `make format` | Auto-format code with black + isort |
| `make build` | Build Docker image |
| `make clean` | Remove caches and build artifacts |
| `make load-test` | Run Locust load test |
| `make chaos-test` | Run chaos engineering experiments |

### Verify Installation

```bash
# Health check
curl http://localhost:8000/health

# Readiness check
curl http://localhost:8000/ready

# Sample recommendation request
curl -X POST http://localhost:8000/recommend \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "user_12345",
    "num_items": 10,
    "context": {"device": "mobile", "location": "US"}
  }'

# Batch recommendation
curl -X POST http://localhost:8000/recommend/batch \
  -H "Content-Type: application/json" \
  -d '{
    "user_ids": ["user_001", "user_002", "user_003"],
    "num_items": 5
  }'
```

---

## 📖 API Reference

### Endpoints Overview

The API exposes **15 endpoints** organized into five functional groups:

#### Core Recommendation

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/recommend` | Get personalized recommendations for a single user |
| `POST` | `/recommend/batch` | Batch recommendations for multiple users |

#### Event Ingestion

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/events` | Ingest a single user event |
| `POST` | `/events/batch` | Batch event ingestion |
| `POST` | `/feedback` | Submit recommendation feedback |

#### Feature Store

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/features/{entity_type}/{entity_id}` | Retrieve features for a user or item |
| `POST` | `/features/{entity_type}/{entity_id}` | Update features for a user or item |

#### Model Management

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/model/info` | Get active model version, architecture, and metadata |
| `POST` | `/model/reload` | Hot-reload model from disk without downtime |

#### Index Management

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/index/status` | ANN index status (size, type, last rebuild) |
| `POST` | `/index/rebuild` | Trigger full index rebuild |

#### System

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Health check (service dependencies) |
| `GET` | `/ready` | Readiness probe for Kubernetes |
| `GET` | `/metrics` | Prometheus metrics endpoint |

---

### `POST /recommend`

Get personalized recommendations for a user.

<details>
<summary><b>Request</b></summary>

```http
POST /recommend
Content-Type: application/json
Authorization: Bearer <jwt_token>
```

```json
{
  "user_id": "user_12345",
  "item_id": "item_67890",
  "num_items": 10,
  "context": {
    "device": "mobile",
    "location": "US",
    "time_of_day": "evening"
  },
  "candidate_pool": ["popular", "trending", "personalized"],
  "filters": {
    "categories": ["electronics", "books"],
    "price_range": [10, 100]
  }
}
```

</details>

<details>
<summary><b>Response</b></summary>

```json
{
  "request_id": "req_abc123",
  "user_id": "user_12345",
  "recommendations": [
    {
      "item_id": "item_111",
      "score": 0.95,
      "explanation": "Based on your recent purchases",
      "category": "electronics",
      "price": 49.99
    }
  ],
  "metadata": {
    "model_version": "v2.1.0",
    "latency_ms": 45,
    "cache_hit": true,
    "ab_test_group": "treatment"
  }
}
```

</details>

---

### `POST /events`

Ingest user interaction events for real-time feature updates and model improvement.

<details>
<summary><b>Request</b></summary>

```http
POST /events
Content-Type: application/json
```

```json
{
  "user_id": "user_12345",
  "item_id": "item_111",
  "interaction_type": "click",
  "timestamp": "2024-01-15T10:30:00Z",
  "context": {
    "position": 1,
    "page": "homepage",
    "session_id": "sess_456"
  }
}
```

**Event weights for aggregation:** `view=1`, `click=2`, `add_to_cart=5`, `purchase=10`

</details>

---

## 🔧 Configuration

### Centralized Config (`app/config.py`)

All configuration is managed through Python dataclasses with an `from_env()` factory method:

| Config Class | Key Settings |
|---|---|
| `ModelConfig` | `embedding_dim=128`, `hidden_layers=[512,256,128]`, `num_heads=8`, `num_negatives=5`, `temperature=0.1` |
| `RedisConfig` | `pool_size=20`, `default_ttl=3600` |
| `KafkaConfig` | `bootstrap_servers`, `group_id`, 4 topic names |
| `DatabaseConfig` | `pool_size=10`, `max_overflow=20` |
| `FeatureStoreConfig` | `sync_interval=300`, `batch_size=1000` |
| `IndexConfig` | `type=scann`, `dim=128`, `num_neighbors=10`, `num_leaves=2000` |
| `TrainingConfig` | `batch_size=2048`, `epochs=100`, `lr=0.001`, `mixed_precision=true` |
| `MonitoringConfig` | Prometheus, Jaeger, OTel endpoints |

### Environment Variables

```bash
# ── Database ──────────────────────────────────────
DATABASE_URL=postgresql://user:pass@localhost:5432/rec_engine
REDIS_URL=redis://localhost:6379

# ── Kafka ─────────────────────────────────────────
KAFKA_BOOTSTRAP_SERVERS=localhost:9092
KAFKA_SCHEMA_REGISTRY_URL=http://localhost:8081

# ── Model ─────────────────────────────────────────
MODEL_PATH=/models/current
EMBEDDING_DIM=128
VECTOR_INDEX_TYPE=scann       # scann | faiss | brute_force

# ── API ───────────────────────────────────────────
API_HOST=0.0.0.0
API_PORT=8000
JWT_SECRET_KEY=your-secret-key

# ── Monitoring ────────────────────────────────────
PROMETHEUS_PORT=9090
GRAFANA_PORT=3000
JAEGER_ENDPOINT=http://localhost:14268/api/traces
OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4317
```

---

## 📊 Monitoring & Observability

### Observability Architecture

```mermaid
graph LR
    subgraph Collection["📥 Data Collection"]
        API["FastAPI Service"]
        Redis["Redis Exporter"]
        Kafka["Kafka Exporter"]
        Node["Node Exporter"]
    end

    subgraph Processing["⚙️ Processing"]
        OTel["OpenTelemetry Collector<br/><i>Receivers: OTLP + Prometheus + Jaeger</i><br/><i>Processors: batch, memory_limiter, tail_sampling</i>"]
    end

    subgraph Backends["💾 Storage & Visualization"]
        Prom["Prometheus<br/><i>Metrics</i>"]
        Jaeg["Jaeger<br/><i>Traces</i>"]
        Graf["Grafana<br/><i>Dashboards</i>"]
        Sen["Sentry<br/><i>Errors</i>"]
    end

    subgraph Intelligence["🧠 Intelligence"]
        Drift["Drift Detection<br/><i>PSI + KL Div + KS Test</i>"]
        AutoRT["Auto-Retrain<br/><i>Threshold + Cooldown</i>"]
        Alert["30+ Alert Rules<br/><i>8 Alert Groups</i>"]
    end

    Collection --> OTel --> Backends
    Prom --> Graf
    Prom --> Alert
    Drift --> AutoRT
    API -.-> Sen
```

### OpenTelemetry Pipeline

| Stage | Configuration |
|---|---|
| **Receivers** | OTLP (gRPC :4317, HTTP :4318), Prometheus (:8889), Jaeger |
| **Processors** | Batch, memory limiter (512MB), attributes, tail sampling (10% default, 100% errors) |
| **Exporters** | OTLP, Jaeger, Prometheus, logging |
| **Pipelines** | 3 pipelines: traces, metrics, logs |

### Alert Rules (30+ rules across 8 groups)

| Group | Rules | Examples |
|---|---|---|
| **API Performance** | Latency, error rate | P95 > 100ms ⚠️, > 500ms 🔴; Error rate > 1% ⚠️, > 5% 🔴 |
| **ANN Index** | Query latency, corruption | Query > 50ms ⚠️, index corruption 🔴, rebuild needed ⚠️ |
| **Cache** | Hit rate, memory, connections | Hit rate < 80% ⚠️, < 50% 🔴; Memory > 90% ⚠️; Connections > 1000 ⚠️ |
| **Kafka** | Consumer lag, broker health | Lag > 10K ⚠️, > 50K 🔴; Broker down 🔴; Disk > 85% ⚠️ |
| **Feature Store** | Sync lag, connections | Lag > 5min ⚠️, > 30min 🔴; Connections > 500 ⚠️ |
| **Training** | GPU utilization, failures | GPU < 30% ⚠️, GPU memory > 95% 🔴; Stalled/failed jobs 🔴 |
| **Model Performance** | CTR, drift, latency | CTR drop ⚠️; PSI drift > 0.1 ⚠️, > 0.25 🔴; Prediction latency ⚠️ |
| **SLO** | Error budget, availability | Error budget burn rate ⚠️; Latency SLO breach (99.9% target) 🔴 |

### Model Drift Detection

Three statistical methods with automated retraining triggers:

| Method | Metric | Description |
|---|---|---|
| **PSI** | Population Stability Index | Detects distribution shifts between reference and current data |
| **KL Divergence** | Kullback-Leibler | Measures information loss between distributions |
| **KS Test** | Kolmogorov-Smirnov | Non-parametric test for distribution equality |

The `AutomatedRetrainingTrigger` monitors all features and initiates retraining when drift exceeds configurable thresholds, with cooldown periods to prevent thrashing.

---

## 🧪 Testing

### Test Suite Overview

The project includes **46+ tests** across unit and integration suites:

| Test File | Tests | Coverage |
|---|---|---|
| `tests/unit/test_api_routes.py` | 12 | API endpoints (health, recommend, events, features, model, index, metrics) |
| `tests/unit/test_model.py` | 8 | Two-Tower model (forward pass, embeddings, loss, save/load) |
| `tests/unit/test_feature_store.py` | 9 | Online store (set/get, batch, TTL, health, stats) |
| `tests/unit/test_index.py` | 6 | FAISS index (build, search, save/load, IndexManager) |
| `tests/unit/test_streaming.py` | 5 | Kafka consumer (deserializer, DLQ, metrics, handlers) |
| `tests/unit/test_config.py` | 6 | Config defaults, env loading, dataclass validation |
| `tests/integration/test_api_integration.py` | 3 | Full recommendation flow, event ingestion, health |
| `tests/integration/test_feature_pipeline.py` | 3 | Feature sync between online/offline stores |
| `tests/integration/test_streaming_pipeline.py` | 3 | Kafka produce→consume roundtrip |

### Running Tests

```bash
# ── Quick: Use Make ───────────────────────────────
make test                  # Unit tests with coverage
make test-integration      # Integration tests (auto-starts services)

# ── Unit Tests ────────────────────────────────────
pytest tests/unit/ -v --cov=app --cov=training --cov=index \
  --cov=streaming --cov=feature_store --cov-report=term-missing

# Unit tests with HTML coverage report
pytest tests/unit/ --cov=app --cov-report=html

# ── Integration Tests (requires services) ────────
docker-compose -f docker-compose.test.yml up -d
pytest tests/integration/ -v -m integration
docker-compose -f docker-compose.test.yml down
```

### Load Testing

```bash
# ── Locust (4 user types) ────────────────────────
make load-test
# Or manually:
locust -f load_testing/locustfile.py \
  --host http://localhost:8000 \
  --users 1000 --spawn-rate 100 --run-time 300s

# ── k6 (staged performance test) ─────────────────
# Stages: 10 → 50 → 100 → 0 VUs over 6.5 minutes
k6 run load_testing/k6_script.js
```

### Chaos Testing

Built-in chaos engineering framework with **10 experiment types** and Prometheus-based impact assessment:

```bash
make chaos-test
# Or manually:
python load_testing/chaos_testing.py \
  --experiment <type> --duration 300s --namespace rec-engine-prod
```

| Experiment | Description |
|---|---|
| `pod_deletion` | Randomly kills pods to test self-healing |
| `network_partition` | Isolates services to test resilience |
| `cpu_pressure` | CPU stress testing under load |
| `memory_pressure` | Memory stress testing |
| `disk_pressure` | Disk I/O stress testing |
| `kafka_broker_failure` | Kills Kafka brokers to test streaming resilience |
| `redis_node_failure` | Kills Redis nodes to test cache failover |
| `db_connection_failure` | Simulates database connection pool exhaustion |
| `dns_failure` | DNS resolution failures |
| `lb_failure` | Load balancer failure simulation |

**Impact Assessment:** The `ChaosTester` orchestrator collects baseline metrics from Prometheus, executes the chaos experiment, monitors degradation in real-time, measures recovery time, and generates a markdown report classifying impact as `nominal` / `degraded` / `failed`.

---

## 🚢 Deployment

### Docker Compose (Development)

Three compose files for different environments:

```bash
# Full development stack (Redis, PostgreSQL, Kafka, Prometheus, Grafana, Jaeger)
docker-compose up -d

# Integration test stack (minimal: Redis, PostgreSQL, Kafka, Zookeeper)
docker-compose -f docker-compose.test.yml up -d

# Full monitoring stack (Prometheus, Grafana, Jaeger, OTel, exporters)
docker-compose -f docker-compose.monitoring.yml up -d
```

### Docker (Production)

Multi-stage Docker build with non-root user, health checks, and multi-architecture support (amd64 + arm64):

```bash
# Build the image
make build
# Or: docker build -t rec-engine:latest -f infrastructure/docker/Dockerfile .

# Run locally
docker run -p 8000:8000 \
  -e REDIS_URL=redis://host.docker.internal:6379 \
  -e DATABASE_URL=postgresql://user:pass@host.docker.internal:5432/rec_engine \
  rec-engine:latest
```

### Kubernetes

```bash
# Deploy with raw manifests
kubectl apply -f infrastructure/kubernetes/

# Or deploy with Helm
helm install rec-engine infrastructure/helm/rec-engine/ \
  --namespace rec-engine-prod --create-namespace

# Verify deployment
kubectl get pods -n rec-engine-prod
kubectl port-forward service/rec-engine-api 8000:80 -n rec-engine-prod
```

**Kubernetes Resources:**

| Resource | Configuration |
|---|---|
| **API Deployment** | 3 replicas, rolling update (maxSurge: 1, maxUnavailable: 0) |
| **HPA** | 3–50 pods, CPU 70% + Memory 80% + custom QPS/latency metrics |
| **PDB** | minAvailable: 2 (ensures availability during updates) |
| **Ingress** | NGINX with TLS (cert-manager), rate limiting (1000 req/min) |
| **NetworkPolicy** | Restricts traffic between services |
| **Init Container** | Pre-loads model from S3 before serving |
| **Resources** | Request: 500m CPU / 512Mi RAM — Limit: 2 CPU / 2Gi RAM |
| **Probes** | Liveness `/health` (10s), Readiness `/ready` (5s), Startup (5s, 30 retries) |
| **Kafka Cluster** | Strimzi: 3 brokers + 3 ZooKeeper, SASL/SCRAM + TLS |
| **Redis Cluster** | 6 nodes (3 masters + 3 replicas), 12GB maxmemory, LRU eviction |
| **Training Job** | 4 parallel GPU workers (V100/A100), 24h deadline, 500GB data PVC |
| **Helm Chart** | Templatized deployment, service, HPA, ingress, configmap |

### Canary Deployment (Argo Rollouts + Istio)

Progressive traffic shifting with automated analysis and rollback:

```mermaid
graph LR
    A["🚀 Deploy Canary"] --> B["10% Traffic"]
    B -->|"✅ Analysis Pass"| C["25% Traffic"]
    C -->|"✅ Analysis Pass"| D["50% Traffic"]
    D -->|"✅ Analysis Pass"| E["75% Traffic"]
    E -->|"✅ Analysis Pass"| F["100% Traffic"]
    B -->|"❌ Fail"| G["🔄 Auto-Rollback"]
    C -->|"❌ Fail"| G
    D -->|"❌ Fail"| G
    E -->|"❌ Fail"| G
    G --> H["📢 Slack Notification"]
```

**Analysis Templates (automatic gates):**

| Metric | Threshold | Action on Failure |
|---|---|---|
| Success Rate | ≥ 95% | Auto-rollback |
| Latency (P95) | ≤ 100ms | Auto-rollback |
| Error Rate | ≤ 1% | Auto-rollback |

**Istio Integration:**
- `VirtualService` for traffic splitting between stable and canary
- `DestinationRule` with circuit breaker (5 consecutive errors → 30s ejection)
- Separate HPA for canary pods (2–20 replicas)
- NetworkPolicy for canary pod isolation

### CI/CD Pipeline

The GitHub Actions pipeline runs an **8-job deployment process** with matrix testing across Python 3.9/3.10/3.11:

```mermaid
graph TD
    A["🔍 Code Quality<br/><i>flake8 + mypy</i>"] --> D
    B["🧪 Unit Tests<br/><i>pytest + coverage (Python 3.9/3.10/3.11)</i>"] --> D
    C["🔗 Integration Tests<br/><i>Redis + PostgreSQL + Kafka</i>"] --> D
    D["⚡ Performance Tests<br/><i>Locust + k6</i>"] --> E
    E["🏗️ Build Image<br/><i>Docker Buildx (amd64 + arm64)</i>"] --> F
    F["🔒 Security Scan<br/><i>Bandit + Safety + Semgrep + Trivy + Grype</i>"] --> G
    G["🚦 Deploy Staging<br/><i>Kustomize + smoke tests</i>"] --> H
    H["🚀 Deploy Production<br/><i>Canary: 10%→25%→50%→75%→100%</i>"]
    H -->|"❌ Failure"| I["🔄 Auto-Rollback<br/><i>+ Slack notification</i>"]
```

**Security scanning:** Bandit (code), Safety (deps), Semgrep (patterns), Trivy (container), Grype (SBOM), SonarCloud (quality)

---

## 🔮 Roadmap

### Short-term (3–6 months)
- 🎯 **Multi-modal recommendations** — image, text, and audio features
- ⚡ **Real-time personalization** — dynamic user profile updates
- 🎰 **Enhanced A/B testing** — multi-armed bandit algorithms
- 🤖 **AutoML integration** — automated feature engineering and model selection

### Medium-term (6–12 months)
- 🔒 **Federated learning** — privacy-preserving collaborative learning
- 🕸️ **Graph neural networks** — knowledge graph-based recommendations
- 🎮 **Reinforcement learning** — sequential recommendation optimization
- 📲 **Edge computing** — on-device inference for ultra-low latency

### Long-term (12+ months)
- 🧬 **Explainable AI** — interpretable recommendation models
- 🌐 **Cross-domain recommendations** — unified suggestions across platforms
- 🌱 **Sustainable AI** — energy-efficient model architectures
- 📊 **GraphQL API** — flexible query interface
- 📦 **Event sourcing** — complete audit trail of user interactions

---

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Development Workflow

```bash
# 1. Fork the repository
# 2. Create a feature branch
git checkout -b feature/amazing-feature

# 3. Set up pre-commit hooks (one-time)
pip install pre-commit
pre-commit install

# 4. Make changes and commit (conventional commits)
git commit -m "feat: add amazing feature"

# 5. Run checks before pushing
make lint
make test

# 6. Push and open a PR
git push origin feature/amazing-feature
```

### Code Quality Standards

| Tool | Purpose | Command |
|---|---|---|
| `black` | Code formatting (line-length=100) | `make format` |
| `isort` | Import sorting (black profile) | `make format` |
| `flake8` | PEP 8 linting | `make lint` |
| `mypy` | Static type checking | `make lint` |
| `bandit` | Security analysis | `bandit -r . --skip=B101` |
| `pytest` | Tests (all must pass) | `make test` |
| `pre-commit` | Git hooks (auto on commit) | `pre-commit run --all-files` |

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---
---

## 🙏 Acknowledgments

- Open source community for the amazing ML/AI libraries
- Netflix, Meta, and Google for architecture inspiration and best practices
- All contributors and users who help improve this project

---

<div align="center">

**Built with ❤️ by the Recommendation Engine Team**

*Powering personalized experiences at scale*

</div>
