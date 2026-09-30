# Security Implementation Guide

This document describes the security implementation for the Real-time Recommendation Engine.

## Overview

The security implementation follows a **zero-trust architecture** with defense-in-depth:

1. **Authentication**: JWT/OIDC with JWKS validation
2. **Authorization**: RBAC with scope-based permissions
3. **Secrets Management**: HashiCorp Vault / AWS Secrets Manager integration
4. **Transport Security**: TLS 1.3 everywhere (mTLS for service-to-service)
5. **Network Segmentation**: Kubernetes NetworkPolicies
6. **Supply Chain Security**: Pre-commit secret scanning, signed images, SBOM

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        EXTERNAL CLIENTS                         │
└─────────────────────────┬───────────────────────────────────────┘
                          │ HTTPS/TLS 1.3
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                     INGRESS NGINX (TLS termination)             │
│  - Cert-manager (Let's Encrypt)                                 │
│  - Security headers (HSTS, CSP, X-Frame-Options)                │
│  - Rate limiting                                                │
└─────────────────────────┬───────────────────────────────────────┘
                          │ mTLS
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                    REC-ENGINE API PODS                          │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ AuthMiddleware                                          │   │
│  │ - JWT validation (RS256 via JWKS / HS256 via Vault)    │   │
│  │ - Scope-based RBAC                                      │   │
│  │ - Audit logging                                         │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────┬───────────────────────────────────────┘
                          │ mTLS (all service connections)
          ┌───────────────┼───────────────┐
          ▼               ▼               ▼
    ┌─────────┐     ┌─────────┐     ┌─────────┐
    │  REDIS  │     │POSTGRES │     │  KAFKA  │
    │  TLS    │     │  TLS    │     │ SASL_SSL│
    └─────────┘     └─────────┘     └─────────┘
          │               │               │
          └───────────────┼───────────────┘
                          ▼
              ┌───────────────────────┐
              │    HASHICORP VAULT    │
              │  (Secrets Management) │
              └───────────────────────┘
```

## Quick Start (Development)

### 1. Generate Certificates

```bash
cd scripts
./generate_certs.sh
```

This creates:
- CA certificate (`certs/ca.crt`)
- Server certificates for each service
- Client certificates for mTLS
- Java keystores for Kafka/ZooKeeper/Schema Registry
- Kubernetes secret manifest (`certs/k8s-secret.yaml`)

### 2. Start with TLS

```bash
docker-compose -f docker-compose.yml -f docker-compose.tls.yml up -d
```

### 3. Verify TLS

```bash
# Test API with client certificate
curl --cert certs/api-client.crt --key certs/api-client.key \
     --cacert certs/ca.crt https://localhost:8443/health
```

## Production Deployment

### Prerequisites

1. **Kubernetes cluster** with:
   - cert-manager installed
   - NGINX Ingress Controller
   - NetworkPolicy support (Calico, Cilium, etc.)
   - Prometheus Operator (for ServiceMonitor)

2. **HashiCorp Vault** or **AWS Secrets Manager** configured

3. **OIDC Provider** (Keycloak, Auth0, Azure AD, etc.) with:
   - JWKS endpoint
   - Client configured for rec-engine-api audience

### Deploy with Helm

```bash
# Install cert-manager
helm repo add jetstack https://charts.jetstack.io
helm install cert-manager jetstack/cert-manager \
  --namespace cert-manager --create-namespace \
  --set installCRDs=true

# Create Vault role for Kubernetes auth
vault write auth/kubernetes/role/rec-engine \
  bound_service_account_names=rec-engine-api \
  bound_service_account_namespaces=rec-engine \
  policies=rec-engine \
  ttl=24h

# Deploy rec-engine
helm install rec-engine ./infrastructure/helm/rec-engine \
  -f ./infrastructure/helm/rec-engine/values-prod.yaml \
  --namespace rec-engine --create-namespace
```

### Apply Network Policies

```bash
kubectl apply -k ./infrastructure/kubernetes/network-policies/
```

## Configuration

### Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `SECURITY_SECRET_BACKEND` | `vault`, `aws_secrets_manager`, or `env` | Yes |
| `VAULT_ADDR` | Vault server URL | If using Vault |
| `VAULT_ROLE` | Kubernetes auth role | If using Vault |
| `OIDC_JWKS_URL` | JWKS endpoint for JWT validation | Yes |
| `OIDC_ISSUER` | Token issuer | Yes |
| `OIDC_AUDIENCE` | Expected audience | Yes |
| `SECURITY_CORS_ORIGINS` | Comma-separated allowed origins | Yes |

### Secret Structure (Vault/AWS Secrets Manager)

```
rec-engine/
├── kafka/
│   ├── username: "api-user"
│   ├── password: "..."
│   ├── ca_file: "/certs/ca.crt"
│   ├── cert_file: "/certs/kafka-client.crt"
│   └── key_file: "/certs/kafka-client.key"
├── redis/
│   ├── password: "..."
│   ├── ca_file: "/certs/ca.crt"
│   ├── cert_file: "/certs/redis-client.crt"
│   └── key_file: "/certs/redis-client.key"
├── database/
│   ├── password: "..."
│   ├── ca_file: "/certs/ca.crt"
│   ├── cert_file: "/certs/postgres-client.crt"
│   └── key_file: "/certs/postgres-client.key"
└── api/
    ├── cert_file: "/certs/api-server.crt"
    └── key_file: "/certs/api-server.key"
```

## RBAC Permissions

| Scope | Endpoints | Description |
|-------|-----------|-------------|
| `rec:read` | `/recommend`, `/user/*/features`, `/item/*/features` | Read recommendations and features |
| `rec:write` | `/feedback` | Submit feedback |
| `metrics:read` | `/metrics` | Read Prometheus metrics |
| `experiments:read` | `/experiments` | View experiments |
| `experiments:write` | `/experiments/*/assign` | Assign users to experiments |
| `rec:admin` | `/admin/*`, `/index/*` | Administrative operations |

### Role Mapping

| Role | Scopes |
|------|--------|
| `rec-engine-admin` | All scopes |
| `rec-engine-user` | `rec:read`, `rec:write` |
| `data-scientist` | `rec:read`, `metrics:read`, `experiments:read` |

## Security Headers

The following headers are automatically added by the ingress controller:

- `Strict-Transport-Security: max-age=31536000; includeSubDomains`
- `X-Frame-Options: DENY`
- `X-Content-Type-Options: nosniff`
- `Content-Security-Policy: default-src 'self'; ...`
- `Referrer-Policy: strict-origin-when-cross-origin`

## Audit Logging

All authentication and authorization decisions are logged:

```json
{
  "timestamp": "2024-01-15T10:30:00Z",
  "level": "INFO",
  "event": "auth_decision",
  "actor": "user_123",
  "action": "recommend",
  "resource": "/recommend",
  "decision": "allow",
  "scopes": ["rec:read"],
  "roles": ["rec-engine-user"],
  "correlation_id": "req-abc123"
}
```

## Pre-commit Secret Scanning

The pre-commit hook scans for:

- API keys (AWS, GitHub, Slack, OpenAI, Google)
- JWT tokens
- Private keys (SSH, PEM)
- Database connection strings
- Passwords in code

### Installation

```bash
pip install pre-commit
pre-commit install
```

### Bypass (Emergency Only)

```bash
git commit --no-verify
```

## Compliance

This implementation supports:

- **SOC 2 Type II**: Audit logging, access controls, encryption
- **GDPR**: Data minimization, right to deletion (implement `DELETE /user/{id}/data`)
- **PCI DSS**: Network segmentation, encryption, access logging
- **HIPAA**: If handling PHI, configure additional audit controls

## Incident Response

### Compromised Certificate

1. Revoke in CA
2. Rotate affected service certificates
3. Update Kubernetes secrets
4. Rollout pods

### Compromised Vault Token

1. Revoke token in Vault
2. Rotate all secrets accessed by token
3. Update Kubernetes service accounts

### Suspicious Activity

1. Check audit logs for failed auth
2. Review NetworkPolicy alerts
3. Check Prometheus alerts for anomaly detection

## Testing

### Run Security Tests

```bash
# Unit tests
pytest tests/unit/test_security.py -v

# Integration tests
pytest tests/integration/test_security_integration.py -v

# Chaos testing (in staging)
python load_testing/chaos_testing.py --scenario network-partition
```

### Penetration Testing Checklist

- [ ] JWT validation bypass attempts
- [ ] SQL injection in feature store
- [ ] NoSQL injection in Redis
- [ ] Path traversal in Parquet import
- [ ] Deserialization attacks (pickle removed)
- [ ] Rate limiting bypass
- [ ] CORS misconfiguration
- [ ] Network policy enforcement

## Troubleshooting

### Certificate Issues

```bash
# Verify certificate chain
openssl verify -CAfile certs/ca.crt certs/api-server.crt

# Check certificate expiry
openssl x509 -in certs/api-server.crt -text -noout | grep -A2 Validity

# Test mTLS connection
openssl s_client -connect localhost:8443 -cert certs/api-client.crt -key certs/api-client.key -CAfile certs/ca.crt
```

### Vault Issues

```bash
# Check Vault agent logs
kubectl logs -l app=rec-engine-api -c vault-agent

# Test Vault connectivity
vault read rec-engine/kafka
```

### Network Policy Issues

```bash
# Test connectivity between pods
kubectl exec -it api-pod -- nc -zv redis-master 6380

# Check NetworkPolicy status
kubectl get networkpolicies -n rec-engine
kubectl describe networkpolicy rec-engine-api -n rec-engine
```