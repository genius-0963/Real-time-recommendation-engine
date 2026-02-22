# Rollback Strategy for Netflix/Meta Scale Recommendation Engine
# Comprehensive rollback procedures with automated and manual options

## Overview

This document outlines the rollback strategy for the recommendation engine, covering various scenarios from canary deployment failures to production incidents. The strategy ensures minimal downtime and maintains service availability during rollback operations.

## Rollback Triggers

### Automated Rollback Triggers

1. **Canary Analysis Failure**
   - Success rate < 95% for 3 consecutive checks
   - P95 latency > 100ms for 3 consecutive checks
   - Error rate > 1% for 3 consecutive checks
   - Health check failures > 50% of pods

2. **Production Deployment Failure**
   - Pod startup failures > 20%
   - Health check timeouts > 30 seconds
   - Critical alerts firing (P99 latency > 500ms, error rate > 5%)

3. **Performance Degradation**
   - QPS drop > 50% compared to baseline
   - CTR drop > 20% compared to baseline
   - Memory usage > 90% for 5+ minutes

### Manual Rollback Triggers

1. **Business Impact**
   - Revenue impact > $10,000/hour
   - User complaints > 100/hour
   - SLA breach notifications

2. **Security Concerns**
   - Vulnerability discovery in new version
   - Data leakage incidents
   - Authentication failures

## Rollback Procedures

### 1. Canary Rollback (Automated)

#### Trigger Conditions
```yaml
# Argo Rollout configuration
analysis:
  threshold:
    pass: 95
    fail: 80
  rollbackOnFailure: true
  unsuccessfulRunHistoryLimit: 2
```

#### Automated Rollback Steps
1. **Detection**: Canary analysis detects failure
2. **Immediate Action**: Stop traffic to canary (0% weight)
3. **Cleanup**: Delete canary deployment
4. **Alerting**: Notify on-call team
5. **Post-mortem**: Create incident ticket

#### Commands
```bash
# Manual canary rollback
kubectl argo rollouts rollback rec-engine-api-canary -n rec-engine-prod

# Check rollback status
kubectl argo rollouts get rollout rec-engine-api-canary -n rec-engine-prod

# View rollback history
kubectl argo rollouts history rec-engine-api-canary -n rec-engine-prod
```

### 2. Production Rollback (Manual)

#### Immediate Rollback (Emergency)
```bash
# Quick rollback to previous stable version
kubectl rollout undo deployment/rec-engine-api -n rec-engine-prod

# Monitor rollback progress
kubectl rollout status deployment/rec-engine-api -n rec-engine-prod --timeout=300s

# Verify service health
curl -f https://api.rec-engine.company.com/health
```

#### Controlled Rollback
```bash
# Step 1: Scale down current deployment
kubectl scale deployment rec-engine-api --replicas=0 -n rec-engine-prod

# Step 2: Deploy previous version
kubectl apply -f infrastructure/kubernetes/api-deployment-previous.yaml -n rec-engine-prod

# Step 3: Wait for deployment
kubectl rollout status deployment/rec-engine-api -n rec-engine-prod --timeout=600s

# Step 4: Verify health
kubectl get pods -n rec-engine-prod -l app=rec-engine-api
curl -f https://api.rec-engine.company.com/health
```

### 3. Database Rollback

#### Feature Store Rollback
```bash
# Rollback feature store schema
psql -h postgres.rec-engine-prod.svc.cluster.local -U postgres -d rec_engine_prod << EOF
-- Rollback to previous schema version
ALTER SCHEMA public RENAME TO current_schema;
ALTER SCHEMA previous_schema RENAME TO public;
ALTER SCHEMA current_schema RENAME TO previous_schema;
EOF

# Verify schema rollback
psql -h postgres.rec-engine-prod.svc.cluster.local -U postgres -d rec_engine_prod -c "\dt"
```

#### Model Rollback
```bash
# Rollback model to previous version
aws s3 cp s3://rec-engine-models/prod/previous/ /models/ --recursive

# Update model symlink
ln -sf /models/previous /models/current

# Restart API pods to load new model
kubectl rollout restart deployment/rec-engine-api -n rec-engine-prod
```

## Rollback Automation Scripts

### Automated Rollback Script
```bash
#!/bin/bash
# rollback.sh - Automated rollback script

set -euo pipefail

NAMESPACE="rec-engine-prod"
DEPLOYMENT="rec-engine-api"
SERVICE="rec-engine-api"
HEALTH_URL="https://api.rec-engine.company.com/health"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

# Health check function
health_check() {
    local max_attempts=30
    local attempt=1
    
    while [ $attempt -le $max_attempts ]; do
        if curl -f -s "$HEALTH_URL" > /dev/null; then
            log_info "Health check passed (attempt $attempt/$max_attempts)"
            return 0
        fi
        
        log_warning "Health check failed (attempt $attempt/$max_attempts)"
        sleep 10
        ((attempt++))
    done
    
    log_error "Health check failed after $max_attempts attempts"
    return 1
}

# Rollback function
rollback_deployment() {
    local reason="$1"
    
    log_info "Starting rollback due to: $reason"
    
    # Get current revision
    local current_revision=$(kubectl rollout history deployment/$DEPLOYMENT -n $NAMESPACE -o jsonpath='{.status.currentRevision}')
    log_info "Current revision: $current_revision"
    
    # Perform rollback
    log_info "Rolling back deployment..."
    kubectl rollout undo deployment/$DEPLOYMENT -n $NAMESPACE
    
    # Wait for rollback to complete
    log_info "Waiting for rollback to complete..."
    kubectl rollout status deployment/$DEPLOYMENT -n $NAMESPACE --timeout=300s
    
    # Verify health
    log_info "Verifying service health..."
    if health_check; then
        log_info "Rollback completed successfully"
        
        # Send notification
        send_notification "Rollback completed successfully" "success"
        
        # Create incident ticket
        create_incident "Rollback completed: $reason"
    else
        log_error "Health check failed after rollback"
        
        # Send critical notification
        send_notification "Rollback failed - manual intervention required" "critical"
        
        # Escalate to on-call
        escalate_incident "Rollback failed - immediate attention required"
        exit 1
    fi
}

# Send notification
send_notification() {
    local message="$1"
    local severity="$2"
    
    # Send to Slack
    curl -X POST -H 'Content-type: application/json' \
        --data "{\"text\":\"$message\", \"severity\":\"$severity\"}" \
        "$SLACK_WEBHOOK_URL"
    
    # Send to PagerDuty if critical
    if [ "$severity" = "critical" ]; then
        curl -X POST -H 'Content-type: application/json' \
            --data "{\"routing_key\":\"$PAGERDUTY_ROUTING_KEY\", \"event_action\":\"trigger\", \"payload\":{\"summary\":\"$message\", \"severity\":\"critical\", \"source\":\"rollback-script\"}}" \
            "https://events.pagerduty.com/v2/enqueue"
    fi
}

# Create incident
create_incident() {
    local summary="$1"
    
    # Create incident in your incident management system
    curl -X POST -H 'Content-type: application/json' \
        --data "{\"summary\":\"$summary\", \"severity\":\"medium\", \"source\":\"rollback-script\"}" \
        "$INCIDENT_API_URL"
}

# Escalate incident
escalate_incident() {
    local summary="$1"
    
    # Escalate to on-call
    curl -X POST -H 'Content-type: application/json' \
        --data "{\"summary\":\"$summary\", \"severity\":\"critical\", \"source\":\"rollback-script\"}" \
        "$ESCALATION_API_URL"
}

# Main execution
main() {
    local reason="${1:-Manual rollback}"
    
    log_info "Starting rollback process..."
    
    # Check if deployment exists
    if ! kubectl get deployment $DEPLOYMENT -n $NAMESPACE > /dev/null 2>&1; then
        log_error "Deployment $DEPLOYMENT not found in namespace $NAMESPACE"
        exit 1
    fi
    
    # Perform rollback
    rollback_deployment "$reason"
    
    log_info "Rollback process completed"
}

# Execute main function
main "$@"
```

### Canary Rollback Script
```bash
#!/bin/bash
# canary_rollback.sh - Canary-specific rollback

set -euo pipefail

NAMESPACE="rec-engine-prod"
ROLLOUT="rec-engine-api-canary"

log_info() {
    echo -e "\033[0;32m[INFO]\033[0m $1"
}

# Rollback canary
rollback_canary() {
    log_info "Rolling back canary deployment..."
    
    # Abort canary rollout
    kubectl argo rollouts abort $ROLLOUT -n $NAMESPACE
    
    # Reset traffic to stable
    kubectl patch virtualservice rec-engine-api -n $NAMESPACE \
        --type='json' \
        -p='[{"op": "replace", "path": "/spec/http/0/route/0/weight", "value":100},
             {"op": "remove", "path": "/spec/http/0/route/1"}]'
    
    # Delete canary deployment
    kubectl delete rollout $ROLLOUT -n $NAMESPACE --ignore-not-found=true
    
    log_info "Canary rollback completed"
}

# Execute rollback
rollback_canary
```

## Rollback Validation

### Health Checks
```bash
# API health check
curl -f https://api.rec-engine.company.com/health

# Metrics endpoint
curl -f https://api.rec-engine.company.com/metrics

# Load test
locust -f tests/performance/smoke_test.py --host https://api.rec-engine.company.com --users 10 --run-time 60s --headless
```

### Performance Validation
```bash
# Check latency
curl -s https://api.rec-engine.company.com/metrics | grep 'request_latency_seconds_bucket{le="0.1"}'

# Check error rate
curl -s https://api.rec-engine.company.com/metrics | grep 'http_requests_total{status=~"5.."}'

# Check QPS
curl -s https://api.rec-engine.company.com/metrics | grep 'http_requests_total'
```

### Business Metrics Validation
```bash
# Check CTR
curl -s https://metrics.rec-engine.company.com/api/v1/query?query=rate(clicks_total[5m])/rate(impressions_total[5m])

# Check engagement rate
curl -s https://metrics.rec-engine.company.com/api/v1/query?query=rate(user_engagement_events_total[5m])/rate(recommendation_requests_total[5m])
```

## Rollback Communication

### Alert Templates

#### Rollback Initiated
```
🚨 Rollback Initiated

Service: Recommendation Engine
Environment: Production
Reason: [Reason for rollback]
Time: [Timestamp]
Impact: [Expected impact]

Actions:
- Rolling back to previous version
- Monitoring service health
- Will update when complete

On-call: [On-call engineer]
```

#### Rollback Completed
```
✅ Rollback Completed

Service: Recommendation Engine
Environment: Production
Previous Version: [Version]
Current Version: [Rolled back version]
Time: [Timestamp]
Duration: [Rollback duration]

Status: Healthy
Performance: Normal
Next Steps: Investigate root cause

On-call: [On-call engineer]
```

### Post-Rollback Actions

1. **Immediate Actions**
   - Verify service health
   - Monitor key metrics
   - Update status page
   - Notify stakeholders

2. **Investigation**
   - Analyze failure logs
   - Review deployment changes
   - Identify root cause
   - Document findings

3. **Prevention**
   - Update deployment process
   - Add additional tests
   - Improve monitoring
   - Update rollback procedures

## Rollback Testing

### Test Scenarios
1. **Canary Failure Test**
   - Deploy canary with intentional failure
   - Verify automatic rollback
   - Check traffic routing

2. **Production Failure Test**
   - Deploy broken version to production
   - Test manual rollback
   - Verify recovery time

3. **Database Rollback Test**
   - Apply schema changes
   - Test rollback procedure
   - Verify data integrity

### Test Validation
```bash
# Run rollback tests
./tests/rollback/test_canary_rollback.sh
./tests/rollback/test_production_rollback.sh
./tests/rollback/test_database_rollback.sh

# Validate rollback time
./tests/rollback/performance_test.sh --target-time 300s
```

## Rollback Metrics

### Key Performance Indicators
- **MTTR (Mean Time To Recovery)**: < 5 minutes
- **Rollback Success Rate**: > 99%
- **Service Downtime**: < 2 minutes
- **Data Loss**: 0%

### Monitoring
```yaml
# Prometheus alerts for rollback metrics
- alert: RollbackInitiated
  expr: increase(rollback_initiated_total[5m]) > 0
  for: 0m
  labels:
    severity: warning
  annotations:
    summary: "Rollback initiated for recommendation engine"

- alert: RollbackFailed
  expr: increase(rollback_failed_total[5m]) > 0
  for: 0m
  labels:
    severity: critical
  annotations:
    summary: "Rollback failed - manual intervention required"
```

## Emergency Procedures

### Complete Service Outage
1. **Immediate Action**: Scale to previous stable version
2. **Communication**: Notify all stakeholders
3. **Investigation**: Full post-mortem
4. **Prevention**: Review and improve procedures

### Partial Service Degradation
1. **Assessment**: Determine impact scope
2. **Selective Rollback**: Rollback affected components
3. **Monitoring**: Enhanced monitoring
4. **Communication**: Status updates

### Security Incident
1. **Immediate Isolation**: Block affected traffic
2. **Rollback**: Return to known good state
3. **Investigation**: Security team involvement
4. **Reporting**: Compliance requirements

## Documentation and Training

### Runbook Updates
- Update rollback procedures after each incident
- Include lessons learned
- Maintain contact information
- Document emergency procedures

### Team Training
- Regular rollback drills
- Emergency response training
- Communication protocols
- Tool usage training

### Continuous Improvement
- Review rollback metrics
- Identify improvement opportunities
- Update automation
- Enhance monitoring

## Conclusion

This rollback strategy ensures that the recommendation engine can quickly and safely recover from deployment failures while maintaining service availability. The combination of automated and manual procedures provides flexibility for different scenarios while maintaining operational excellence.

Regular testing and continuous improvement of rollback procedures are essential for maintaining high availability and minimizing the impact of deployment failures.
