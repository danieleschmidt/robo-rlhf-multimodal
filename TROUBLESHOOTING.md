# Troubleshooting Guide

## Common Issues and Solutions

### 1. Autonomous SDLC Failures

#### Generation 1 Issues
**Problem**: Basic functionality tests failing
```bash
❌ GENERATION 1 FAILED - REVIEW REQUIRED
```

**Solutions**:
```bash
# Check Python environment
python3 --version  # Should be 3.8+

# Verify dependencies
pip list | grep numpy
pip install -e .

# Run individual tests
python generation1_minimal_test.py
```

#### Generation 2 Issues  
**Problem**: Security validation failures
```bash
Security scan: 15 files, 2 issues
❌ Generation 2 partially successful
```

**Solutions**:
```bash
# Review security issues
grep -r "eval\|exec" robo_rlhf/
grep -r "password.*=" robo_rlhf/

# Fix hardcoded secrets
git secret scan
bandit -r robo_rlhf/
```

#### Generation 3 Issues
**Problem**: Performance degradation
```bash
Parallel computation: 0 tasks, 0.0 ops/sec
⚠️ Generation 3 partial success
```

**Solutions**:
```bash
# Check multiprocessing
python -c "import multiprocessing; print(multiprocessing.cpu_count())"

# Monitor resource usage
top -p $(pgrep -f python)
htop

# Tune worker processes
export WORKERS=4
```

### 2. Quality Gates Failures

#### Code Quality Issues
**Problem**: Code quality score below 90%
```bash
Quality gate 'Code Quality - Basic': warning (score: 70)
```

**Solutions**:
```bash
# Run code formatter
black robo_rlhf/
isort robo_rlhf/

# Fix linting issues
flake8 robo_rlhf/
pylint robo_rlhf/

# Type checking
mypy robo_rlhf/
```

#### Test Coverage Issues
**Problem**: Low test coverage
```bash
Test Coverage: fail (score: 30)
Coverage: 0.4%
```

**Solutions**:
```bash
# Run existing tests
pytest tests/ -v

# Check coverage
pytest --cov=robo_rlhf --cov-report=html

# Add missing tests
# Create tests/test_[module].py files
```

### 3. Global-First Implementation Issues

#### Localization Problems
**Problem**: Translation not working
```bash
Translation not found for key 'welcome' in locale 'es_ES'
```

**Solutions**:
```bash
# Verify locale support
python -c "
from global_first_simplified import GlobalFirstEngine
engine = GlobalFirstEngine()
print(engine.supported_languages)
"

# Check translation files
ls -la locales/
```

#### Compliance Issues  
**Problem**: GDPR compliance failure
```bash
GDPR compliance: ❌ NON-COMPLIANT
Missing: 2 requirements need implementation
```

**Solutions**:
```bash
# Review compliance requirements
python -c "
from global_first_simplified import GlobalFirstEngine
engine = GlobalFirstEngine()  
result = engine.implement_compliance_framework()
print(result['regions'])
"

# Implement missing features
# - Right to deletion
# - Data portability
# - Consent management
```

### 4. Deployment Issues

#### Container Issues
**Problem**: Docker build failures
```bash
ERROR: Could not find a version that satisfies the requirement torch>=1.12.0
```

**Solutions**:
```bash
# Use multi-stage build
docker build --target production -t robo-rlhf .

# Check base image
docker run --rm python:3.11-slim pip list | grep torch

# Clear build cache
docker builder prune
```

#### Kubernetes Issues
**Problem**: Pods not starting
```bash
kubectl get pods
NAME                     READY   STATUS    RESTARTS   AGE
robo-rlhf-xxx-xxx        0/1     Pending   0          5m
```

**Solutions**:
```bash
# Check pod events
kubectl describe pod robo-rlhf-xxx-xxx

# Check node resources
kubectl top nodes

# Scale down and up
kubectl scale deployment robo-rlhf --replicas=0
kubectl scale deployment robo-rlhf --replicas=3
```

#### Database Connection Issues
**Problem**: Database connection timeout
```bash
sqlalchemy.exc.OperationalError: could not connect to server
```

**Solutions**:
```bash
# Test connectivity
pg_isready -h database-host -p 5432

# Check environment variables
echo $DATABASE_URL

# Verify credentials
psql $DATABASE_URL -c "SELECT 1;"

# Check network policy
kubectl get networkpolicies
```

### 5. Performance Issues

#### High Memory Usage
**Problem**: Memory consumption > 4GB
```bash
Container memory usage: 6.2GB/4GB (155%)
OOMKilled
```

**Solutions**:
```bash
# Monitor memory usage
docker stats
kubectl top pods

# Optimize Python memory
export PYTHONMALLOC=malloc
export MALLOC_TRIM_THRESHOLD_=100000

# Use memory profilers
pip install memory-profiler
python -m memory_profiler script.py
```

#### Slow API Response
**Problem**: API response times > 1 second
```bash
Average response time: 2.3s (target: <200ms)
```

**Solutions**:
```bash
# Enable caching
redis-cli ping
redis-cli info memory

# Profile API calls
pip install py-spy
py-spy top --pid $(pgrep -f uvicorn)

# Add async processing
# Convert synchronous code to async/await
```

#### Database Performance
**Problem**: Slow queries
```bash
Query duration: 5.2s (SELECT * FROM large_table)
```

**Solutions**:
```sql
-- Add indexes
CREATE INDEX idx_table_column ON large_table(column);

-- Analyze queries
EXPLAIN ANALYZE SELECT * FROM large_table WHERE column = 'value';

-- Update statistics  
ANALYZE large_table;

-- Connection pooling
-- Set pool_size=20, max_overflow=30
```

### 6. Security Issues

#### SSL Certificate Issues
**Problem**: Certificate expired or invalid
```bash
SSL certificate error: certificate verify failed
```

**Solutions**:
```bash
# Check certificate expiry
openssl x509 -in cert.pem -text -noout | grep "Not After"

# Renew with Let's Encrypt
certbot renew --dry-run

# Update ingress
kubectl patch ingress robo-rlhf-ingress -p '{"spec":{"tls":[{"secretName":"new-tls-secret"}]}}'
```

#### Secret Management Issues
**Problem**: Secrets not found
```bash
Error: Secret "robo-rlhf-secrets" not found
```

**Solutions**:
```bash
# Create secrets
kubectl create secret generic robo-rlhf-secrets   --from-literal=database-password=xxx   --from-literal=jwt-secret=xxx

# Verify secrets
kubectl get secrets
kubectl describe secret robo-rlhf-secrets

# Update deployment to use secrets
kubectl set env deployment/robo-rlhf --from=secret/robo-rlhf-secrets
```

### 7. Monitoring Issues

#### Missing Metrics
**Problem**: Prometheus not collecting metrics
```bash
Warning: No data points found for query
```

**Solutions**:
```bash
# Check metrics endpoint
curl http://localhost:8000/metrics

# Verify Prometheus config
kubectl get configmap prometheus-config -o yaml

# Restart Prometheus
kubectl rollout restart deployment/prometheus
```

#### Log Collection Issues  
**Problem**: Logs not appearing in Elasticsearch
```bash
No logs found in Kibana for the last 24 hours
```

**Solutions**:
```bash
# Check Fluentd status
kubectl get pods -l app=fluentd

# Verify log format
kubectl logs robo-rlhf-xxx-xxx | head -10

# Check Elasticsearch health
curl http://elasticsearch:9200/_cluster/health
```

### 8. Integration Issues

#### External API Failures
**Problem**: Third-party API integration failing
```bash
HTTPError: 429 Client Error: Too Many Requests
```

**Solutions**:
```bash
# Implement retry logic
import tenacity

@tenacity.retry(
    wait=tenacity.wait_exponential(multiplier=1, min=4, max=10),
    stop=tenacity.stop_after_attempt(5)
)
def call_external_api():
    # API call logic
    pass

# Add circuit breaker
pip install circuitbreaker

from circuitbreaker import circuit

@circuit(failure_threshold=5, recovery_timeout=30)
def external_api_call():
    # Protected API call
    pass
```

## Diagnostic Commands

### System Health Check
```bash
#!/bin/bash
# health_check.sh

echo "=== System Health Check ==="

# Python environment
echo "Python version: $(python3 --version)"
echo "Virtual env: $VIRTUAL_ENV"

# Dependencies
echo "Key packages:"
python3 -c "
import sys
packages = ['torch', 'numpy', 'fastapi', 'psutil']
for pkg in packages:
    try:
        exec(f'import {pkg}')
        print(f'  ✅ {pkg}')
    except ImportError:
        print(f'  ❌ {pkg} - missing')
"

# Autonomous SDLC status
echo "=== SDLC Status ==="
python3 generation1_minimal_test.py 2>&1 | grep -E "(✅|❌|COMPLETE|FAILED)"

# Quality gates
echo "=== Quality Gates ==="
python3 comprehensive_quality_gates.py 2>&1 | grep -E "(Gates|Score|Status)"

# Global features
echo "=== Global Features ==="
python3 global_first_simplified.py 2>&1 | grep -E "(✅|Languages|Regions|Score)"

echo "=== Health Check Complete ==="
```

### Performance Profiling
```bash
#!/bin/bash
# performance_profile.sh

echo "=== Performance Profiling ==="

# CPU usage
echo "CPU Info:"
lscpu | grep -E "(Model name|CPU\(s\)|Thread)"

# Memory usage  
echo "Memory Info:"
free -h
ps aux --sort=-%mem | head -10

# Disk usage
echo "Disk Usage:"
df -h
du -sh /tmp/* 2>/dev/null | sort -hr | head -5

# Network
echo "Network Stats:"
ss -tuln | grep -E "(8000|5432|6379)"

# Python process info
echo "Python Processes:"
pgrep -f python | xargs ps -fp

echo "=== Profiling Complete ==="
```

### Database Diagnostics
```sql
-- database_diagnostics.sql

-- Connection info
SELECT 
    count(*) as connection_count,
    state,
    application_name
FROM pg_stat_activity 
WHERE state IS NOT NULL
GROUP BY state, application_name;

-- Query performance
SELECT 
    query,
    calls,
    total_time,
    mean_time,
    rows
FROM pg_stat_statements
ORDER BY total_time DESC
LIMIT 10;

-- Table sizes
SELECT 
    schemaname,
    tablename,
    pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) as size
FROM pg_tables
WHERE schemaname NOT IN ('information_schema', 'pg_catalog')
ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC;

-- Index usage
SELECT
    indexrelname as index_name,
    idx_tup_read,
    idx_tup_fetch,
    idx_scan
FROM pg_stat_user_indexes
ORDER BY idx_tup_read DESC;
```

## Emergency Procedures

### Complete System Restart
```bash
#!/bin/bash
# emergency_restart.sh

echo "🚨 EMERGENCY RESTART PROCEDURE"
echo "This will restart all system components"
read -p "Continue? (y/N): " confirm

if [ "$confirm" != "y" ]; then
    echo "Aborted"
    exit 1
fi

# Scale down
kubectl scale deployment robo-rlhf --replicas=0
kubectl wait --for=delete pod -l app=robo-rlhf --timeout=60s

# Clear cache
redis-cli FLUSHALL

# Scale up
kubectl scale deployment robo-rlhf --replicas=3
kubectl wait --for=condition=available deployment/robo-rlhf --timeout=300s

# Verify
kubectl get pods -l app=robo-rlhf
curl -f http://localhost:8000/health || echo "❌ Health check failed"

echo "✅ Emergency restart complete"
```

### Rollback Procedure
```bash
#!/bin/bash
# rollback.sh

PREVIOUS_VERSION=${1:-"previous"}

echo "🔄 ROLLING BACK TO: $PREVIOUS_VERSION"

# Rollback deployment
kubectl rollout undo deployment/robo-rlhf --to-revision=$PREVIOUS_VERSION

# Wait for rollback
kubectl rollout status deployment/robo-rlhf --timeout=300s

# Verify health
sleep 30
curl -f http://localhost:8000/health

echo "✅ Rollback complete"
```

## Support Contacts

### Escalation Matrix

| Severity | Response Time | Contact |
|----------|---------------|---------|
| P0 - Critical | 15 minutes | +1-555-ROBO-RLHF |
| P1 - High | 2 hours | devops@robo-rlhf.ai |
| P2 - Medium | 24 hours | support@robo-rlhf.ai |
| P3 - Low | 72 hours | github.com/issues |

### Team Contacts
- **DevOps**: devops@robo-rlhf.ai
- **Security**: security@robo-rlhf.ai  
- **Architecture**: architects@robo-rlhf.ai
- **QA**: qa@robo-rlhf.ai

---

**Last Updated**: January 23, 2025
**Version**: 1.0.0
