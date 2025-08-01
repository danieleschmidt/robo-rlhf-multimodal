# Incident Response Runbook

## Overview
This runbook provides step-by-step procedures for responding to various types of incidents in the Robo-RLHF-Multimodal system.

## Incident Classification

### Severity Levels
- **P0 (Critical)**: Complete system outage, data loss, security breach
- **P1 (High)**: Partial outage, significant degradation, failed deployment
- **P2 (Medium)**: Performance issues, non-critical failures, alerts
- **P3 (Low)**: Minor issues, cosmetic problems, enhancements

### Incident Types
- **Service Outage**: Application or component completely unavailable
- **Performance Degradation**: Slow response times, high latency
- **Data Issues**: Corruption, loss, or inconsistency
- **Security Incident**: Breach, unauthorized access, vulnerability
- **Infrastructure**: Hardware, network, or cloud provider issues

## Initial Response (First 15 Minutes)

### 1. Acknowledge and Assess
```bash
# Check overall system health
make health-check

# Check service status
docker compose ps

# Review recent alerts
curl http://localhost:9090/api/v1/alerts
```

### 2. Determine Severity
- **P0/P1**: Immediate escalation and war room
- **P2**: Standard response procedures  
- **P3**: Normal business hours handling

### 3. Establish Communication
- Create incident channel: #incident-YYYY-MM-DD-HH-MM
- Post initial status update
- Notify stakeholders based on severity

### 4. Initial Triage
```bash
# Check application logs
make logs-app | tail -100

# Check system resources
docker stats

# Check database connectivity
docker compose exec mongodb mongosh --eval "db.adminCommand('ping')"

# Check Redis connectivity
docker compose exec redis redis-cli ping
```

## Service Outage Response

### Symptoms
- Health check failures
- 5xx HTTP responses
- Connection timeouts
- Container restarts

### Investigation Steps

1. **Check Container Status**
```bash
# View container status
docker compose ps

# Check container logs
docker compose logs robo-rlhf --tail=100

# Check resource usage
docker stats --no-stream
```

2. **Check Dependencies**
```bash
# Database connectivity
docker compose exec robo-rlhf curl -f mongodb:27017

# Redis connectivity  
docker compose exec robo-rlhf curl -f redis:6379

# External service checks
curl -f https://api.wandb.ai/health
```

3. **Check System Resources**
```bash
# Disk space
df -h

# Memory usage
free -h

# CPU load
uptime
```

### Resolution Steps

1. **Restart Failed Services**
```bash
# Restart specific service
docker compose restart robo-rlhf

# Full system restart (if necessary)
docker compose down
docker compose up -d
```

2. **Scale Services**
```bash
# Scale application instances
docker compose up -d --scale robo-rlhf=3
```

3. **Emergency Maintenance**
```bash
# Enable maintenance mode
echo "MAINTENANCE_MODE=true" >> .env
docker compose restart robo-rlhf
```

## Performance Degradation Response

### Symptoms
- High response times (>5s)
- Increased error rates
- Resource exhaustion alerts
- Queue backlogs

### Investigation Steps

1. **Check Performance Metrics**
```bash
# View Grafana dashboards
open http://localhost:3000

# Check Prometheus metrics
curl http://localhost:9090/api/v1/query?query=rate(http_request_duration_seconds_sum[5m])
```

2. **Analyze Bottlenecks**
```bash
# Database performance
docker compose exec mongodb mongosh --eval "db.currentOp()"

# Redis performance
docker compose exec redis redis-cli info stats

# Application profiling
docker compose exec robo-rlhf python -m py_spy top -p 1
```

3. **Check Resource Limits**
```bash
# Container resource usage
docker stats

# Host system resources
htop
```

### Resolution Steps

1. **Immediate Relief**
```bash
# Scale up services
docker compose up -d --scale robo-rlhf=2

# Restart memory-intensive services
docker compose restart robo-rlhf
```

2. **Database Optimization**
```bash
# Clear slow queries
docker compose exec mongodb mongosh --eval "db.runCommand({planCacheClear: 1})"

# Redis memory cleanup
docker compose exec redis redis-cli FLUSHEXPIRED
```

3. **Enable Circuit Breakers**
```python
# In application code
CIRCUIT_BREAKER_ENABLED = True
MAX_CONCURRENT_REQUESTS = 100
```

## Data Issues Response

### Symptoms
- Data corruption alerts
- Backup failures
- Inconsistent query results
- Validation errors

### Investigation Steps

1. **Assess Data Integrity**
```bash
# Check database status
docker compose exec mongodb mongosh --eval "db.runCommand({dbStats: 1})"

# Verify recent backups
ls -la backups/ | head -10

# Check data validation
docker compose exec robo-rlhf python -m robo_rlhf.scripts.validate_data
```

2. **Identify Scope**
```bash
# Check affected collections
docker compose exec mongodb mongosh --eval "db.adminCommand('listCollections')"

# Check data volumes
du -sh data/
```

### Resolution Steps

1. **Stop Data Writes**
```bash
# Enable read-only mode
echo "READ_ONLY_MODE=true" >> .env
docker compose restart robo-rlhf
```

2. **Create Emergency Backup**
```bash
make db-backup
cp -r data/ data_backup_$(date +%Y%m%d_%H%M%S)
```

3. **Restore from Backup**
```bash
# Stop services
docker compose stop robo-rlhf

# Restore database
make db-reset
docker compose exec mongodb mongorestore /data/backup/latest

# Restart services
docker compose start robo-rlhf
```

## Security Incident Response

### Symptoms
- Unauthorized access alerts
- Unusual network traffic
- Failed authentication spikes
- Security scan alerts

### Investigation Steps

1. **Immediate Assessment**
```bash
# Check authentication logs
docker compose logs robo-rlhf | grep -i "auth\|login\|unauthorized"

# Check network connections
netstat -tuln
```

2. **Isolate Affected Systems**
```bash
# Remove from load balancer
# Block suspicious IPs at firewall level
# Rotate API keys and tokens
```

### Resolution Steps

1. **Contain the Incident**
```bash
# Disable affected accounts
# Change passwords and API keys
# Enable additional logging
echo "SECURITY_LOG_LEVEL=DEBUG" >> .env
```

2. **Investigate and Remediate**
```bash
# Review audit logs
# Scan for vulnerabilities
# Apply security patches
```

3. **Restore and Monitor**
```bash
# Re-enable services gradually
# Enhanced monitoring
# Security review
```

## Communication Templates

### Initial Incident Report
```
🚨 INCIDENT ALERT - P[X] 

**Summary**: Brief description of the issue
**Impact**: What services/users are affected
**Status**: INVESTIGATING/IDENTIFIED/MONITORING/RESOLVED
**ETA**: Estimated resolution time
**Actions**: What we're doing about it
**Updates**: Next update in 30 minutes
```

### Resolution Notification
```
✅ RESOLVED - P[X] Incident

**Duration**: X hours Y minutes
**Root Cause**: Brief explanation
**Resolution**: What was done to fix it
**Prevention**: Steps taken to prevent recurrence
**Post-Mortem**: Will be published within 48 hours
```

## Post-Incident Activities

### Immediate (Within 2 hours)
- [ ] Confirm full service restoration
- [ ] Update incident channel with resolution
- [ ] Notify all stakeholders
- [ ] Document timeline and actions taken

### Short-term (Within 24 hours)
- [ ] Create detailed incident report
- [ ] Schedule post-mortem meeting
- [ ] Identify immediate improvements
- [ ] Update monitoring/alerting if needed

### Long-term (Within 1 week)
- [ ] Conduct blameless post-mortem
- [ ] Create action items for prevention
- [ ] Update runbooks and procedures
- [ ] Share learnings with team

## Escalation Procedures

### Technical Escalation
1. **On-call engineer** (immediate response)
2. **Senior engineer** (complex technical issues)
3. **Team lead** (architectural decisions)
4. **Principal engineer** (system-wide changes)

### Management Escalation
1. **Team lead** (resource decisions)
2. **Engineering manager** (business impact)
3. **Director** (customer communication)
4. **C-level** (major business impact)

## Tools and Resources

### Monitoring
- Grafana: http://localhost:3000
- Prometheus: http://localhost:9090
- Application logs: `make logs-app`

### Communication
- Incident Slack channel
- Status page updates
- Customer communication templates

### Documentation
- System architecture diagrams
- Network topology maps
- Dependency charts
- Recovery procedures

## Checklists

### P0 Incident Checklist
- [ ] Acknowledge within 5 minutes
- [ ] Create incident channel
- [ ] Notify management immediately
- [ ] Establish war room if needed
- [ ] Update status page
- [ ] Provide hourly updates
- [ ] Document all actions

### Resolution Checklist
- [ ] Verify fix resolves issue
- [ ] Check for side effects
- [ ] Monitor for stability
- [ ] Update documentation
- [ ] Communicate resolution
- [ ] Schedule post-mortem

Remember: Stay calm, communicate clearly, and focus on resolution first, attribution later.