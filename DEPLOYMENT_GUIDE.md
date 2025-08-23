# Deployment Guide - Robo-RLHF-Multimodal

## Overview

Complete production deployment guide for Robo-RLHF-Multimodal with autonomous SDLC capabilities.

## Prerequisites

### System Requirements
- **CPU**: 8+ cores (Intel/AMD x86_64 or Apple Silicon)
- **RAM**: 16GB+ (32GB recommended for production)
- **Storage**: 100GB+ SSD
- **Network**: High bandwidth for model downloads

### Software Dependencies
- **Python**: 3.8+ (3.11 recommended)
- **Docker**: 20.10+
- **Kubernetes**: 1.24+ (optional, for container orchestration)
- **Git**: Latest version

### Cloud Provider Requirements
- **AWS**: EKS cluster, S3 bucket, RDS instance
- **GCP**: GKE cluster, Cloud Storage, Cloud SQL
- **Azure**: AKS cluster, Blob Storage, Azure Database

## Quick Start Deployment

### 1. Local Development Setup
```bash
# Clone repository
git clone https://github.com/danieleschmidt/robo-rlhf-multimodal
cd robo-rlhf-multimodal

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate  # Windows

# Install dependencies
pip install -e ".[full]"

# Run autonomous SDLC validation
python comprehensive_quality_gates.py

# Start local development server
uvicorn robo_rlhf.web.app:app --reload --port 8000
```

### 2. Docker Deployment
```bash
# Build production image
docker build -t robo-rlhf:latest .

# Run with Docker Compose
docker-compose -f docker-compose-production.yml up -d

# Verify deployment
docker-compose ps
curl http://localhost:8000/v1/health
```

### 3. Kubernetes Deployment
```bash
# Apply production configuration
kubectl apply -f k8s-production.yaml

# Check deployment status
kubectl get pods -l app=robo-rlhf-multimodal
kubectl get services

# Scale deployment
kubectl scale deployment robo-rlhf-multimodal --replicas=5
```

## Environment-Specific Configurations

### Development Environment
```yaml
# config/development.yaml
environment: development
debug: true
log_level: DEBUG

database:
  url: sqlite:///./dev.db
  
redis:
  url: redis://localhost:6379/0

scaling:
  min_replicas: 1
  max_replicas: 3
```

### Staging Environment
```yaml
# config/staging.yaml
environment: staging
debug: false
log_level: INFO

database:
  url: postgresql://user:pass@staging-db:5432/robo_rlhf
  
redis:
  url: redis://staging-redis:6379/0
  
scaling:
  min_replicas: 2
  max_replicas: 10
```

### Production Environment
```yaml
# config/production.yaml
environment: production
debug: false
log_level: INFO

database:
  url: postgresql://user:pass@prod-db:5432/robo_rlhf
  pool_size: 20
  max_overflow: 30

redis:
  url: redis://prod-redis:6379/0
  sentinel_mode: true

scaling:
  min_replicas: 5
  max_replicas: 50
  target_cpu: 70
  target_memory: 80
```

## Infrastructure as Code

### Terraform Configuration
```hcl
# terraform/main.tf
provider "aws" {
  region = var.aws_region
}

module "vpc" {
  source = "./modules/vpc"
  
  cidr_block = "10.0.0.0/16"
  availability_zones = ["us-east-1a", "us-east-1b", "us-east-1c"]
}

module "eks" {
  source = "./modules/eks"
  
  cluster_name = "robo-rlhf-cluster"
  vpc_id = module.vpc.vpc_id
  subnet_ids = module.vpc.private_subnet_ids
  
  node_groups = {
    main = {
      instance_types = ["m5.xlarge", "m5.2xlarge"]
      min_size = 3
      max_size = 20
      desired_size = 5
    }
  }
}

module "rds" {
  source = "./modules/rds"
  
  identifier = "robo-rlhf-db"
  engine = "postgres"
  engine_version = "14.9"
  instance_class = "db.r6g.xlarge"
  allocated_storage = 100
  
  vpc_id = module.vpc.vpc_id
  subnet_ids = module.vpc.database_subnet_ids
}
```

### Deploy with Terraform
```bash
cd terraform/

# Initialize Terraform
terraform init

# Plan deployment
terraform plan -var-file="production.tfvars"

# Apply configuration
terraform apply -var-file="production.tfvars"
```

## CI/CD Pipeline Setup

### GitHub Actions Configuration
```yaml
# .github/workflows/deploy.yml
name: Deploy to Production

on:
  push:
    branches: [main]
  
jobs:
  autonomous-sdlc:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      
      - name: Install dependencies
        run: |
          pip install -e ".[dev]"
      
      - name: Run Autonomous SDLC - Generation 1
        run: python generation1_minimal_test.py
      
      - name: Run Autonomous SDLC - Generation 2
        run: python generation2_robust_implementation.py
      
      - name: Run Autonomous SDLC - Generation 3
        run: python generation3_scalable_implementation.py
      
      - name: Run Quality Gates
        run: python comprehensive_quality_gates.py
      
      - name: Test Global Features
        run: python global_first_simplified.py
  
  build-and-deploy:
    needs: autonomous-sdlc
    runs-on: ubuntu-latest
    steps:
      - name: Build Docker image
        run: |
          docker build -t $ECR_REGISTRY/$ECR_REPOSITORY:$GITHUB_SHA .
          docker push $ECR_REGISTRY/$ECR_REPOSITORY:$GITHUB_SHA
      
      - name: Deploy to EKS
        run: |
          aws eks update-kubeconfig --name robo-rlhf-cluster
          kubectl set image deployment/robo-rlhf-multimodal app=$ECR_REGISTRY/$ECR_REPOSITORY:$GITHUB_SHA
```

## Monitoring and Observability

### Prometheus Configuration
```yaml
# monitoring/prometheus.yml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'robo-rlhf'
    static_configs:
      - targets: ['robo-rlhf-service:8000']
    metrics_path: '/metrics'
    scrape_interval: 30s
```

### Grafana Dashboards
```json
{
  "dashboard": {
    "title": "Robo-RLHF Production Metrics",
    "panels": [
      {
        "title": "Autonomous SDLC Status",
        "type": "stat",
        "targets": [
          {
            "expr": "autonomous_sdlc_overall_score"
          }
        ]
      },
      {
        "title": "Training Jobs",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(training_jobs_total[5m])"
          }
        ]
      }
    ]
  }
}
```

## Security Configuration

### SSL/TLS Setup
```yaml
# k8s/ingress.yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: robo-rlhf-ingress
  annotations:
    cert-manager.io/cluster-issuer: "letsencrypt-prod"
    nginx.ingress.kubernetes.io/ssl-redirect: "true"
spec:
  tls:
    - hosts:
        - api.robo-rlhf.ai
      secretName: robo-rlhf-tls
  rules:
    - host: api.robo-rlhf.ai
      http:
        paths:
          - path: /
            pathType: Prefix
            backend:
              service:
                name: robo-rlhf-service
                port:
                  number: 8000
```

### Environment Variables
```bash
# Production environment variables
export DATABASE_URL="postgresql://user:pass@prod-db/robo_rlhf"
export REDIS_URL="redis://prod-redis:6379/0"
export SECRET_KEY="your-super-secret-key-here"
export JWT_SECRET="your-jwt-secret-here"
export AUTONOMOUS_SDLC_ENABLED="true"
export QUALITY_GATES_THRESHOLD="85"
export GLOBAL_COMPLIANCE="gdpr,ccpa,pdpa"
```

## Health Checks and Monitoring

### Kubernetes Health Checks
```yaml
# k8s/deployment.yaml
spec:
  template:
    spec:
      containers:
        - name: robo-rlhf
          livenessProbe:
            httpGet:
              path: /health
              port: 8000
            initialDelaySeconds: 30
            periodSeconds: 10
          readinessProbe:
            httpGet:
              path: /health/ready
              port: 8000
            initialDelaySeconds: 5
            periodSeconds: 5
```

### Custom Metrics
```python
# robo_rlhf/monitoring.py
from prometheus_client import Counter, Histogram, Gauge

# Autonomous SDLC metrics
autonomous_sdlc_executions = Counter(
    'autonomous_sdlc_executions_total',
    'Total SDLC executions',
    ['phase', 'status']
)

quality_gates_score = Gauge(
    'quality_gates_overall_score',
    'Overall quality gates score'
)

training_duration = Histogram(
    'training_job_duration_seconds',
    'Training job duration in seconds'
)
```

## Troubleshooting

### Common Issues

#### 1. Pod Startup Failures
```bash
# Check pod logs
kubectl logs -l app=robo-rlhf-multimodal --tail=100

# Check pod descriptions
kubectl describe pod <pod-name>

# Common fixes
kubectl delete pod <pod-name>  # Force restart
kubectl scale deployment robo-rlhf-multimodal --replicas=0
kubectl scale deployment robo-rlhf-multimodal --replicas=3
```

#### 2. Database Connection Issues
```bash
# Test database connectivity
kubectl exec -it <pod-name> -- python -c "
from robo_rlhf.database import test_connection
test_connection()
"

# Check database pod
kubectl get pods -l app=postgresql
kubectl logs <postgresql-pod-name>
```

#### 3. Quality Gates Failures
```bash
# Run quality gates manually
kubectl exec -it <pod-name> -- python comprehensive_quality_gates.py

# Check specific gate failures
kubectl logs <pod-name> | grep "Quality gate"
```

### Performance Optimization

#### Database Optimization
```sql
-- Add indexes for common queries
CREATE INDEX idx_training_jobs_status ON training_jobs(status);
CREATE INDEX idx_preferences_created_at ON preferences(created_at);

-- Analyze query performance
EXPLAIN ANALYZE SELECT * FROM training_jobs WHERE status = 'running';
```

#### Application Tuning
```yaml
# k8s/deployment.yaml - Resource limits
resources:
  requests:
    cpu: 1000m
    memory: 2Gi
  limits:
    cpu: 4000m
    memory: 8Gi

# Environment variables for tuning
env:
  - name: WORKERS
    value: "4"
  - name: MAX_REQUESTS
    value: "1000"
  - name: MAX_REQUESTS_JITTER
    value: "100"
```

## Backup and Disaster Recovery

### Database Backups
```bash
# Automated backup script
#!/bin/bash
pg_dump $DATABASE_URL | gzip > "backup_$(date +%Y%m%d_%H%M%S).sql.gz"
aws s3 cp backup_*.sql.gz s3://robo-rlhf-backups/
```

### Configuration Backups
```bash
# Backup Kubernetes configurations
kubectl get all -o yaml > k8s-backup.yaml
kubectl get configmaps,secrets -o yaml > k8s-config-backup.yaml
```

## Scaling Guidelines

### Horizontal Scaling
```bash
# Scale based on CPU usage
kubectl autoscale deployment robo-rlhf-multimodal --cpu-percent=70 --min=3 --max=50

# Scale based on custom metrics
kubectl apply -f hpa-custom-metrics.yaml
```

### Vertical Scaling
```yaml
# Vertical Pod Autoscaler
apiVersion: autoscaling.k8s.io/v1
kind: VerticalPodAutoscaler
metadata:
  name: robo-rlhf-vpa
spec:
  targetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: robo-rlhf-multimodal
  updatePolicy:
    updateMode: "Auto"
```

## Support and Maintenance

### Regular Maintenance Tasks
1. **Weekly**: Review logs and metrics
2. **Monthly**: Update dependencies and security patches
3. **Quarterly**: Capacity planning and cost optimization
4. **Annually**: Architecture review and compliance audit

### Emergency Contacts
- **On-call Engineer**: +1-555-ROBO-RLHF
- **DevOps Team**: devops@robo-rlhf.ai  
- **Security Team**: security@robo-rlhf.ai

---

**Last Updated**: January 23, 2025  
**Version**: 1.0.0
