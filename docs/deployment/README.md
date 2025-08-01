# Deployment Guide

This directory contains comprehensive deployment documentation and configuration for Robo-RLHF-Multimodal.

## Quick Start

### Local Development
```bash
# Start development environment
make up-dev

# Or using docker-compose directly
docker compose --profile dev up -d
```

### Production Deployment
```bash
# Start production services
make up

# Or using docker-compose directly
docker compose up -d
```

### GPU Training
```bash
# Start GPU training environment
make up-gpu

# Or using docker-compose directly
docker compose --profile gpu up -d
```

## Available Services

### Core Services
- **robo-rlhf**: Main application server (port 8080)
- **mongodb**: Database for storing demonstrations and preferences (port 27017)
- **redis**: Caching and session storage (port 6379)

### Development Services (Profile: dev)
- **robo-rlhf-dev**: Development container with hot reloading
- **jupyter**: Jupyter Lab server (port 8888)
- **tensorboard**: TensorBoard for visualization (port 6006)

### GPU Services (Profile: gpu)
- **robo-rlhf-gpu**: GPU-enabled training container

### Monitoring Services (Profile: monitoring)
- **prometheus**: Metrics collection (port 9090)
- **grafana**: Metrics visualization (port 3000)

## Docker Images

The project uses multi-stage Docker builds:

### Available Stages
1. **base**: Common dependencies and system packages
2. **development**: Full development environment with tools
3. **production**: Minimal runtime environment
4. **gpu-base**: CUDA-enabled base image
5. **gpu-production**: GPU-enabled production environment

### Building Images
```bash
# Build all images
make docker-build

# Build specific stages
make docker-build-prod    # Production only
make docker-build-dev     # Development only
make docker-build-gpu     # GPU only
```

## Environment Configuration

### Required Environment Variables
```bash
# Core configuration
ENVIRONMENT=production|development|testing
CUDA_VISIBLE_DEVICES=0
LOG_LEVEL=INFO

# External services
WANDB_API_KEY=your_wandb_key
HF_TOKEN=your_huggingface_token

# Database
DATABASE_URL=mongodb://admin:password@mongodb:27017/robo_rlhf?authSource=admin
REDIS_URL=redis://redis:6379/0

# Development
DEBUG=0|1
JUPYTER_TOKEN=your_jupyter_token
```

### Configuration Files
- `.env`: Local environment variables (not tracked in git)
- `.env.example`: Template with all available variables
- `configs/redis.conf`: Redis configuration
- `scripts/mongo-init.js`: MongoDB initialization script

## Data Persistence

### Volumes
- `mongodb_data`: Database files
- `redis_data`: Redis persistence
- `prometheus_data`: Metrics data
- `grafana_data`: Dashboard configuration
- `jupyter-data`: Jupyter configuration
- `dev-cache`: Development cache

### Bind Mounts
- `./data`: Training data and datasets
- `./configs`: Configuration files (read-only)
- `./checkpoints`: Model checkpoints
- `./logs`: Application and training logs

## Health Checks

All services include health checks:

### Application Health
```bash
curl http://localhost:8080/health
```

### Database Health
```bash
# MongoDB
docker compose exec mongodb mongosh --eval "db.adminCommand('ping')"

# Redis
docker compose exec redis redis-cli ping
```

### Service Status
```bash
# Check all services
make health-check

# View service status
docker compose ps
```

## Security Considerations

### Container Security
- Non-root user (uid:1000, gid:1000)
- Minimal attack surface in production images
- No unnecessary packages or tools
- Security scanning with bandit

### Network Security
- Isolated Docker network
- No exposed database ports in production
- Health check endpoints only

### Data Security
- Environment variable validation
- Secrets management via external providers
- No hardcoded credentials

## Scaling

### Horizontal Scaling
```bash
# Scale specific services
docker compose up -d --scale robo-rlhf=3

# Load balancer configuration needed for multiple instances
```

### Resource Limits
Configure in docker-compose.override.yml:
```yaml
services:
  robo-rlhf:
    deploy:
      resources:
        limits:
          memory: 4G
          cpus: '2'
        reservations:
          memory: 2G
          cpus: '1'
```

## Monitoring

### Metrics Collection
- Prometheus scrapes application metrics
- Grafana provides visualization dashboards
- Health check monitoring

### Log Aggregation
```bash
# View logs
make logs           # All services
make logs-app       # Main application only
docker compose logs -f service_name
```

### Debugging
```bash
# Access container shell
docker compose exec robo-rlhf bash
docker compose exec robo-rlhf-dev bash

# Debug networking
docker compose exec robo-rlhf ping mongodb
docker compose exec robo-rlhf nslookup redis
```

## Backup and Recovery

### Database Backup
```bash
# Create backup
make db-backup

# Manual backup
docker compose exec mongodb mongodump --out /data/backup
docker compose cp mongodb:/data/backup ./backups/
```

### Model Checkpoint Backup
```bash
# Backup checkpoints to cloud storage
# (Implementation depends on cloud provider)
```

### Recovery
```bash
# Reset database (WARNING: destroys all data)
make db-reset

# Restore from backup
docker compose exec mongodb mongorestore /data/backup
```

## Troubleshooting

### Common Issues

1. **Port conflicts**
   ```bash
   # Check port usage
   netstat -tulpn | grep :8080
   # Change ports in docker-compose.yml
   ```

2. **GPU not available**
   ```bash
   # Check NVIDIA Docker runtime
   docker run --rm --gpus all nvidia/cuda:11.8-base nvidia-smi
   ```

3. **Permission issues**
   ```bash
   # Fix file permissions
   sudo chown -R 1000:1000 data/ logs/ checkpoints/
   ```

4. **Memory issues**
   ```bash
   # Check Docker memory limits
   docker system df
   docker system prune
   ```

### Log Analysis
```bash
# Application logs
docker compose logs robo-rlhf | grep ERROR

# Database logs  
docker compose logs mongodb | tail -100

# System resource usage
docker stats
```

## Production Checklist

- [ ] Environment variables configured
- [ ] SSL certificates installed (if needed)
- [ ] Database backups scheduled
- [ ] Monitoring alerts configured
- [ ] Resource limits set
- [ ] Security scanning completed
- [ ] Load testing performed
- [ ] Disaster recovery plan tested

## Support

For deployment issues:
1. Check logs with `make logs`
2. Verify health checks with `make health-check`
3. Review configuration files
4. Consult troubleshooting section
5. Open an issue with deployment details