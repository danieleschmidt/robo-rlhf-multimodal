# Monitoring & Observability

This document covers the comprehensive monitoring and observability setup for Robo-RLHF-Multimodal.

## Overview

The monitoring stack includes:
- **Prometheus**: Metrics collection and storage
- **Grafana**: Metrics visualization and dashboards
- **Application metrics**: Custom ML training and inference metrics
- **Health checks**: Service availability monitoring
- **Logging**: Structured application and container logs

## Quick Start

### Start Monitoring Stack
```bash
# Start with monitoring profile
make up-monitoring

# Or using docker-compose
docker compose --profile monitoring up -d
```

### Access Dashboards
- **Grafana**: http://localhost:3000 (admin/admin)
- **Prometheus**: http://localhost:9090
- **Application**: http://localhost:8080

## Metrics Categories

### System Metrics
- **CPU Usage**: Process and container CPU utilization
- **Memory Usage**: RAM and GPU memory consumption
- **Disk I/O**: Read/write operations and throughput
- **Network**: Request rates and bandwidth usage

### Application Metrics
- **HTTP Metrics**: Request rate, response time, error rate
- **Database Metrics**: Query performance, connection pool usage
- **Cache Metrics**: Hit rate, eviction rate, memory usage
- **Custom Business Metrics**: User-defined application KPIs

### ML Training Metrics
- **Training Loss**: Loss values over training steps/epochs
- **Validation Metrics**: Accuracy, precision, recall, F1-score
- **Learning Rate**: Current learning rate and scheduling
- **Batch Processing**: Batch size, processing time, throughput
- **Model Performance**: Inference latency, prediction accuracy

### Infrastructure Metrics
- **Container Metrics**: CPU, memory, network per container
- **GPU Metrics**: Utilization, memory, temperature, power
- **Storage Metrics**: Disk usage, I/O wait times
- **Service Health**: Uptime, response times, error rates

## Custom Metrics Implementation

### Adding Application Metrics

```python
from prometheus_client import Counter, Histogram, Gauge, start_http_server

# Define metrics
REQUEST_COUNT = Counter('http_requests_total', 'Total HTTP requests', ['method', 'endpoint', 'status'])
REQUEST_LATENCY = Histogram('http_request_duration_seconds', 'HTTP request latency')
ACTIVE_USERS = Gauge('active_users_total', 'Number of active users')
TRAINING_LOSS = Gauge('training_loss', 'Current training loss')

# Instrument your code
@REQUEST_LATENCY.time()
def handle_request():
    REQUEST_COUNT.labels(method='GET', endpoint='/api/train', status='200').inc()
    # Your application logic here
    
def update_training_metrics(loss_value, accuracy):
    TRAINING_LOSS.set(loss_value)
    TRAINING_ACCURACY.set(accuracy)

# Start metrics server
start_http_server(8001)
```

### Custom Metrics Endpoint
```python
from fastapi import FastAPI
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST

app = FastAPI()

@app.get("/metrics")
async def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
```

## Dashboards

### Pre-built Dashboards
1. **System Overview**: High-level system health and performance
2. **Application Performance**: HTTP metrics, database, cache performance
3. **ML Training**: Training metrics, GPU utilization, model performance
4. **Infrastructure**: Container metrics, resource utilization
5. **Business Metrics**: User activity, feature usage, success rates

### Creating Custom Dashboards

1. **Access Grafana**: http://localhost:3000
2. **Login**: admin/admin (change on first login)
3. **Create Dashboard**: Click "+" → Dashboard
4. **Add Panel**: Configure queries and visualizations
5. **Save**: Give meaningful name and tags

### Dashboard Best Practices
- Use meaningful panel titles and descriptions
- Set appropriate time ranges and refresh intervals
- Use variables for dynamic filtering
- Include threshold alerts where appropriate
- Organize panels logically (overview → details)

## Alerting

### Prometheus Alert Rules
```yaml
# configs/alert_rules.yml
groups:
  - name: robo-rlhf-alerts
    rules:
      - alert: ApplicationDown
        expr: up{job="robo-rlhf-app"} == 0
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "Application is down"
          description: "Application has been down for more than 1 minute"
      
      - alert: HighErrorRate
        expr: rate(http_requests_total{status=~"5.."}[5m]) > 0.1
        for: 2m
        labels:
          severity: warning
        annotations:
          summary: "High error rate detected"
          description: "Error rate is {{ $value }} requests/sec"
      
      - alert: TrainingStalled
        expr: increase(training_steps_total[10m]) == 0
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "Training appears to be stalled"
          description: "No training progress in the last 10 minutes"
```

### Grafana Alerting
1. **Create Alert Rule**: Dashboard → Panel → Alert tab
2. **Set Conditions**: Define thresholds and evaluation frequency
3. **Configure Notifications**: Email, Slack, webhook integrations
4. **Test Alerts**: Verify alert firing and recovery

## Health Checks

### Application Health Endpoints
```python
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "0.1.0",
        "checks": {
            "database": await check_database(),
            "cache": await check_redis(),
            "gpu": check_gpu_availability()
        }
    }

@app.get("/ready")
async def readiness_check():
    # Check if app is ready to handle requests
    return {"status": "ready"}

@app.get("/live")
async def liveness_check():
    # Basic liveness check
    return {"status": "alive"}
```

### Service Health Monitoring
```bash
# Manual health checks
curl http://localhost:8080/health
curl http://localhost:8080/ready
curl http://localhost:8080/live

# Automated monitoring
make health-check
```

## Logging

### Structured Logging Setup
```python
import logging
import json
from datetime import datetime

class StructuredLogger:
    def __init__(self, name):
        self.logger = logging.getLogger(name)
        handler = logging.StreamHandler()
        handler.setFormatter(self._get_formatter())
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)
    
    def _get_formatter(self):
        def format_record(record):
            return json.dumps({
                "timestamp": datetime.utcnow().isoformat(),
                "level": record.levelname,
                "logger": record.name,
                "message": record.getMessage(),
                "module": record.module,
                "function": record.funcName,
                "line": record.lineno,
            })
        
        formatter = logging.Formatter()
        formatter.format = format_record
        return formatter
    
    def info(self, message, **kwargs):
        self.logger.info(message, extra=kwargs)
    
    def error(self, message, **kwargs):
        self.logger.error(message, extra=kwargs)
```

### Log Aggregation
```bash
# View logs from all services
docker compose logs -f

# View specific service logs
docker compose logs -f robo-rlhf
docker compose logs -f mongodb

# Follow logs with grep
docker compose logs -f | grep ERROR
```

## Performance Monitoring

### Training Performance
```python
import time
from contextlib import contextmanager

@contextmanager
def measure_time(metric_name):
    start_time = time.perf_counter()
    try:
        yield
    finally:
        duration = time.perf_counter() - start_time
        TRAINING_TIME_HISTOGRAM.labels(operation=metric_name).observe(duration)

# Usage
with measure_time("forward_pass"):
    output = model(batch)

with measure_time("backward_pass"):
    loss.backward()
```

### Resource Monitoring
```python
import psutil
import torch

def collect_system_metrics():
    # CPU and memory
    cpu_percent = psutil.cpu_percent()
    memory = psutil.virtual_memory()
    
    # GPU metrics (if available)
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.memory_allocated() / 1024**3  # GB
        gpu_utilization = torch.cuda.utilization()
    
    # Update Prometheus metrics
    CPU_USAGE.set(cpu_percent)
    MEMORY_USAGE.set(memory.percent)
    GPU_MEMORY_USAGE.set(gpu_memory)
```

## Troubleshooting

### Common Monitoring Issues

1. **Metrics Not Appearing**
   - Check Prometheus targets: http://localhost:9090/targets
   - Verify application metrics endpoint: http://localhost:8080/metrics
   - Check firewall and network connectivity

2. **Grafana Dashboard Issues**
   - Verify Prometheus data source configuration
   - Check query syntax and metric names
   - Ensure time ranges are appropriate

3. **High Resource Usage**
   - Adjust scrape intervals in prometheus.yml
   - Reduce metric retention period
   - Optimize query efficiency

4. **Missing GPU Metrics**
   - Install nvidia-ml-py for GPU monitoring
   - Check CUDA availability
   - Verify Docker GPU runtime configuration

### Debug Commands
```bash
# Check Prometheus configuration
docker compose exec prometheus promtool check config /etc/prometheus/prometheus.yml

# Test metric queries
curl http://localhost:9090/api/v1/query?query=up

# Check Grafana logs
docker compose logs grafana

# Monitor resource usage
docker stats
```

## Best Practices

### Metric Design
- Use consistent naming conventions
- Include relevant labels for filtering
- Avoid high-cardinality labels
- Set appropriate metric types (Counter, Gauge, Histogram)

### Dashboard Design
- Start with overview, drill down to details
- Use consistent color schemes and units
- Include annotations for deployments and incidents
- Set up alerts for critical metrics

### Performance
- Monitor monitoring system resource usage
- Use recording rules for expensive queries
- Implement metric retention policies
- Regular cleanup of old data

### Security
- Secure Grafana with proper authentication
- Use HTTPS for external access
- Implement network segmentation
- Regular security updates

## Integration with External Systems

### Alertmanager Integration
```yaml
# docker-compose.yml addition
alertmanager:
  image: prom/alertmanager:latest
  ports:
    - "9093:9093"
  volumes:
    - ./configs/alertmanager.yml:/etc/alertmanager/alertmanager.yml
```

### External Storage
```yaml
# prometheus.yml addition
remote_write:
  - url: "https://prometheus-remote-write-endpoint/api/v1/write"
    headers:
      Authorization: "Bearer YOUR_TOKEN"
```

### Log Shipping
```yaml
# Add to docker-compose.yml
filebeat:
  image: elastic/filebeat:7.17.0
  volumes:
    - ./logs:/usr/share/filebeat/logs:ro
    - ./filebeat.yml:/usr/share/filebeat/filebeat.yml:ro
```

This monitoring setup provides comprehensive observability for the Robo-RLHF-Multimodal system, enabling proactive monitoring, performance optimization, and rapid issue resolution.