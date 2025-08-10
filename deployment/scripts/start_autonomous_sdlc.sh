#!/bin/bash
# Production startup script for Autonomous SDLC Executor

set -euo pipefail

# Configuration
ROBO_RLHF_ENV="${ROBO_RLHF_ENV:-production}"
ROBO_RLHF_LOG_LEVEL="${ROBO_RLHF_LOG_LEVEL:-INFO}"
ROBO_RLHF_LOG_DIR="${ROBO_RLHF_LOG_DIR:-/app/logs}"
ROBO_RLHF_DATA_DIR="${ROBO_RLHF_DATA_DIR:-/app/data}"
HEALTH_CHECK_PORT="${HEALTH_CHECK_PORT:-8080}"
API_PORT="${API_PORT:-8081}"

# Logging function
log() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] [STARTUP] $*" | tee -a "${ROBO_RLHF_LOG_DIR}/startup.log"
}

# Error handling
error_exit() {
    log "ERROR: $1" >&2
    exit 1
}

# Signal handling for graceful shutdown
shutdown_handler() {
    log "Received shutdown signal, gracefully stopping autonomous SDLC executor..."
    
    if [[ -n "${AUTONOMOUS_PID:-}" ]]; then
        kill -TERM "$AUTONOMOUS_PID" 2>/dev/null || true
        wait "$AUTONOMOUS_PID" 2>/dev/null || true
    fi
    
    log "Autonomous SDLC executor stopped gracefully"
    exit 0
}

# Set up signal traps
trap shutdown_handler SIGTERM SIGINT SIGQUIT

# Startup banner
log "🚀 Starting Robo-RLHF Autonomous SDLC Executor"
log "   Environment: $ROBO_RLHF_ENV"
log "   Log Level: $ROBO_RLHF_LOG_LEVEL"
log "   Log Directory: $ROBO_RLHF_LOG_DIR"
log "   Data Directory: $ROBO_RLHF_DATA_DIR"
log "   Health Check Port: $HEALTH_CHECK_PORT"
log "   API Port: $API_PORT"

# Pre-flight checks
log "Running pre-flight checks..."

# Check Python environment
python3 --version || error_exit "Python not available"
log "✅ Python environment check passed"

# Check required directories
for dir in "$ROBO_RLHF_LOG_DIR" "$ROBO_RLHF_DATA_DIR" "${ROBO_RLHF_CACHE_DIR:-/app/cache}"; do
    if [[ ! -d "$dir" ]]; then
        mkdir -p "$dir" || error_exit "Cannot create directory: $dir"
    fi
    if [[ ! -w "$dir" ]]; then
        error_exit "Directory not writable: $dir"
    fi
done
log "✅ Directory permissions check passed"

# Check database connectivity (if configured)
if [[ -n "${POSTGRES_HOST:-}" ]]; then
    log "Checking database connectivity..."
    timeout 30 bash -c "until nc -z ${POSTGRES_HOST} ${POSTGRES_PORT:-5432}; do sleep 1; done" || error_exit "Cannot connect to PostgreSQL"
    log "✅ Database connectivity check passed"
fi

# Check Redis connectivity (if configured)
if [[ -n "${REDIS_HOST:-}" ]]; then
    log "Checking Redis connectivity..."
    timeout 30 bash -c "until nc -z ${REDIS_HOST} ${REDIS_PORT:-6379}; do sleep 1; done" || error_exit "Cannot connect to Redis"
    log "✅ Redis connectivity check passed"
fi

# Check Docker availability (if Docker-in-Docker is enabled)
if [[ -S /var/run/docker.sock ]]; then
    log "Checking Docker connectivity..."
    docker version >/dev/null 2>&1 || log "⚠️  Docker not available - some SDLC actions may fail"
    log "✅ Docker connectivity check completed"
fi

# Initialize application
log "Initializing autonomous SDLC executor..."

# Set up Python path
export PYTHONPATH="/app:${PYTHONPATH:-}"

# Create configuration file if it doesn't exist
CONFIG_FILE="${ROBO_RLHF_DATA_DIR}/autonomous_config.yaml"
if [[ ! -f "$CONFIG_FILE" ]]; then
    log "Creating default configuration..."
    cat > "$CONFIG_FILE" << EOF
environment: ${ROBO_RLHF_ENV}
debug: false

autonomous:
  max_parallel: 4
  quality_threshold: 0.85
  optimization_frequency: 5
  auto_rollback: true

security:
  enable_input_validation: true
  max_commands_per_minute: 50
  max_command_timeout: 1800
  allowed_commands:
    - python
    - python3
    - pytest
    - mypy
    - bandit
    - docker
    - npm
    - pip

optimization:
  enable_caching: true
  cache_size: 2000
  cache_ttl: 7200
  enable_parallel: true
  auto_scale: true
  max_workers: 8

monitoring:
  enable_metrics: true
  metrics_port: ${HEALTH_CHECK_PORT}
  api_port: ${API_PORT}
  log_level: ${ROBO_RLHF_LOG_LEVEL}

data_collection:
  output_dir: ${ROBO_RLHF_DATA_DIR}
  
logging:
  level: ${ROBO_RLHF_LOG_LEVEL}
  file: ${ROBO_RLHF_LOG_DIR}/autonomous_sdlc.log
  structured: true
  
database:
  host: ${POSTGRES_HOST:-localhost}
  port: ${POSTGRES_PORT:-5432}
  database: ${POSTGRES_DB:-autonomous_sdlc}
  user: ${POSTGRES_USER:-sdlc_user}
  password: ${POSTGRES_PASSWORD:-}

redis:
  host: ${REDIS_HOST:-localhost}
  port: ${REDIS_PORT:-6379}
  password: ${REDIS_PASSWORD:-}
EOF
    log "✅ Configuration file created: $CONFIG_FILE"
fi

# Start health check server in background
log "Starting health check server on port $HEALTH_CHECK_PORT..."
python3 -c "
import http.server
import socketserver
import json
import threading
import time
from datetime import datetime

class HealthCheckHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/health':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            health_data = {
                'status': 'healthy',
                'timestamp': datetime.utcnow().isoformat(),
                'environment': '${ROBO_RLHF_ENV}',
                'version': '1.0.0',
                'uptime': time.time() - start_time
            }
            self.wfile.write(json.dumps(health_data).encode())
        elif self.path == '/metrics':
            self.send_response(200)
            self.send_header('Content-type', 'text/plain')
            self.end_headers()
            # Basic Prometheus metrics
            metrics = '''
# HELP autonomous_sdlc_uptime_seconds Total uptime of the autonomous SDLC executor
# TYPE autonomous_sdlc_uptime_seconds counter
autonomous_sdlc_uptime_seconds {}

# HELP autonomous_sdlc_status Status of the autonomous SDLC executor
# TYPE autonomous_sdlc_status gauge
autonomous_sdlc_status 1
'''.format(time.time() - start_time)
            self.wfile.write(metrics.encode())
        else:
            self.send_error(404)
    
    def log_message(self, format, *args):
        pass  # Suppress access logs

start_time = time.time()
httpd = socketserver.TCPServer(('', $HEALTH_CHECK_PORT), HealthCheckHandler)
print(f'Health check server started on port $HEALTH_CHECK_PORT')
httpd.serve_forever()
" &
HEALTH_CHECK_PID=$!
log "✅ Health check server started (PID: $HEALTH_CHECK_PID)"

# Wait for health check server to be ready
sleep 2

# Start main autonomous SDLC application
log "Starting main autonomous SDLC application..."

# Run the autonomous SDLC executor
python3 -c "
import asyncio
import sys
import signal
import os
from pathlib import Path

# Add app to Python path
sys.path.insert(0, '/app')

try:
    from robo_rlhf.quantum import AutonomousSDLCExecutor
    from robo_rlhf.quantum.autonomous import SDLCPhase
    from robo_rlhf.core import setup_logging, get_config
    
    async def main():
        print('[AUTONOMOUS] 🤖 Initializing Autonomous SDLC Executor...')
        
        # Setup logging
        setup_logging(
            level='${ROBO_RLHF_LOG_LEVEL}',
            log_file='${ROBO_RLHF_LOG_DIR}/autonomous_sdlc.log',
            structured=True,
            console=True
        )
        
        # Initialize executor
        config_file = Path('${CONFIG_FILE}')
        project_path = Path('/app')
        
        # Load configuration if available
        config = {}
        if config_file.exists():
            import yaml
            with open(config_file) as f:
                config = yaml.safe_load(f)
        
        executor = AutonomousSDLCExecutor(project_path, config)
        
        print('[AUTONOMOUS] ✅ Autonomous SDLC Executor initialized successfully')
        print('[AUTONOMOUS] 🚀 Ready to execute autonomous SDLC operations')
        print('[AUTONOMOUS] 📊 Health check: http://localhost:${HEALTH_CHECK_PORT}/health')
        print('[AUTONOMOUS] 📈 Metrics: http://localhost:${HEALTH_CHECK_PORT}/metrics')
        
        # Main execution loop
        while True:
            try:
                # Run continuous autonomous operations
                target_phases = [
                    SDLCPhase.MONITORING,
                    SDLCPhase.OPTIMIZATION
                ]
                
                print('[AUTONOMOUS] 🔄 Executing autonomous SDLC cycle...')
                results = await executor.execute_autonomous_sdlc(target_phases)
                
                print(f'[AUTONOMOUS] ✅ SDLC cycle completed: {results.get(\"overall_success\", False)}')
                
                # Wait before next cycle
                await asyncio.sleep(300)  # 5 minutes between cycles
                
            except KeyboardInterrupt:
                print('[AUTONOMOUS] 🛑 Received interrupt signal, shutting down...')
                break
            except Exception as e:
                print(f'[AUTONOMOUS] ❌ Error in autonomous execution: {e}')
                await asyncio.sleep(60)  # Wait 1 minute before retry
    
    if __name__ == '__main__':
        asyncio.run(main())

except ImportError as e:
    print(f'[AUTONOMOUS] ❌ Import error: {e}')
    print('[AUTONOMOUS] ⚠️  Running in minimal mode - some features may not be available')
    print('[AUTONOMOUS] 🔄 Entering health check only mode...')
    
    # Minimal health check mode
    import time
    while True:
        print(f'[AUTONOMOUS] 💓 Autonomous SDLC health check - {time.strftime(\"%Y-%m-%d %H:%M:%S\")}')
        time.sleep(60)

except Exception as e:
    print(f'[AUTONOMOUS] ❌ Fatal error: {e}')
    sys.exit(1)
" &
AUTONOMOUS_PID=$!

log "✅ Autonomous SDLC executor started (PID: $AUTONOMOUS_PID)"

# Wait for processes
log "🚀 Autonomous SDLC Executor is running"
log "   Health check: http://localhost:$HEALTH_CHECK_PORT/health"
log "   Metrics: http://localhost:$HEALTH_CHECK_PORT/metrics"
log "   Main process PID: $AUTONOMOUS_PID"

# Keep the script running and wait for signals
wait $AUTONOMOUS_PID