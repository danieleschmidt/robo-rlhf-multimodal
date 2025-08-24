#!/bin/bash

SERVICE_URL="http://localhost:8000"

echo "Checking service health..."

# Check health endpoint
if curl -f "${SERVICE_URL}/health" > /dev/null 2>&1; then
    echo "✅ Health check passed"
else
    echo "❌ Health check failed"
    exit 1
fi

# Check readiness endpoint
if curl -f "${SERVICE_URL}/ready" > /dev/null 2>&1; then
    echo "✅ Readiness check passed"
else
    echo "❌ Readiness check failed"
    exit 1
fi

# Check metrics endpoint
if curl -f "${SERVICE_URL}:8080/metrics" > /dev/null 2>&1; then
    echo "✅ Metrics endpoint accessible"
else
    echo "❌ Metrics endpoint not accessible"
    exit 1
fi

echo "All health checks passed!"
