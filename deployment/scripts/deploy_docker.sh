#!/bin/bash
set -e

echo "Deploying with Docker Compose..."

# Build images
docker-compose build

# Start services
docker-compose up -d

# Wait for health checks
echo "Waiting for services to be healthy..."
sleep 30

# Check status
docker-compose ps

echo "Deployment completed successfully!"
