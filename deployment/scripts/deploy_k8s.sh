#!/bin/bash
set -e

echo "Deploying Robo-RLHF-Multimodal to Kubernetes..."

# Create namespace
kubectl create namespace robo-rlhf --dry-run=client -o yaml | kubectl apply -f -

# Apply ConfigMap
kubectl apply -f configmap.yaml

# Apply PVC
kubectl apply -f pvc.yaml

# Apply ServiceAccount
kubectl apply -f serviceaccount.yaml

# Apply Service
kubectl apply -f service.yaml

# Apply Deployment
kubectl apply -f deployment.yaml

# Apply HPA
kubectl apply -f hpa.yaml

# Apply Ingress
kubectl apply -f ingress.yaml

# Wait for deployment to be ready
kubectl wait --for=condition=available --timeout=300s deployment/robo-rlhf-multimodal -n robo-rlhf

echo "Deployment completed successfully!"

# Show status
kubectl get all -n robo-rlhf
