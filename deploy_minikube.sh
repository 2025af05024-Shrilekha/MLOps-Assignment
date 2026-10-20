#!/usr/bin/env bash
# Build the API image, load it into minikube, deploy, scale, verify.
# Usage: scripts/deploy_minikube.sh [replicas]   (default 3)
set -euo pipefail
IMAGE="heart-risk-api:v1"
REPLICAS="${1:-3}"

echo "== 1. minikube cluster"
minikube status >/dev/null 2>&1 || minikube start --driver=docker --cpus=2 --memory=4096
kubectl get nodes -o wide

echo "== 2. build image on the host docker"
docker build -t "$IMAGE" .

echo "== 3. load image into the minikube node"
minikube image load "$IMAGE"
minikube image ls | grep heart-risk-api

echo "== 4. apply manifests (namespace, deployment, service, ingress)"
kubectl apply -f k8s/app/namespace.yaml
kubectl apply -f k8s/app/deployment.yaml -f k8s/app/service.yaml
kubectl apply -f k8s/app/ingress.yaml || echo "(ingress skipped - enable with: minikube addons enable ingress)"
kubectl -n heart-risk rollout restart deployment/heart-api   # pick up a re-loaded :v1 image
kubectl -n heart-risk rollout status deployment/heart-api --timeout=180s

echo "== 5. scale to $REPLICAS replicas"
kubectl -n heart-risk scale deployment/heart-api --replicas="$REPLICAS"
kubectl -n heart-risk rollout status deployment/heart-api --timeout=180s
kubectl -n heart-risk get deploy,rs,pods,svc,ingress -o wide

echo "== 6. verify through the service"
kubectl -n heart-risk port-forward svc/heart-api 8080:80 >/tmp/pf-api.log 2>&1 &
PF=$!
trap 'kill $PF 2>/dev/null || true' EXIT
sleep 4
scripts/smoke_test.sh http://localhost:8080
echo "Done. Keep it open with: kubectl -n heart-risk port-forward svc/heart-api 8080:80"
