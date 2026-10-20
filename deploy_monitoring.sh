#!/usr/bin/env bash
# Pull Prometheus + Grafana images, load them into minikube, deploy with kubectl.
set -euo pipefail
PROM_IMG="prom/prometheus:v2.54.1"
GRAF_IMG="grafana/grafana:11.2.0"

echo "== 1. pull images on the host"
docker pull "$PROM_IMG"
docker pull "$GRAF_IMG"

echo "== 2. load images into the minikube node"
minikube image load "$PROM_IMG"
minikube image load "$GRAF_IMG"

echo "== 3. deploy (kustomize builds the ConfigMaps from the files in k8s/monitoring)"
kubectl apply -k k8s/monitoring
kubectl -n monitoring rollout status deployment/prometheus --timeout=180s
kubectl -n monitoring rollout status deployment/grafana --timeout=180s
kubectl -n monitoring get pods,svc

cat <<'EOF'

Open them (each in its own terminal; Codespaces auto-forwards the ports):
  kubectl -n heart-risk port-forward svc/heart-api 8080:80
  kubectl -n monitoring port-forward svc/prometheus 9090:9090
  kubectl -n monitoring port-forward svc/grafana 3000:3000
Grafana login: admin / HeartRisk@2026   -> dashboard "Heart Risk API - Service & Model Monitoring"
Generate traffic:  python scripts/generate_traffic.py --url http://localhost:8080 --n 500
EOF
