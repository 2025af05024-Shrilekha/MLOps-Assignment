#!/usr/bin/env bash
# Usage: scripts/smoke_test.sh [base_url]   (default http://localhost:8000)
# Exits non-zero on any failure so CI fails loudly.
set -euo pipefail
BASE="${1:-http://localhost:8000}"
HERE="$(cd "$(dirname "$0")" && pwd)"

echo "== waiting for $BASE/health"
for i in $(seq 1 30); do
  if curl -fsS "$BASE/health" >/dev/null 2>&1; then break; fi
  sleep 2
  [ "$i" = 30 ] && { echo "API never became healthy"; exit 1; }
done
curl -fsS "$BASE/health"; echo

echo "== POST /predict (high-risk sample)"
OUT=$(curl -fsS -X POST "$BASE/predict" -H 'Content-Type: application/json' -d @"$HERE/sample_request.json")
echo "$OUT"
echo "$OUT" | grep -q '"confidence"' || { echo "no confidence in response"; exit 1; }

echo "== POST /predict with invalid input must return 422"
CODE=$(curl -s -o /dev/null -w '%{http_code}' -X POST "$BASE/predict" \
  -H 'Content-Type: application/json' -d '{"age": -1}')
[ "$CODE" = "422" ] || { echo "expected 422, got $CODE"; exit 1; }

echo "== GET /metrics"
curl -fsS "$BASE/metrics" | grep -E '^heart_api_predictions_total' || { echo "metrics missing"; exit 1; }
echo "SMOKE TEST PASSED"
