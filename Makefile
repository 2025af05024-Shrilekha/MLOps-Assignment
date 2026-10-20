IMAGE ?= heart-risk-api:v1
GH_USER ?= $(shell gh api user -q .login 2>/dev/null | tr '[:upper:]' '[:lower:]')
GHCR_IMAGE = ghcr.io/$(GH_USER)/heart-risk-api

.PHONY: install data eda train mlflow-ui lint test api docker-build docker-run docker-test \
        ghcr-push k8s-deploy k8s-scale monitoring traffic all clean

install:      ; pip install -r requirements.txt
data:         ; python -m heartml.data
eda:          ; python -m heartml.eda
train:        ; python -m heartml.train
mlflow-ui:    ; mlflow ui --backend-store-uri ./mlruns --host 0.0.0.0 --port 5000
lint:         ; flake8 .
test:         ; pytest --cov=heartml --cov=api --cov-report=term-missing
api:          ; uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

docker-build: ; docker build -t $(IMAGE) .
docker-run:   ; docker rm -f heart-api 2>/dev/null; docker run -d --name heart-api -p 8000:8000 $(IMAGE)
docker-test: docker-run
	scripts/smoke_test.sh http://localhost:8000

ghcr-push:
	@test -n "$(GH_USER)" || (echo "set GH_USER=<your-github-username>"; exit 1)
	docker tag $(IMAGE) $(GHCR_IMAGE):v1
	docker tag $(IMAGE) $(GHCR_IMAGE):latest
	docker push $(GHCR_IMAGE):v1
	docker push $(GHCR_IMAGE):latest

k8s-deploy:   ; scripts/deploy_minikube.sh 3
k8s-scale:    ; kubectl -n heart-risk scale deployment/heart-api --replicas=$(or $(N),4)
monitoring:   ; scripts/deploy_monitoring.sh
traffic:      ; python scripts/generate_traffic.py --url http://localhost:8080 --n 500

all: data eda train lint test docker-build docker-test

clean:
	rm -rf mlruns reports/runs .pytest_cache .coverage
