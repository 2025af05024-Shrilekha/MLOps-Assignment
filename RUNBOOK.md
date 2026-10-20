# RUNBOOK - building this project in GitHub Codespaces, stage by stage

Each stage ends with a **commit + push**, so your Git history shows the project growing step by step.
Take the screenshot listed at the end of each stage and save it into `screenshots/` (you need them for the report).

Replace `<you>` with your GitHub username, in **lowercase**.

---

## Stage 0 - Repo and Codespace (one-time)

1. On GitHub, create a new repository called `heart-risk-mlops`. Make it **public** (simplest for GHCR and grading), and **don't** add a README.
2. On the repo page, go to **Code -> Codespaces -> "..." -> New with options**, pick **4-core / 16 GB**, and click **Create codespace**.
   Minikube plus Prometheus plus Grafana need the 4-core machine. The default 2-core one is too small.
3. Drag `heart-risk-mlops.zip` from your computer into the Codespace **Explorer** panel, then run this in the terminal:

```bash
unzip -o heart-risk-mlops.zip -d . && rm heart-risk-mlops.zip
git config user.name  "Shrilekha Sivakumar"
git config user.email "<your-github-email>"
git branch -M main
```

All the files are now on disk. You'll **add them in stages** below, not all at once.

## Stage 1 - Dev environment and project scaffold

```bash
git add .devcontainer requirements.txt requirements-serve.txt .gitignore .flake8 pytest.ini Makefile README.md RUNBOOK.md heartml/__init__.py heartml/config.py screenshots/.gitkeep
git commit -m "stage 1: project scaffold, devcontainer (python 3.11, docker-in-docker, minikube), pinned requirements"
git push -u origin main
```

Next, open the Command Palette (`Ctrl/Cmd+Shift+P`), run **Codespaces: Rebuild Container**, and wait about 3-5 minutes.
After the rebuild, check that everything is there:

```bash
python --version && docker --version && minikube version && kubectl version --client
```

📸 `01_codespace_tools.png`

## Stage 2 - Data acquisition and EDA

```bash
make data      # python -m heartml.data -> data/raw + data/processed/heart_clean.csv
make eda       # python -m heartml.eda  -> reports/figures/*.png, reports/eda_summary.json
```

You should see `Rows=303 ... Positive rate=0.459` and missing values `ca 4, thal 2`.
Open the PNGs in `reports/figures/`. Also try the notebook `notebooks/01_eda_and_training.ipynb`
(choose the Python 3.11 kernel when VS Code asks).

```bash
git add heartml/data.py heartml/eda.py data/ reports/figures reports/eda_summary.json reports/eda_describe.csv
git commit -m "stage 2: UCI download script with fallback, cleaning, EDA visualisations"
git push
```

📸 `02_eda_heatmap.png`, `02_class_balance.png`

## Stage 3 - Train two models, track with MLflow, save best with joblib

```bash
make train     # ~1 min: GridSearchCV for LogisticRegression + RandomForest, 5-fold CV
cat models/model_meta.json
make mlflow-ui # Codespaces pops up "Open in Browser" for port 5000
```

In the MLflow UI, open the experiment **heart-risk-classifier**. Then:

- Tick both runs and click **Compare**. This shows the params and metrics side by side.
- Open a run and look under **Artifacts**. You'll find the `plots/` (ROC, confusion matrix, feature importance), `tuning/grid_search_results.csv`, `model/` (MLflow model with signature), and `joblib/model.joblib` (on the best run only).

Press `Ctrl+C` to stop the UI. Then commit:

```bash
git add heartml/features.py heartml/train.py heartml/predict.py models/ mlruns/ reports/runs notebooks/
git commit -m "stage 3: preprocessing pipeline, LR vs RF tuning, MLflow tracking, best model saved with joblib"
git push
```

📸 `03_mlflow_runs.png`, `03_mlflow_compare.png`, `03_mlflow_artifacts.png`

## Stage 4 - Unit tests and lint

```bash
make lint
make test      # 23 tests + coverage
```

```bash
git add tests/
git commit -m "stage 4: pytest suite for data, features, models and API; flake8 config"
git push
```

📸 `04_pytest.png`

## Stage 5 - Prediction API (FastAPI) and metrics endpoint

```bash
make api       # uvicorn on :8000  (leave running, open a 2nd terminal)
```

In the second terminal:

```bash
curl -s localhost:8000/health
curl -s -X POST localhost:8000/predict -H 'Content-Type: application/json' -d @scripts/sample_request.json
curl -s localhost:8000/metrics | grep heart_api_
curl -s localhost:8000/metrics/summary
```

Also open the Swagger UI by going to the forwarded port 8000 and adding `/docs` to the URL. Stop the API with `Ctrl+C`, then commit:

```bash
git add api/ scripts/sample_request.json scripts/smoke_test.sh scripts/generate_traffic.py
git commit -m "stage 5: FastAPI /predict with validation, JSON request logging, Prometheus /metrics"
git push
```

📸 `05_swagger.png`, `05_predict_curl.png`

## Stage 6 - Docker: build, run, test

```bash
docker build -t heart-risk-api:v1 .
docker images | grep heart-risk-api
docker run -d --name heart-api -p 8000:8000 heart-risk-api:v1
docker ps                       # STATUS should become "(healthy)"
scripts/smoke_test.sh http://localhost:8000
docker logs heart-api | tail    # one JSON log line per request
docker rm -f heart-api
```

```bash
git add Dockerfile .dockerignore
git commit -m "stage 6: multi-stage non-root Docker image with healthcheck"
git push
```

📸 `06_docker_build.png`, `06_docker_smoke.png`

## Stage 7 - Push the image to GHCR

First, create a token: GitHub -> **Settings -> Developer settings -> Personal access tokens -> Tokens (classic)** -> scopes
`write:packages` and `read:packages`. Copy it, then:

```bash
export CR_PAT=<paste-token>
echo $CR_PAT | docker login ghcr.io -u <you> --password-stdin
docker tag heart-risk-api:v1 ghcr.io/<you>/heart-risk-api:v1
docker tag heart-risk-api:v1 ghcr.io/<you>/heart-risk-api:latest
docker push ghcr.io/<you>/heart-risk-api:v1
docker push ghcr.io/<you>/heart-risk-api:latest
```

Your GitHub profile now has a **Packages** tab listing `heart-risk-api`. Open the package, go to **Package settings ->
Connect repository** -> `heart-risk-mlops`, and optionally change its visibility to **Public**.
Nothing new to commit at this stage. GHCR is evidence only.

📸 `07_ghcr_push.png`, `07_ghcr_package.png`

## Stage 8 - Minikube: cluster, load image, pods, scale

```bash
# 1. cluster
minikube start --driver=docker --cpus=2 --memory=4096
kubectl get nodes -o wide

# 2. load the local image into the minikube node (no registry pull needed)
minikube image load heart-risk-api:v1
minikube image ls | grep heart-risk-api

# 3. create namespace, deployment (2 pods), service, ingress
kubectl apply -f k8s/app/namespace.yaml
kubectl apply -f k8s/app/deployment.yaml -f k8s/app/service.yaml
minikube addons enable ingress && kubectl apply -f k8s/app/ingress.yaml
kubectl -n heart-risk rollout status deployment/heart-api
kubectl -n heart-risk get pods -o wide

# 4. scale
kubectl -n heart-risk scale deployment/heart-api --replicas=4
kubectl -n heart-risk get pods -w           # Ctrl+C once 4/4 Running
kubectl -n heart-risk scale deployment/heart-api --replicas=3

# 5. call the API through the Service (keep this running in its own terminal)
kubectl -n heart-risk port-forward svc/heart-api 8080:80
```

In another terminal:

```bash
scripts/smoke_test.sh http://localhost:8080
kubectl -n heart-risk logs -l app=heart-api --tail=5 --prefix   # see which pod served each request
kubectl -n heart-risk get deploy,rs,pods,svc,ingress
```

(`scripts/deploy_minikube.sh` does steps 1-5 in a single command, if you want a one-shot version.)

```bash
git add k8s/app scripts/deploy_minikube.sh
git commit -m "stage 8: k8s manifests (deployment, LoadBalancer service, ingress), minikube deploy + scaling"
git push
```

📸 `08_kubectl_pods.png`, `08_scaled_4.png`, `08_curl_via_service.png`

## Stage 9 - GitHub Actions pipeline (automates stages 2-8)

```bash
git add .github/workflows/mlops-pipeline.yml
git commit -m "stage 9: CI/CD - lint, test, train+quality gate, docker smoke, GHCR push, minikube deploy/scale, monitoring check"
git push
gh run watch            # or open the Actions tab
```

The workflow has four chained jobs. If any one fails, every job after it stops.

1. **Lint & unit tests**: flake8 and pytest (JUnit and coverage uploaded).
2. **Data, EDA & training**: MLflow runs, a quality gate (CV AUC ≥ 0.85), and a model-comparison table in the job summary. Model, reports and mlruns are uploaded as artifacts.
3. **Docker**: builds the image, runs the container, smoke-tests it, and pushes `latest`, `v1` and `sha-xxxxxxx` to GHCR using the built-in `GITHUB_TOKEN`.
4. **Minikube**: starts a cluster on the runner, loads the image, deploys, scales to 3 and checks it, smoke-tests through the Service, deploys Prometheus and Grafana, and checks that Prometheus sees all 3 pods. Cluster state is uploaded as an artifact.

> If the GHCR push fails with `denied`, open the package's **Package settings -> Manage Actions access**, add the
> `heart-risk-mlops` repo with **Write**, and re-run the job.
>
> To prove the pipeline fails on bad code (the brief asks for this), break a test on a branch, open a PR,
> and screenshot the red run.

📸 `09_actions_success.png`, `09_actions_summary.png`, `09_actions_artifacts.png`, `09_actions_failed_pr.png`

## Stage 10 - Monitoring: Prometheus and Grafana on minikube

```bash
# 1. pull images and load them into the minikube node
docker pull prom/prometheus:v2.54.1
docker pull grafana/grafana:11.2.0
minikube image load prom/prometheus:v2.54.1
minikube image load grafana/grafana:11.2.0

# 2. deploy (kustomize turns the config files into ConfigMaps)
kubectl apply -k k8s/monitoring
kubectl -n monitoring rollout status deployment/prometheus
kubectl -n monitoring rollout status deployment/grafana
kubectl -n monitoring get pods,svc
```

Run each of these in its own terminal (Codespaces forwards the ports for you):

```bash
kubectl -n heart-risk port-forward svc/heart-api 8080:80      # if not still running
kubectl -n monitoring port-forward svc/prometheus 9090:9090
kubectl -n monitoring port-forward svc/grafana 3000:3000
```

Generate traffic so the graphs have something to show:

```bash
python scripts/generate_traffic.py --url http://localhost:8080 --n 600 --delay 0.1
```

* **Prometheus** (port 9090): go to **Status -> Targets**. All `heart-api` pods should show **UP**. Try the query
  `sum by (pod) (rate(heart_api_requests_total[1m]))`.
* **Grafana** (port 3000): log in with `admin` / `HeartRisk@2026`. The dashboard **Heart Risk API - Service & Model Monitoring**
  is already provisioned (Dashboards -> Heart Risk MLOps).

**Building a dashboard by hand** (worth showing in the video):
Dashboards -> **New -> New dashboard -> Add visualization** -> data source **Prometheus**. Then:

1. Add a panel for **Request rate** with the query `sum by (endpoint) (rate(heart_api_requests_total[1m]))`, as a Time series with unit req/s.
2. Add a panel for **p95 latency** with the query `histogram_quantile(0.95, sum by (le) (rate(heart_api_request_latency_seconds_bucket{endpoint="/predict"}[2m])))`, unit seconds.
3. Add a panel for **Predictions by class** with the query `sum by (label) (increase(heart_api_predictions_total[5m]))`, as a Pie chart.
4. Click **Save** and name it, e.g. `Heart API - my dashboard`.

Live scaling demo: run `kubectl -n heart-risk scale deployment/heart-api --replicas=5`. A new line appears in **Requests per pod**,
and **API pods up** changes to 5.

```bash
git add k8s/monitoring scripts/deploy_monitoring.sh
git commit -m "stage 10: Prometheus (k8s pod discovery, alert rules) + Grafana with provisioned dashboard"
git push
```

📸 `10_prom_targets.png`, `10_grafana_dashboard.png`, `10_grafana_custom_panel.png`, `10_scale_in_grafana.png`

## Stage 11 - Evidence and report

```bash
git add screenshots/
git commit -m "stage 11: screenshots for report"
git push
```

Record a 3-5 minute screen video. It should show: the Actions run going green, the MLflow compare view, `docker run` plus curl, `kubectl get pods` and a scale, then Grafana updating under traffic.

### Troubleshooting

| Symptom | Fix |
|---|---|
| `minikube start` OOM or very slow | Use a 4-core Codespace. You can also try `minikube start --memory=3072`. |
| Pod `ErrImageNeverPull` | You skipped `minikube image load heart-risk-api:v1`, or the tag differs. |
| Changed code but pods run the old version | Rebuild the image, run `minikube image load` again, then `kubectl -n heart-risk rollout restart deploy/heart-api`. |
| Grafana panels empty | Check that Prometheus targets are UP, then generate traffic and wait about 30 s (the rate windows need data). |
| `unzip` asks to overwrite | Answer `A` (all). |
| Port not opening | Open the **Ports** tab in VS Code and click the globe icon for that port. |
