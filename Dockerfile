# ---------- stage 1: build wheels ----------
FROM python:3.11-slim AS builder
WORKDIR /build
COPY requirements-serve.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements-serve.txt

# ---------- stage 2: runtime ----------
FROM python:3.11-slim
LABEL org.opencontainers.image.title="heart-risk-api" \
      org.opencontainers.image.description="Heart disease risk prediction API (FastAPI + scikit-learn)"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MODEL_PATH=/app/models/model.joblib \
    PORT=8000

COPY --from=builder /install /usr/local
WORKDIR /app
COPY heartml/ heartml/
COPY api/ api/
COPY models/model.joblib models/model_meta.json models/

RUN useradd --create-home --uid 10001 appuser && chown -R appuser /app
USER appuser

EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=3s --start-period=10s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health').status==200 else 1)"

CMD ["sh", "-c", "uvicorn api.main:app --host 0.0.0.0 --port ${PORT} --workers 1 --no-access-log"]
