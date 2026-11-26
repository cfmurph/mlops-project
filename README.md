# MLOps starter (free stack)

Iris classifier with training, experiment tracking, tests, an API, Docker, CI/CD and Kubernetes.

```
push to GitHub ─▶ GitHub Actions: train ─▶ pytest (accuracy gate) ─▶ build image ─▶ push to ghcr.io
                                                                                         │
                                                         kind / minikube (local K8s) ◀───┘
```

## Run locally
```bash
uv venv -p 3.12 .venv && uv pip install -p .venv -r requirements.txt
.venv/bin/python -m src.train          # trains, logs to ./mlruns, saves model.joblib
.venv/bin/pytest                       # data + accuracy + API tests
.venv/bin/uvicorn src.app:app          # http://localhost:8000/docs
.venv/bin/mlflow ui                    # compare runs at http://localhost:5000 (needs: uv pip install mlflow)
```

## Docker
```bash
docker build -t iris-api:local .
docker run -p 8000:8000 iris-api:local
curl -X POST localhost:8000/predict -H 'Content-Type: application/json' \
  -d '{"sepal_length":5.1,"sepal_width":3.5,"petal_length":1.4,"petal_width":0.2}'
```

## CI/CD (GitHub Actions)
Push to a GitHub repo. Every push/PR trains and tests; pushes to `main` also publish
`ghcr.io/<user>/<repo>:latest`. No secrets to configure.

## Kubernetes (local, free)
```bash
brew install kind
kind create cluster
kind load docker-image iris-api:local
kubectl apply -f k8s/
kubectl port-forward svc/iris-api 8000:80
```
