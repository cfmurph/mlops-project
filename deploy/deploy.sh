#!/usr/bin/env bash
# Run on the PROD VM by the CI/CD deploy job, from /opt/housing.
# Expects envs/prod.env (written by CI, includes IMAGE_TAG) and GHCR_USER / GHCR_TOKEN in the environment.
set -euo pipefail
cd "$(dirname "$0")"

echo "$GHCR_TOKEN" | docker login ghcr.io -u "$GHCR_USER" --password-stdin
compose() { docker compose -f docker-compose.yaml -f docker-compose.prod.yaml --env-file envs/prod.env "$@"; }
compose pull
compose up -d --remove-orphans
docker logout ghcr.io

# Healthy only when the API answers and reports the prod environment.
for _ in $(seq 1 30); do
  if curl -fsS http://localhost:8000/health | grep -q '"env":"prod"'; then
    echo "Deployed $(grep '^IMAGE_TAG=' envs/prod.env)"; exit 0
  fi
  sleep 5
done
echo "API did not become healthy" >&2
compose ps >&2
exit 1
