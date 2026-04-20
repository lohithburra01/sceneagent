#!/usr/bin/env bash
# Bring the SceneAgent stack up on a local minikube cluster.
#
# PREREQUISITES:
#   1. minikube must be running (`minikube start`).
#   2. You must have already copied the secrets example and filled in the key:
#         cp k8s/99-secrets.example.yaml k8s/99-secrets.yaml
#         $EDITOR k8s/99-secrets.yaml         # paste GEMINI_API_KEY
#         kubectl apply -f k8s/99-secrets.yaml
#      apply.sh refuses to continue if the secret is missing.
#
# This script:
#   - starts `minikube mount ./data:/mnt/data` in the background
#   - points docker at minikube's daemon and builds sceneagent-api:dev + sceneagent-web:dev
#   - applies 00..40 manifests in order
#   - waits for each rollout
#   - prints the browser URL for the web service
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

NS=sceneagent

# --- sanity checks --------------------------------------------------------
command -v minikube >/dev/null 2>&1 || { echo >&2 "error: minikube not found in PATH"; exit 1; }
command -v kubectl  >/dev/null 2>&1 || { echo >&2 "error: kubectl not found in PATH";  exit 1; }
command -v docker   >/dev/null 2>&1 || { echo >&2 "error: docker not found in PATH";   exit 1; }

minikube status >/dev/null 2>&1 || {
  echo >&2 "error: minikube isn't running. Start it with 'minikube start' and retry."
  exit 1
}

# --- namespace + secret gate ---------------------------------------------
kubectl apply -f k8s/00-namespace.yaml

if ! kubectl -n "$NS" get secret agent-secrets >/dev/null 2>&1; then
  cat >&2 <<'EOF'
error: Secret "agent-secrets" does not exist in namespace sceneagent.

Create it before running apply.sh:
    cp k8s/99-secrets.example.yaml k8s/99-secrets.yaml
    $EDITOR k8s/99-secrets.yaml           # paste GEMINI_API_KEY
    kubectl apply -f k8s/99-secrets.yaml

Then re-run ./k8s/apply.sh.
EOF
  exit 1
fi

# --- host mount for scene data -------------------------------------------
echo "[apply.sh] starting 'minikube mount ./data:/mnt/data' in background..."
mkdir -p data
minikube mount ./data:/mnt/data >/tmp/sceneagent-minikube-mount.log 2>&1 &
MOUNT_PID=$!
# give minikube mount a moment to negotiate
sleep 3
if ! kill -0 "$MOUNT_PID" 2>/dev/null; then
  echo >&2 "error: minikube mount died immediately. Check /tmp/sceneagent-minikube-mount.log."
  exit 1
fi
echo "[apply.sh] minikube mount PID=$MOUNT_PID (log: /tmp/sceneagent-minikube-mount.log)"
trap 'echo "[apply.sh] stopping minikube mount (PID=$MOUNT_PID)"; kill $MOUNT_PID 2>/dev/null || true' EXIT

# --- build images inside minikube's docker daemon ------------------------
echo "[apply.sh] eval \$(minikube docker-env) ..."
eval "$(minikube docker-env)"

echo "[apply.sh] building sceneagent-api:dev ..."
docker build -t sceneagent-api:dev ./api

echo "[apply.sh] building sceneagent-web:dev ..."
docker build -t sceneagent-web:dev ./web

# --- apply manifests in numeric order ------------------------------------
echo "[apply.sh] applying manifests..."
kubectl apply -f k8s/10-postgres.yaml
kubectl apply -f k8s/20-redis.yaml
kubectl apply -f k8s/30-api.yaml
kubectl apply -f k8s/40-web.yaml

# --- wait for rollouts ---------------------------------------------------
echo "[apply.sh] waiting for rollouts..."
kubectl -n "$NS" rollout status statefulset/postgres --timeout=180s
kubectl -n "$NS" rollout status deploy/redis        --timeout=120s
kubectl -n "$NS" rollout status deploy/api          --timeout=180s
kubectl -n "$NS" rollout status deploy/web          --timeout=180s

echo
echo "[apply.sh] all rollouts complete. Pods:"
kubectl -n "$NS" get pods -o wide

echo
echo "[apply.sh] web URL:"
minikube service -n "$NS" web --url

# Keep mount process alive in foreground so the user can Ctrl-C to tear down.
echo
echo "[apply.sh] minikube mount is still running in background (PID=$MOUNT_PID)."
echo "[apply.sh] Press Ctrl-C to stop the mount (pods keep running)."
wait "$MOUNT_PID"
