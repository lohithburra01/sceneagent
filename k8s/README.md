# Kubernetes deployment

Plain YAML manifests for running SceneAgent on a local `minikube` cluster. No
Helm, no operators; the whole thing is ~5 files and an `apply.sh` wrapper.

## Files

| File | What it does |
|---|---|
| `00-namespace.yaml` | Creates the `sceneagent` namespace. |
| `10-postgres.yaml` | Postgres 16 + pgvector as a StatefulSet with a 2Gi PVC. Auto-applies `db/init/001_schema.sql` via a `postgres-init` ConfigMap. ClusterIP service on 5432. |
| `20-redis.yaml` | Redis 7 Deployment + ClusterIP service on 6379. |
| `30-api.yaml` | FastAPI deployment (`sceneagent-api:dev`), ClusterIP service on 8000, `DATABASE_URL`/`REDIS_URL` pointing at in-cluster services, `GEMINI_API_KEY` pulled from the `agent-secrets` Secret. Mounts `/mnt/data` (host) read-only. |
| `40-web.yaml` | Next.js deployment (`sceneagent-web:dev`), NodePort service on `:3000` → nodePort `30000`. |
| `99-secrets.example.yaml` | Template for the `agent-secrets` Secret. Copy to `99-secrets.yaml`, fill in your Gemini key, then `kubectl apply` it. |

## Apply order

```bash
# 1. One-time: create your real secret (never committed)
cp k8s/99-secrets.example.yaml k8s/99-secrets.yaml
$EDITOR k8s/99-secrets.yaml           # paste GEMINI_API_KEY

# 2. Apply everything in numeric order
kubectl apply -f k8s/00-namespace.yaml
kubectl apply -f k8s/99-secrets.yaml
kubectl apply -f k8s/10-postgres.yaml
kubectl apply -f k8s/20-redis.yaml
kubectl apply -f k8s/30-api.yaml
kubectl apply -f k8s/40-web.yaml

# 3. Watch rollouts
kubectl -n sceneagent rollout status statefulset/postgres
kubectl -n sceneagent rollout status deploy/redis
kubectl -n sceneagent rollout status deploy/api
kubectl -n sceneagent rollout status deploy/web
```

Or just run `./k8s/apply.sh` which does all of the above plus the minikube
mount and in-cluster image build.

## Mounting the scene data into minikube

The `api` pod expects the prepared InteriorGS scene under `/app/data` (the
pod path) which comes from `/mnt/data` on the minikube VM. Before applying
the API manifest, start the host mount in a separate terminal:

```bash
minikube mount ./data:/mnt/data
# keep this running -- it's a long-lived foreground process
```

`apply.sh` starts this in the background for you. If you're not using
minikube, replace the `hostPath` in `30-api.yaml` with a PVC (or a
`hostPath` appropriate to your node) that points at the scene directory.

## Building images inside minikube's docker daemon

Both `sceneagent-api:dev` and `sceneagent-web:dev` are built locally. We
never push to a registry; instead we use minikube's built-in docker daemon
so `imagePullPolicy: IfNotPresent` finds them:

```bash
eval $(minikube docker-env)
docker build -t sceneagent-api:dev ./api
docker build -t sceneagent-web:dev ./web
```

`apply.sh` does this for you too.

## Accessing the app

```bash
# Get a browser-reachable URL for the frontend (NodePort 30000)
minikube service -n sceneagent web --url

# Optional: port-forward the API for curl / debugging
kubectl -n sceneagent port-forward svc/api 8000:8000

# Optional: port-forward Postgres to inspect the DB
kubectl -n sceneagent port-forward svc/postgres 5432:5432
```

## Regenerating the init ConfigMap

The schema is duplicated inline in `10-postgres.yaml` so `kubectl apply -f`
works with no extra flags. If you edit `db/init/001_schema.sql`, resync the
ConfigMap block by hand, or swap to:

```bash
kubectl -n sceneagent create configmap postgres-init \
  --from-file=db/init/001_schema.sql \
  --dry-run=client -o yaml | kubectl apply -f -
```

## Teardown

```bash
kubectl delete namespace sceneagent
```

That drops everything including the PVC.
