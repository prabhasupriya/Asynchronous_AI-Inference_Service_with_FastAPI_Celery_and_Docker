# Asynchronous AI Inference Service

A production-shaped, containerized microservice that serves a Scikit-Learn
text-classification model two ways:

- **Synchronously**, for quick, low-latency predictions.
- **Asynchronously**, via a Celery + Redis task queue, for workloads you
  don't want blocking the web server's event loop.

Built with **FastAPI**, **Celery**, **Redis**, and **Scikit-Learn**, fully
containerized with Docker and orchestrated with Docker Compose.

---

## Architecture

```
                    ┌────────────┐   1. POST /predict_async   ┌──────────────┐
                    │   Client   │ ──────────────────────────▶│   FastAPI    │
                    │            │◀────────────────────────── │   (api)      │
                    └────────────┘   3. {"task_id": "..."}     └──────┬───────┘
                          │                                            │ 2. task.delay()
                          │ 8. GET /status/{task_id}                   ▼
                          │                                    ┌──────────────┐
                          └───────────────────────────────────▶│    Redis     │
                                                                │ Broker +     │
                                                                │ Result       │
                                                                │ Backend      │
                                                                └──────┬───────┘
                                                        4. consume      │ 7. store
                                                           task         │ result+state
                                                                ▼
                                                        ┌──────────────┐
                                                        │Celery Worker │
                                                        │ (loads model │
                                                        │  once/process│
                                                        │  5&6. infer) │
                                                        └──────────────┘
```

**How the three services interact:**

1. **FastAPI (`api` service)** is the only component clients talk to. It
   exposes `/predict_sync` (blocking) and `/predict_async` +
   `/status/{task_id}` (non-blocking). On startup, FastAPI's `lifespan`
   handler loads the Scikit-Learn model/vectorizer into memory once, so
   `/predict_sync` never touches disk per-request.
2. **Redis** plays two distinct roles simultaneously:
   - **Message broker** — `/predict_async` serializes the task and pushes
     it onto a Redis-backed queue via `predict_task.delay(...)`.
   - **Result backend** — once a worker finishes (or fails) a task, Celery
     writes the state (`SUCCESS`/`FAILURE`) and return value back into
     Redis, keyed by task ID.
3. **Celery Worker (`worker` service)** is a separate, independently
   scalable process pool that pulls tasks off the Redis queue. It loads
   the model **once per worker process** (via the `worker_process_init`
   signal — not on every task execution) and runs the exact same
   `run_inference()` function that the synchronous endpoint uses, so
   behavior is identical no matter which path a request takes.

Because the API and the worker are separate containers, the worker can be
scaled independently (`docker compose up --scale worker=4`) without ever
touching the web tier, and a slow/heavy inference never blocks other HTTP
requests.

---

## Project Structure

```
project-root/
├── app/
│   ├── main.py             # FastAPI app, routes, lifespan model loading
│   ├── schemas.py          # Pydantic request/response models
│   ├── dependencies.py     # DI helpers (e.g. ensure_model_loaded)
│   └── config.py           # Pydantic BaseSettings (env-driven config)
├── worker/
│   ├── celery_app.py       # Celery instance (Redis broker + backend)
│   └── tasks.py            # Background inference task + logging
├── ml/
│   ├── train.py            # Generates model.pkl / vectorizer.pkl
│   ├── inference.py        # Shared load_models() / run_inference()
│   └── artifacts/          # model.pkl, vectorizer.pkl (generated)
├── tests/
│   ├── test_api.py         # FastAPI endpoint tests
│   └── test_tasks.py       # Celery task / ML logic tests
├── Dockerfile.api           # Multi-stage build → Uvicorn entrypoint
├── Dockerfile.worker         # Multi-stage build → Celery entrypoint
├── docker-compose.yml       # redis + api + worker orchestration
├── .env.example              # Documents every env var used
└── requirements.txt
```

---

## Prerequisites

- Docker & Docker Compose v2 (`docker compose version`)
- (Optional, for local non-Docker dev) Python 3.11+

---

## Quickstart (Docker Compose — recommended)

The model artifacts are baked into both Docker images at build time via the
`ml/train.py` script's output. **Generate them once before building:**

```bash
# 1. (Local Python env) install deps and train the model
python -m venv .venv && source .venv/bin/activate   # optional but recommended
pip install -r requirements.txt
python -m ml.train
# -> creates ml/artifacts/model.pkl and ml/artifacts/vectorizer.pkl

# 2. Copy the env template (defaults work out of the box for Compose)
cp .env.example .env

# 3. Build and start everything
docker compose up --build
```

This starts three containers:

| Service  | Purpose                              | Port |
|----------|---------------------------------------|------|
| `redis`  | Broker + result backend               | 6379 |
| `api`    | FastAPI app (Uvicorn)                 | 8000 |
| `worker` | Celery worker consuming tasks         | —    |

Wait for `redis` to report healthy (compose enforces this automatically
via `depends_on: condition: service_healthy` for both `api` and `worker`),
then visit **http://localhost:8000/docs** for interactive Swagger UI.

To scale workers horizontally:

```bash
docker compose up --build --scale worker=3
```

To stop everything:

```bash
docker compose down
```

---

## Running Locally Without Docker (dev mode)

```bash
pip install -r requirements.txt
python -m ml.train                     # generate artifacts

# Terminal 1: start Redis (or `docker run -p 6379:6379 redis:7-alpine`)
redis-server

# Terminal 2: start the Celery worker
celery -A worker.celery_app worker --loglevel=info

# Terminal 3: start FastAPI
uvicorn app.main:app --reload
```

---

## API Usage Examples

### 1. `POST /predict_sync` — blocking inference

```bash
curl -X POST http://localhost:8000/predict_sync \
  -H "Content-Type: application/json" \
  -d '{"text": "I love this product, it works great"}'
```

Response `200 OK`:
```json
{"prediction": "positive", "confidence": 0.87}
```

Invalid payload (empty text) → `422 Unprocessable Entity`:
```bash
curl -X POST http://localhost:8000/predict_sync \
  -H "Content-Type: application/json" \
  -d '{"text": ""}'
```

### 2. `POST /predict_async` — dispatch a background task

```bash
curl -X POST http://localhost:8000/predict_async \
  -H "Content-Type: application/json" \
  -d '{"text": "This is terrible, total waste of money"}'
```

Response `202 Accepted`:
```json
{"task_id": "3f9c2a10-3b44-4b8e-9c2d-4b1a2e9f0f11"}
```

### 3. `GET /status/{task_id}` — poll for the result

```bash
curl http://localhost:8000/status/3f9c2a10-3b44-4b8e-9c2d-4b1a2e9f0f11
```

While running:
```json
{"task_id": "3f9c2a10-...", "status": "PENDING", "result": null}
```

Once complete:
```json
{
  "task_id": "3f9c2a10-...",
  "status": "SUCCESS",
  "result": {"prediction": "negative", "confidence": 0.93}
}
```

If the task raised an exception, `/status` still returns `200 OK` (the
API itself never crashes on a worker-side failure):
```json
{
  "task_id": "3f9c2a10-...",
  "status": "FAILURE",
  "result": null,
  "error": "ValueError: Input text is empty after preprocessing."
}
```

### 4. `GET /health`

```bash
curl http://localhost:8000/health
# {"status": "ok"}
```

---

## Environment Variables

All configuration is externalized — see [`.env.example`](.env.example) for
the authoritative, documented list. Summary:

| Variable                    | Purpose                                    | Default (local)              |
|------------------------------|---------------------------------------------|-------------------------------|
| `REDIS_URL`                 | Celery broker + result backend URL          | `redis://localhost:6379/0`   |
| `LOG_LEVEL`                 | Python logging level for api + worker       | `INFO`                        |
| `MODEL_PATH`                | Path to the classifier `.pkl`               | `ml/artifacts/model.pkl`     |
| `VECTORIZER_PATH`           | Path to the TF-IDF vectorizer `.pkl`        | `ml/artifacts/vectorizer.pkl`|
| `CELERY_TASK_ALWAYS_EAGER`  | Run tasks synchronously in-process (tests)  | `false`                       |

In `docker-compose.yml`, `REDIS_URL` is overridden to
`redis://redis:6379/0` so containers resolve each other by service name
over the Docker bridge network, rather than `localhost`.

---

## Testing

```bash
pip install -r requirements.txt
export CELERY_TASK_ALWAYS_EAGER=true   # run Celery tasks in-process, no broker needed
python -m pytest tests/ -v
```

- **`tests/test_api.py`** — exercises the FastAPI contract directly via
  `TestClient`: 200 on valid `/predict_sync`, 422 on missing/empty/invalid
  `text`, task dispatch on `/predict_async` (Celery `.delay()` mocked so
  no live broker is required), and all three `/status` states
  (`PENDING` / `SUCCESS` / `FAILURE`) — including proving a worker-side
  exception is safely surfaced as JSON rather than crashing the endpoint.
- **`tests/test_tasks.py`** — exercises the Celery task and the shared ML
  inference logic: calling the task function directly, running it via
  `.apply()` in eager mode, and confirming a raised exception from
  `run_inference()` propagates as a task failure rather than being
  silently swallowed.

All 16 tests pass without a live Redis instance thanks to
`CELERY_TASK_ALWAYS_EAGER=true` and targeted mocking of `AsyncResult` /
`.delay()` — matching the FAQ's guidance to test Celery logic via eager
mode or direct function calls.

---

## Design Notes / Rationale

- **Model loaded once, not per-request.** `ml/inference.py` holds module-
  level globals populated by `load_models()`. FastAPI calls it once in
  `lifespan` (startup); Celery calls it once per worker process via the
  `worker_process_init` signal — never inside the task body per call.
- **Shared inference code.** Both the sync endpoint and the async task
  call the identical `run_inference()` function, so there's exactly one
  place that defines "what a prediction is."
- **Graceful failure handling.** A crashing Celery task is caught by
  Celery itself, marked `FAILURE`, and `/status` reports that state with
  a stringified error — it never raises inside the FastAPI handler.
- **Multi-stage Docker builds.** Each Dockerfile has a `builder` stage
  (installs into `--user` site-packages) and a slim `runtime` stage that
  copies only the installed packages + app code — no compilers or pip
  caches ship in the final image. Both images also run as a non-root user.
- **Config via environment only.** `app/config.py` uses
  `pydantic-settings` to read everything from env vars / `.env`, with no
  hardcoded URLs or secrets anywhere in the codebase.

---

## Troubleshooting

See the task FAQ for common issues (Celery `ConnectionError` to Redis,
sharing model files between containers, `/status` stuck on `PENDING`,
whether Postgres is needed). In short:

- Always set `REDIS_URL=redis://redis:6379/0` (the Compose service name,
  not `localhost`) when running inside containers.
- Model artifacts are baked into both images via `COPY ml/` in each
  Dockerfile — regenerate them with `python -m ml.train` before
  rebuilding if you change the training data.
- `/status` stuck on `PENDING` almost always means the worker crashed on
  startup (check `docker compose logs worker`) or the backend URL is
  misconfigured.

## watch youtude video here
[click here](https://youtu.be/fn_srZucXKI)


