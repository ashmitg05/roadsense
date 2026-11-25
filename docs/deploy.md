# Deploy

## 0. Budget alert first
GCP Console → Billing → Budgets & alerts → create a budget (e.g. ₹500/month) with email alerts at 50/90/100%.
Cloud Run with min-instances 0 costs ~nothing idle, but a scraped public demo can still run up CPU time.

## 1. Backend → Cloud Run (CPU)
```
gcloud auth login
gcloud config set project <PROJECT_ID>
gcloud services enable run.googleapis.com artifactregistry.googleapis.com cloudbuild.googleapis.com
gcloud run deploy roadsense-api --source . --region asia-south1 \
  --cpu 2 --memory 2Gi --min-instances 0 --max-instances 1 --concurrency 4 \
  --allow-unauthenticated \
  --set-env-vars ALLOWED_ORIGINS=https://<your-app>.vercel.app,RATE_LIMIT_PER_MIN=30
```
`--max-instances 1`: video jobs live in process memory (see `src/inference/api.py`), and it caps cost.

Smoke test: `curl https://<service-url>/health`

## 2. Frontend → Vercel
```
cd app
npx vercel --prod      # framework: Vite; root: app
```
Set env var `VITE_API_URL=https://<service-url>` in the Vercel project, then redeploy.

## 3. Raspberry Pi 4/5 benchmark
```
# Raspberry Pi OS 64-bit, Python 3.11
git clone <repo> && cd roadsense
python -m venv .venv && . .venv/bin/activate
pip install -r requirements-serve.txt pyyaml pillow imagehash psutil
# copy models/ and data/splits + data/yolo (or rerun Phase 1 scripts) to the Pi
python -m src.optimize.benchmark --models models --tag pi5 --threads 4
```
No `--pt` on the Pi (no PyTorch): the table then has ONNX rows only. Use a heatsink/fan; thermal throttling shows up as a fat p95.
