# JobMatcher

Purpose
- A Python microservice that performs job matching logic. Contains `app.py`, `requirements.txt`, and a `Dockerfile` for containerized use.

Files
- `app.py` - entrypoint for the service. Look inside to determine whether it's a Flask/FastAPI app or a script.
- `requirements.txt` - Python dependencies.
- `Dockerfile` - for building a container image.

Contract
- Inputs: resumes/CV data or job descriptions (likely via HTTP POSTs or queued messages).
- Outputs: matched job IDs or scored matches.

How to run locally

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python app.py
```

If `app.py` is a Flask app, it'll default to port 5000; if it's FastAPI/uvicorn, check the file for host/port. Adjust the run command accordingly (for uvicorn: `uvicorn app:app --reload --host 0.0.0.0 --port 8000`).

Run with Docker

```bash
docker build -t jobmatcher:latest .
docker run --rm -p 5000:5000 jobmatcher:latest
```

Deployment
- Use the Docker image and deploy to your preferred container platform. Ensure required environment variables (API keys, DB URLs) are supplied at deploy time.

Smoke test

```bash
curl http://localhost:5000/health
# or post a small JSON payload to the matching endpoint (see app.py)
```

Notes & next steps
- Inspect `app.py` to identify endpoints and required env vars. I can open it and provide an exact run command and example payloads.
