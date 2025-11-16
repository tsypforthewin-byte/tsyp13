# CVExtractor

Purpose
- Service that extracts structured data from uploaded CVs. Contains `app.py`, `Dockerfile`, and `requirements.txt`.

Files
- `app.py` - service entrypoint (likely a web API that accepts CV files or URLs and returns JSON extraction results).
- `requirements.txt` - Python dependencies.
- `Dockerfile` - for containerized deployment.

Contract
- Inputs: CV file (PDF, DOCX, or text) or a pointer to storage.
- Outputs: JSON with fields like name, contact info, experiences, education, skills, etc.

Run locally

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python app.py
```

If `app.py` is a FastAPI app, run with uvicorn:

```bash
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

Run with Docker

```bash
docker build -t cv-extractor:latest .
docker run --rm -p 8000:8000 cv-extractor:latest
```

Example request
- If the service exposes a POST `/extract` endpoint that accepts a `multipart/form-data` file field named `file`, a test using curl might be:

```bash
curl -F "file=@/path/to/resume.pdf" http://localhost:8000/extract
```

Notes & next steps
- Inspect `app.py` to find the exact endpoints, expected payloads, and required environment variables (for storage, logging, or model access).
- Add unit tests for common CV formats and an end-to-end smoke test.
