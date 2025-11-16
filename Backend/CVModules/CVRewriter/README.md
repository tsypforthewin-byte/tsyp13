# CVRewriter

Purpose
- Service that rewrites or improves CV text based on extracted data or reviewer suggestions. Contains `app.py`, `Dockerfile`, and `requirements.txt`.

Files
- `app.py` - entrypoint which likely exposes an HTTP API for rewrite requests.
- `requirements.txt` - Python dependencies.
- `Dockerfile` - container image definition.

Contract
- Inputs: raw CV text or extracted CV JSON plus rewrite parameters (tone, length, role focus).
- Outputs: rewritten CV text and optionally a confidence or explanation.

Run locally

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python app.py
```

If the app is FastAPI, run with uvicorn:

```bash
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

Docker

```bash
docker build -t cv-rewriter:latest .
docker run --rm -p 8000:8000 cv-rewriter:latest
```

Example request

```bash
curl -X POST http://localhost:8000/rewrite \
  -H "Content-Type: application/json" \
  -d '{"text": "Experienced software engineer...", "tone":"concise"}'
```

Notes
- Check `app.py` for exact endpoint paths and request schemas. Provide any API keys or model endpoints via environment variables.
