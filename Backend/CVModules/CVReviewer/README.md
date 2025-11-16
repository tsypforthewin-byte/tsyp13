# CVReviewer

Purpose
- Service that reviews/examines extracted CVs and returns review notes, scores, or suggestions. Contains `main.py`, `requirements.txt`, and a `Dockerfile`.

Files
- `main.py` - entrypoint for the reviewer logic.
- `requirements.txt` - Python dependencies.
- `Dockerfile` - container build instructions.

Contract
- Inputs: extracted CV JSON or raw CV text.
- Outputs: review results, scores, and suggested improvements.

Run locally

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python main.py
```

If the module exposes an HTTP API via Flask/FastAPI, use the appropriate uvicorn or flask run command and check the configured port.

Docker

```bash
docker build -t cv-reviewer:latest .
docker run --rm -p 5000:5000 cv-reviewer:latest
```

Testing
- Send a sample extracted JSON to the reviewer endpoint and validate the output contains expected score fields and comments.

Notes
- Ensure any external model/API keys used for scoring are provided as environment variables when running or deploying the container.
