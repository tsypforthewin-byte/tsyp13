# AIInterviewer

Purpose
- Simple Python-based service for running an AI-powered interviewer flow. Contains `main.py`, `requirements.txt`, and a `Dockerfile`.

Files
- `main.py` - service entrypoint. Likely contains the app logic to start the interviewer service.
- `requirements.txt` - Python dependencies.
- `Dockerfile` - container image build instructions.

Contract
- Inputs: probably REST requests or CLI input depending on the `main.py` implementation.
- Outputs: interviewer responses and logs.
- Success: service starts and responds on the configured port or prints expected CLI output.

How to run locally (recommended)
1. Create and activate a Python virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
``` 

2. Inspect `main.py` to confirm the entrypoint (look for `if __name__ == "__main__"` or a Flask/uvicorn run).

3. Run the service

```bash
python main.py
```

Note: If `main.py` uses Flask, it will likely run on port 5000 by default. If it uses uvicorn/fastapi, it may use port 8000. Inspect the file for exact port and host configuration.

How to run with Docker

```bash
docker build -t aiinterviewer:latest .
docker run --rm -p 5000:5000 aiinterviewer:latest
```

Adjust the host port mapping if the app uses a different internal port.

Deployment notes
- The repository contains a `Dockerfile` that can be used in a container registry and deployed to a container platform (Cloud Run, ECS, Kubernetes).

Tests & verification
- Add small smoke tests like a curl to the health endpoint or root path after startup. Example:

```bash
curl http://localhost:5000/health
``` 

Assumptions
- `main.py` is the runnable entrypoint. If it exposes a web server, check which port it uses and any required environment variables.

If you want, I can open `main.py` and add explicit run examples or create a small health-check test.
