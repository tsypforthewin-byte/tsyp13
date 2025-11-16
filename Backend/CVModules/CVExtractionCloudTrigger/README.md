# CVExtractionCloudTrigger

Purpose
- Cloud-trigger component designed to start the CV extraction process when an event (such as a file upload) occurs. Contains `cloudbuild.yaml`, `main.py`, and `requirements.txt`.

Files
- `cloudbuild.yaml` - Cloud Build config (likely used to deploy or run CI/CD steps on Google Cloud).
- `main.py` - trigger code that processes incoming events and calls the extractor.
- `requirements.txt` - Python dependencies.

How it likely works
- `main.py` is written to be run as a cloud function or cloud-run service that receives events (e.g., a storage bucket object finalize event or Pub/Sub message) and then calls the `CVExtractor` service or enqueues a job.

Run locally
1. Install dependencies

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Run the trigger code directly (if it supports a local test harness)

```bash
python main.py
```

3. For event simulation, create a small JSON file representing the cloud event and pass it to the script if it supports CLI testing, or run a small test wrapper that imports `main` and calls the handler with a sample event.

Deploy to Google Cloud (example)

If `main.py` is a Google Cloud Function handler:

```bash
gcloud functions deploy cv-extraction-trigger \
  --runtime python310 \
  --trigger-resource YOUR_BUCKET_NAME \
  --trigger-event google.storage.object.finalize \
  --entry-point YOUR_HANDLER_NAME \
  --region YOUR_REGION
```

If `cloudbuild.yaml` is intended to run the deployment, use:

```bash
gcloud builds submit --config cloudbuild.yaml .
```

Notes
- Inspect `main.py` to determine the exact handler signature and whether it expects Pub/Sub messages or storage events.
- Ensure service account permissions are configured for any calls made to other services.

Testing & verification
- Verify the trigger by uploading a test file to the configured bucket and confirm downstream extraction is invoked.
