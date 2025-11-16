import functions_framework
import logging
import os
import requests
from google.cloud import firestore

# --- Configuration ---
EXTRACTION_SERVICE_URL = os.getenv("EXTRACTION_SERVICE_URL")
if not EXTRACTION_SERVICE_URL:
    raise ValueError("EXTRACTION_SERVICE_URL must be set!")

# --- Firestore Status Setup ---
db = firestore.Client(project="tsyp-477814", database="tsypstore")
STATUS_COLLECTION = "cv_status"  # Separate collection for status

# --- CloudEvent Trigger Function ---
@functions_framework.cloud_event
def trigger_cv_extraction(cloud_event):
    """
    Triggered by GCS object creation (finalized).
    Extracts user_id from file path and calls extraction service.
    Updates status in Firestore so frontend can listen in real-time.
    """
    try:
        data = cloud_event.data
        bucket_name = data["bucket"]
        file_name = data["name"]

        # Only PDFs
        if not file_name.lower().endswith(".pdf"):
            logging.info(f"Ignoring non-PDF file: {file_name}")
            return

        # Extract user_id from path (first folder) and cv_id
        user_id = file_name.split("/")[0]
        cv_id = file_name.split("/")[-1]

        # Reference Firestore status document
        status_doc_ref = db.collection(STATUS_COLLECTION).document(f"{user_id}_{cv_id}")
        status_doc_ref.set({"user_id": user_id, "cv_id": cv_id, "status": "processing"})

        # Call extraction service
        payload = {"gcs_path": f"gs://{bucket_name}/{file_name}", "user_id": user_id}
        response = requests.post(EXTRACTION_SERVICE_URL, json=payload)

        # Update status based on response
        if response.status_code == 200:
            status_doc_ref.update({"status": "done"})
            logging.info(f"Extraction done for {file_name}")
        else:
            status_doc_ref.update({"status": "failed", "error": response.text})
            logging.error(f"Extraction failed for {file_name}: {response.status_code} {response.text}")

    except Exception as e:
        # Update status on exception
        try:
            status_doc_ref.update({"status": "failed", "error": str(e)})
        except Exception:
            logging.error(f"Error updating status in Firestore: {e}")
        logging.error(f"Error in trigger function: {e}")
