# CVModules

This folder groups several services that together support CV ingestion, extraction, review, and rewriting.

Submodules
- `CVExtractionCloudTrigger/` - cloud trigger component used to start extraction when a CV is uploaded or an event occurs.
- `CVExtractor/` - service that extracts structured data from CVs.
- `CVReviewer/` - service that reviews/examines extracted CVs and produces review notes or scores.
- `CVRewriter/` - service that rewrites or improves CV text.

How they typically interact
1. A CV is uploaded to storage (or submitted via an API).
2. `CVExtractionCloudTrigger` receives the event and kicks off an extraction job.
3. `CVExtractor` processes the CV and produces structured JSON with extracted fields.
4. `CVReviewer` scores or annotates the extracted CV.
5. `CVRewriter` can produce improved versions of CV text based on reviewer annotations or user requests.

Each submodule contains its own README with exact run/deploy instructions. See each folder for details and example commands.

Common notes
- All submodules include `requirements.txt` indicating Python packages required.
- Dockerfiles are present in several modules for containerization.
- For local development, run modules individually and use small test inputs to exercise the end-to-end flow.
