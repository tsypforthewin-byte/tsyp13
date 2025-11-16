# -----------------------------
# Base Image
# -----------------------------
FROM python:3.12-slim

# Prevent Python buffering
ENV PYTHONUNBUFFERED=1

# -----------------------------
# System Dependencies
# -----------------------------
RUN apt-get update && apt-get install -y \
    gcc g++ \
    && rm -rf /var/lib/apt/lists/*

# -----------------------------
# Working Directory
# -----------------------------
WORKDIR /app

# -----------------------------
# Copy Requirements and Install
# -----------------------------
COPY requirements.txt .

# Increase pip timeout and retry to avoid download failures
RUN pip install --upgrade pip && \
    pip install --no-cache-dir --timeout 1000 --retries 5 -r requirements.txt

# -----------------------------
# Copy Application Code
# -----------------------------
COPY app.py .

# -----------------------------
# Expose Port
# -----------------------------
EXPOSE 8080

# -----------------------------
# Command to Run FastAPI with Gunicorn + UvicornWorker
# -----------------------------
CMD ["gunicorn", "--bind", "0.0.0.0:$PORT", "--workers", "1", "--timeout", "300", "-k", "uvicorn.workers.UvicornWorker", "app:app"]
