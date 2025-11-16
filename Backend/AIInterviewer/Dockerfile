# -----------------------------
# Base Image
# -----------------------------
FROM python:3.12-slim

# -----------------------------
# System Dependencies
# -----------------------------
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    gfortran \
    build-essential \
    wget \
    gnupg2 \
    && rm -rf /var/lib/apt/lists/*

# Working Directory
# -----------------------------
WORKDIR /app

# -----------------------------
# Copy and Install Python Dependencies
# -----------------------------
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install gunicorn

# -----------------------------
# Copy Application Code
# -----------------------------
COPY main.py .

# -----------------------------
# Expose Port
# -----------------------------
EXPOSE 8080

# -----------------------------
# Command to Run
# -----------------------------

# For Cloud deployment (Gunicorn production server)
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "--workers=1", "--threads=2", "--timeout=300", "-k", "uvicorn.workers.UvicornWorker", "main:app"]