# Use official Python slim image
FROM python:3.12-slim

# Prevent Python buffering
ENV PYTHONUNBUFFERED=1

# Install system dependencies (needed for compiling some Python packages)
RUN apt-get update && apt-get install -y \
    gcc g++ \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements first (for better Docker layer caching)
COPY requirements.txt .

# Upgrade pip and install dependencies with retries and timeout
RUN pip install --upgrade pip && \
    pip install --no-cache-dir --timeout 1000 --retries 5 -r requirements.txt

# Copy the main application file
COPY app.py .

# Expose port
EXPOSE 8080

# Run the application using Gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "--timeout", "300", "app:app"]