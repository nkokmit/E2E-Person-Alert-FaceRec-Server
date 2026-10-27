# Dockerfile
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive

WORKDIR /app

# Install system dependencies for OpenCV and networking
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy source code and default models/configs
COPY . .

# Ensure data directories exist
RUN mkdir -p data/snapshots/preview data/snapshots/events data/faces models/faces

EXPOSE 8000

CMD ["uvicorn", "app.api_min:app", "--host", "0.0.0.0", "--port", "8000"]

