# syntax=docker/dockerfile:1

# Use a slim base to keep image size reasonable while still
# supporting OpenCV + MediaPipe runtime dependencies.
FROM python:3.12-slim

# Keep Python output unbuffered for clear container logs.
ENV PYTHONUNBUFFERED=1

# Set app working directory inside the container.
WORKDIR /app

# Install minimal OS packages required by OpenCV/video stack.
# ffmpeg is useful for video IO in many environments.
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency list first for Docker layer caching.
COPY requirements.txt /app/requirements.txt

# Upgrade pip and install Python dependencies.
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r /app/requirements.txt

# Copy the full project after dependencies are installed.
COPY . /app

# Ensure runtime directories exist inside container.
RUN mkdir -p /app/uploads /app/results

# Expose API port.
EXPOSE 8000

# Start FastAPI via Uvicorn.
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]
