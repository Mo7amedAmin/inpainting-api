# Use a CUDA base image for GPU support
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04
# FROM python:3.10-slim

# Install Python and system dependencies
RUN apt-get update && apt-get install -y \
    python3 python3-pip git \
    libgl1-mesa-glx \
    libglib2.0-0 && \
    ln -s /usr/bin/python3 /usr/bin/python

# Set working directory inside the container
WORKDIR /app

# Copy requirements and install them
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy the rest of the project
COPY . .

# Set environment variables to avoid warnings
ENV PYTHONUNBUFFERED=1 \
    TRANSFORMERS_CACHE=/app/cache \
    HF_HOME=/app/cache

# Run the FastAPI app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
