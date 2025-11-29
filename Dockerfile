# Dockerfile for ML Pipeline Components
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ ./src/
COPY pipeline.py .

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Default command
CMD ["python", "pipeline.py", "--mode", "mlflow"]
