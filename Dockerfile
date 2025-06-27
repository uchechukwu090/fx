# Use Python 3.11 slim image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Set environment variables (let docker-compose.yml handle overrides)
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV FLASK_APP=app.py

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    gfortran \
    libatlas-base-dev \
    liblapack-dev \
    libblas-dev \
    libffi-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create directory for persistent data
RUN mkdir -p /app/data

# Set permissions (optional: safer to manage on host, but okay for containers)
RUN chmod -R 755 /app/data

# Expose port (use 5000, not $PORT, for Docker compatibility)
EXPOSE 5000

# Health check (use fixed port for Docker healthcheck)
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:5000/api/health || exit 1

# Run the application with gunicorn
CMD gunicorn --bind 0.0.0.0:5000 --workers 2 --threads 4 --timeout 120 --worker-class sync app:app
