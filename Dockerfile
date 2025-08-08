# Use a slim Python base image
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# Set working directory
WORKDIR /app

# Install system dependencies for pip packages (LangChain/Groq/Pinecone may require build tools)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    libffi-dev \
    libssl-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies first (cache layer)
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Copy project files
COPY . .

# Expose the FastAPI port
EXPOSE 8000

# Healthcheck to ensure container is alive
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/docs || exit 1

# Run FastAPI app with Uvicorn (1 worker for memory safety)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
