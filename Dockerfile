# Use slim Python base image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy all your project files into the container
COPY . /app

# Install system dependencies if needed (e.g., gcc for some packages)
RUN apt-get update && apt-get install -y build-essential && \
    pip install --upgrade pip && \
    pip install -r requirements.txt && \
   apt-get purge -y --auto-remove build-essential && \
    rm -rf /var/lib/apt/lists/*
    
# Expose the port your FastAPI app will run on
EXPOSE 8000

# Run the FastAPI server with a single worker to avoid OOM
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
