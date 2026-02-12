FROM python:3.10-slim

WORKDIR /code

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Create cache directories with proper permissions
RUN mkdir -p /code/.cache/huggingface \
    /code/.cache/datasets \
    /code/.cache/triton && \
    chown -R nobody:nogroup /code/.cache && \
    chmod -R 777 /code/.cache

# Ensure /tmp directory has proper permissions for logging
RUN chmod 777 /tmp

# Set environment variables for Hugging Face cache
ENV HF_HOME=/code/.cache/huggingface
ENV HF_DATASETS_CACHE=/code/.cache/datasets
ENV TRITON_CACHE_DIR=/code/.cache/triton

# Copy requirements first to leverage Docker cache
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Ensure cache directories have proper permissions after copying
RUN chown -R nobody:nogroup /code/.cache && \
    chmod -R 777 /code/.cache

# Set environment variables for Hugging Face Spaces
ENV PYTHONUNBUFFERED=1
ENV FLASK_APP=app.py
ENV FLASK_ENV=production
ENV PORT=7860

# Switch to non-root user
USER nobody

# Expose the port the app runs on
EXPOSE 7860

# Command to run the application
CMD ["python", "app.py"] 