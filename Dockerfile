FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

# Set working directory
WORKDIR /workspace

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    unzip \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY *.py ./

# Create necessary directories
RUN mkdir -p /workspace/tracking-compressed \
    /workspace/statsbomb_pl_data \
    /workspace/metadata_SecondSpectrum \
    /workspace/outputs

# Set environment variables
ENV PYTHONPATH=/workspace

# Default command
CMD ["python", "main_pipeline.py"]

