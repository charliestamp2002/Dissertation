#!/bin/bash

# Simple setup script for RunPod
echo "Setting up environment for main_pipeline.py..."

# Install Python dependencies
pip install -r requirements.txt

# Extract data if compressed
if [ -f "football_data.tar.gz" ]; then
    echo "Extracting football_data.tar.gz..."
    tar -xzf football_data.tar.gz
fi

# Create outputs directory
mkdir -p outputs

echo "Setup complete! You can now run: python main_pipeline.py"