#!/bin/bash

if command -v nvidia-smi &> /dev/null; then
    echo "NVIDIA GPU detected, installing CUDA dependencies..."
    pip install -r requirements-cuda.txt
elif command -v rocminfo &> /dev/null; then
    echo "AMD GPU detected, installing ROCm dependencies..."
    pip install -r requirements-rocm.txt
else
    echo "No compatible GPU detected, installing CPU dependencies..."
    pip install -r requirements.txt
fi
