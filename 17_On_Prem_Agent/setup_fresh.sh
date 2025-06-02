#!/bin/bash

# Exit on error
set -e

echo "Setting up fresh environment for vllm and TinyLlama..."

# Check Python version
if ! command -v python3 &> /dev/null; then
    echo "Python 3 is required. Please install Python 3 first."
    exit 1
fi

# Check if Rust is installed
if ! command -v rustc &> /dev/null; then
    echo "Installing Rust compiler..."
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
    source "$HOME/.cargo/env"
else
    echo "Rust is already installed, skipping installation..."
    source "$HOME/.cargo/env"
fi

# Create and activate virtual environment
echo "Creating virtual environment..."
python3 -m venv .venv
source .venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
python -m pip install --upgrade pip

# Install PyTorch CPU version first
echo "Installing PyTorch CPU version..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install all other requirements
echo "Installing Python packages..."
pip install -r requirements.txt

echo "Setup complete! You can now run the vllm server with:"
echo "VLLM_PLATFORM=cpu CUDA_VISIBLE_DEVICES=\"\" python -m vllm.entrypoints.openai.api_server --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --port 8001 --device cpu --tensor-parallel-size 1 --gpu-memory-utilization 0.0 --worker-cls vllm.worker.cpu_worker.CPUWorker --max-model-len 2048 --dtype float32 --swap-space 2 --max-num-batched-tokens 2048 --max-num-seqs 256 --model-impl transformers --disable-async-output-proc --disable-custom-all-reduce --disable-cascade-attn" 