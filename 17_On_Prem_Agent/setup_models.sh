#!/bin/bash

# Exit on error
set -e

echo "Setting up environment for vllm and TinyLlama..."

# Create and activate virtual environment if it doesn't exist
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv .venv
fi

# Activate virtual environment
source .venv/bin/activate

# Check if Rust is installed
if ! command -v rustc &> /dev/null; then
    echo "Installing Rust compiler..."
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
    source "$HOME/.cargo/env"
else
    echo "Rust is already installed, skipping installation..."
    source "$HOME/.cargo/env"
fi

# Upgrade pip
echo "Upgrading pip..."
python -m pip install --upgrade pip

# Install PyTorch CPU version first
echo "Installing PyTorch CPU version..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install core dependencies in smaller batches
echo "Installing core dependencies (batch 1)..."
pip install transformers>=4.51.0 accelerate>=0.27.2

echo "Installing core dependencies (batch 2)..."
pip install safetensors>=0.4.1 sentencepiece>=0.2.0 ninja>=1.11.1

# Install vllm with specific version
echo "Installing vllm..."
pip install vllm==0.8.3

# Install other necessary packages in smaller batches
echo "Installing other packages (batch 1)..."
pip install blake3>=0.4.1 compressed-tensors>=0.9.2 depyf>=0.18.0 einops>=0.7.0

echo "Installing other packages (batch 2)..."
pip install gguf>=0.10.0 importlib_metadata>=7.0.1 lark>=1.2.2 llguidance>=0.7.9

echo "Installing other packages (batch 3)..."
pip install lm-format-enforcer>=0.10.11 mistral_common[opencv]>=1.5.4 numba>=0.61.0

echo "Installing other packages (batch 4)..."
pip install opencv-python-headless>=4.11.0 outlines>=0.1.11 partial-json-parser>=0.0.1

echo "Installing other packages (batch 5)..."
pip install prometheus-fastapi-instrumentator>=7.0.0 py-cpuinfo>=9.0.0 ray[cgraph]!=2.44.*,>=2.43.0

echo "Installing other packages (batch 6)..."
pip install scipy>=1.12.0 watchfiles>=0.21.0 xgrammar>=0.1.17 filelock>=3.16.1

echo "Setup complete! You can now run the vllm server with:"
echo "VLLM_PLATFORM=cpu CUDA_VISIBLE_DEVICES=\"\" python -m vllm.entrypoints.openai.api_server --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --port 8001 --device cpu --tensor-parallel-size 1 --gpu-memory-utilization 0.0 --worker-cls vllm.worker.cpu_worker.CPUWorker --max-model-len 2048 --dtype float32 --swap-space 2 --max-num-batched-tokens 2048 --max-num-seqs 256 --model-impl transformers --disable-async-output-proc --disable-custom-all-reduce --disable-cascade-attn"

# pip install --upgrade pip
# chmod +x setup_models.sh
#./setup_models.sh
#source "$HOME/.cargo/env" && source .venv/bin/activate && ./setup_models.sh

# rm -rf .venv && chmod +x setup_models.sh && ./setup_models.sh
#uv pip install --no-cache-dir --verbose vllm==0.8.3
#VLLM_PLATFORM=cpu CUDA_VISIBLE_DEVICES="" python -m vllm.entrypoints.openai.api_server --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --port 8001 --device cpu --tensor-parallel-size 1 --gpu-memory-utilization 0.0 --worker-cls vllm.worker.cpu_worker.CPUWorker --max-model-len 2048 --dtype float32 --swap-space 2 --max-num-batched-tokens 2048 --max-num-seqs 256 --model-impl transformers --disable-async-output-proc --disable-custom-all-reduce --disable-cascade-attn
#1 uv pip install setuptools_rust
# uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu && uv pip install vllm==0.8.3 --no-build-isolation
