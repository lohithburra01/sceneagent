#!/usr/bin/env bash
# Activate SceneAgent pipeline env in WSL2.
# Usage (inside WSL):  source pipeline/setup_wsl.sh
export PATH="/usr/local/cuda-12.1/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
export CUDA_HOME="/usr/local/cuda-12.1"
export LD_LIBRARY_PATH="/usr/local/cuda-12.1/lib64:${LD_LIBRARY_PATH:-}"
export TORCH_CUDA_ARCH_LIST="8.9"  # RTX 4060 Laptop = Ada Lovelace sm_89
# CUDA 12.1 nvcc requires gcc <= 12; Ubuntu 24.04 default is gcc-13.
export CC=gcc-12
export CXX=g++-12
source "$HOME/venvs/sa/bin/activate"
# Pull HF_TOKEN (and any other secrets) from the gitignored .env
ENV_FILE="/mnt/c/Users/91910/Downloads/sceneagent/.env"
if [ -f "$ENV_FILE" ]; then
    set -a
    # shellcheck disable=SC1090
    . "$ENV_FILE"
    set +a
fi
