#!/bin/bash
# bland-v2 box setup: llama-server (CUDA) + Qwen3-0.6B-Base Q5_K_S gguf. Idempotent.
set -x
mkdir -p /workspace/blandv2 && cd /workspace/blandv2

# --- llama.cpp (server target only) ---
if [ ! -d llama.cpp ]; then
  git clone --depth 1 https://github.com/ggml-org/llama.cpp >> setup.log 2>&1
fi
cd llama.cpp
if [ ! -x build/bin/llama-server ]; then
  cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86 > ../cmake.log 2>&1
  echo CMAKE_RC=$?
  cmake --build build --config Release -j 48 --target llama-server > ../build.log 2>&1
  echo BUILD_RC=$?
fi
./build/bin/llama-server --version

# --- model ---
cd /workspace/blandv2
if [ ! -s Qwen3-0.6B-Base.Q5_K_S.gguf ]; then
  curl -sL -o gguf.part https://huggingface.co/mradermacher/Qwen3-0.6B-Base-GGUF/resolve/main/Qwen3-0.6B-Base.Q5_K_S.gguf \
    && mv gguf.part Qwen3-0.6B-Base.Q5_K_S.gguf
fi
ls -la Qwen3-0.6B-Base.Q5_K_S.gguf
sha256sum Qwen3-0.6B-Base.Q5_K_S.gguf | tee gguf.sha256
echo SETUP_DONE
