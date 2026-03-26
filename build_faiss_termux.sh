#!/bin/bash
# build_faiss_termux.sh - Advanced FAISS-CPU build script for Termux (Android)
# Created by Jules for Glowing-Guacamole AI Assistant

set -e

echo "--- [1/5] Updating Termux packages and installing dependencies ---"
pkg update && pkg upgrade -y
pkg install -y clang cmake make git swig openblas python-numpy libomp python

echo "--- [2/5] Cloning FAISS repository ---"
if [ ! -d "faiss" ]; then
    git clone --depth 1 https://github.com/facebookresearch/faiss.git
fi
cd faiss

echo "--- [3/5] Configuring CMake for Termux (ARM/AArch64 optimized) ---"
# We use -DFAISS_OPT_LEVEL=generic to avoid x86-specific instructions like AVX2.
# We explicitly link to Termux's OpenBLAS.
mkdir -p build
cd build

cmake .. \
    -DFAISS_ENABLE_GPU=OFF \
    -DFAISS_ENABLE_PYTHON=ON \
    -DPython_EXECUTABLE=$(which python) \
    -DCMAKE_BUILD_TYPE=Release \
    -DFAISS_OPT_LEVEL=generic \
    -DBUILD_TESTING=OFF \
    -DBUILD_SHARED_LIBS=ON

echo "--- [4/5] Compiling FAISS core and Python bindings ---"
make -j$(nproc)
make -j$(nproc) swigfaiss

echo "--- [5/5] Installing Python bindings ---"
# Fix: Corrected path from 'faiss/build' to '../faiss/python' relative to 'faiss/build'
cd ../faiss/python
python setup.py install

echo "--- ✅ FAISS-CPU for Termux build complete! ---"
echo "You can now 'import faiss' in your Python environment."
