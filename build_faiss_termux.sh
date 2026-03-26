#!/bin/bash
# build_faiss_termux.sh - Advanced FAISS-CPU build script for Termux (Android)
# Created by Jules for Glowing-Guacamole AI Assistant

set -e

# Error handling function
error_exit() {
    echo "❌ Error: $1"
    exit 1
}

echo "--- [1/6] Updating Termux packages and installing dependencies ---"
pkg update && pkg upgrade -y || error_exit "Failed to update pkg packages."
pkg install -y clang cmake make git swig openblas python-numpy libomp python || error_exit "Failed to install dependencies."

echo "--- [2/6] Cloning FAISS repository ---"
if [ ! -d "faiss" ]; then
    git clone --depth 1 https://github.com/facebookresearch/faiss.git || error_exit "Failed to clone FAISS repo."
fi
cd faiss

echo "--- [3/6] Configuring CMake for Termux (ARM/AArch64 optimized) ---"
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
    -DBUILD_SHARED_LIBS=ON || error_exit "CMake configuration failed."

echo "--- [4/6] Compiling FAISS core and Python bindings ---"
make -j$(nproc) || error_exit "FAISS core compilation failed."
make -j$(nproc) swigfaiss || error_exit "SWIG FAISS compilation failed."

echo "--- [5/6] Installing Python bindings ---"
# Standard FAISS structure: python/ folder is in the root of the repo.
cd ../python
python setup.py install || error_exit "Python bindings installation failed."

echo "--- [6/6] Verifying FAISS installation ---"
cd ../..
python -c "import faiss; print(f'✅ FAISS version {faiss.__version__} successfully installed in Termux!')" || error_exit "FAISS verification failed."

echo "--- ✅ FAISS-CPU for Termux build complete! ---"
