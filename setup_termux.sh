#!/bin/bash

# Glowing-Guacamole AI Assistant - Termux Setup Script
# This script prepares the Termux environment and installs all dependencies.

echo "--- Updating Termux packages... ---"
pkg update && pkg upgrade -y

echo "--- Installing required system packages... ---"
pkg install python git clang cmake ninja make binutils -y

echo "--- Creating virtual environment (optional but recommended)... ---"
python -m venv venv
source venv/bin/activate

echo "--- Installing Python dependencies... ---"
# Note: llama-cpp-python might take a while to compile on a phone
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt

echo "--- Installing the assistant as a package... ---"
pip install -e .

echo ""
echo "--- ✅ Setup Complete! ---"
echo "To run the CLI version: python cli.py"
echo "To run the GUI version (requires X11/VNC): python gui_flet.py"
echo ""
echo "Don't forget to place your 'your-model.gguf' in the project root."
