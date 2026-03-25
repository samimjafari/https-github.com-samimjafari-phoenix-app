#!/bin/bash
echo "--- Installing tools for Kali Linux ---"
sudo apt update
sudo apt install -y python3 python3-pip git nodejs npm g++ build-essential \
    libgirepository1.0-dev libglib2.0-dev libgtk-3-dev \
    libayatana-appindicator3-dev gir1.2-appindicator3-0.1
pip3 install --upgrade pip setuptools wheel
pip3 install -r requirements.txt
pip3 install -e .
echo "--- Installation complete! ---"
