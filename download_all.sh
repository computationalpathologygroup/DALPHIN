#!/bin/bash
set -e

# This script downloads and extracts the DALPHIN dataset.
# Ensure you have curl and unzip installed on your system.
# In total, the dataset is 1.17 GB in size, so downloading with a 100 Mbps connection should take about 1-2 minutes.

SCRIPT_DIR=$(dirname "$(readlink -f "$0")")
cd "$SCRIPT_DIR"

DATA_DIR="./data"
DOWNLOAD_URL="https://zenodo.org/api/records/18609450/files-archive"
ARCHIVE_NAME="dalphin.zip"

mkdir -p "$DATA_DIR"
cd "$DATA_DIR"

echo "Downloading data from $DOWNLOAD_URL..."
curl -L -o "$ARCHIVE_NAME" "$DOWNLOAD_URL"

# Extract the main archive
echo "Extracting $ARCHIVE_NAME..."
unzip -o "$ARCHIVE_NAME"
rm "$ARCHIVE_NAME"

# Extract sub-archive
if [ -f "images.zip" ]; then
    echo "Extracting images.zip..."
    unzip -o "images.zip"
    rm "images.zip"
else
    echo "Warning: images.zip not found, skipping..."
fi

echo "Download and extraction complete."