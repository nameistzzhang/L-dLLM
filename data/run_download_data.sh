#!/bin/bash

# Define the array of all datasets to download
DATASETS=( 
    "fineweb-edu"
)

# Define your python script name here
PYTHON_SCRIPT="download_data.py"

# Create a directory to store log files for easy debugging
LOG_DIR="./download_logs"
mkdir -p "$LOG_DIR"

echo "Starting batch download process at $(date)..."
echo "Logs will be saved in $LOG_DIR"

# Loop through each dataset and run the python script sequentially
for DATASET in "${DATASETS[@]}"; do
    echo "---------------------------------------------------"
    echo "Starting download for: $DATASET"
    
    # Run the download command and redirect stdout and stderr to a log file
    if python "$PYTHON_SCRIPT" --dataset "$DATASET" > "$LOG_DIR/${DATASET}.log" 2>&1; then
        echo "[SUCCESS] Finished downloading $DATASET at $(date)"
    else
        echo "[FAILED] Error downloading $DATASET. Check $LOG_DIR/${DATASET}.log for details."
    fi
done

echo "---------------------------------------------------"
echo "All download tasks completed at $(date)."