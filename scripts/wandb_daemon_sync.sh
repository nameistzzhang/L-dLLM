#!/bin/bash

# Define the target directory
TARGET_DIR="/scratch/aszalay1/tianze/L-dLLM"

# Navigate to the target directory or exit if it fails
cd "$TARGET_DIR" || { echo "Error: Cannot change directory to $TARGET_DIR"; exit 1; }

echo "Starting automatic wandb sync in $TARGET_DIR"

# Infinite loop to execute the command periodically
while true; do
    # Log the current execution time
    echo "[$(date)] Executing wandb sync --sync-all..."
    
    # Run the wandb sync command
    wandb sync --sync-all
    
    # Sleep for 600 seconds (10 minutes)
    sleep 600
done