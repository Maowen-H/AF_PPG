#!/bin/bash

# --- Script Name: run_evaluation.sh ---
# Purpose: Activates the necessary Conda environment and executes the model evaluation Python script.

# --- Configuration ---
CONDA_ENV_NAME="AF_Detect_New"         # <--- IMPORTANT: REPLACE with your actual Conda environment name!
PYTHON_SCRIPT="evaluate.py"            # Name of your evaluation script
LOG_DIR="./evaluation_logs"            # Directory to store logs and output files
LOG_FILE="$LOG_DIR/evaluation_$(date +%Y%m%d_%H%M%S).log"

# 1. Create log directory if it doesn't exist
mkdir -p "$LOG_DIR"

# 2. Check if the Python script exists
if [ ! -f "$PYTHON_SCRIPT" ]; then
    echo "ERROR: Evaluation script '$PYTHON_SCRIPT' not found." | tee "$LOG_FILE"
    exit 1
fi

# 3. Check for Conda and activate the environment
if ! command -v conda &> /dev/null; then
    echo "ERROR: Conda command not found. Cannot proceed with environment activation." | tee "$LOG_FILE"
    exit 1
fi

echo "--- Activating Conda environment: $CONDA_ENV_NAME ---" | tee -a "$LOG_FILE"
# The 'source' command is crucial for activating Conda correctly in scripts
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV_NAME"

if [ $? -ne 0 ]; then
    echo "ERROR: Failed to activate Conda environment '$CONDA_ENV_NAME'." | tee -a "$LOG_FILE"
    echo "Please check if the environment name is correct." | tee -a "$LOG_FILE"
    exit 1
fi

# 4. Execute the Python evaluation script
echo "--- Starting model evaluation @ $(date) ---" | tee -a "$LOG_FILE"
# Run the Python script. The output is redirected (tee -a) to both the console AND the log file.
# We also use 'set -x' temporarily to show the exact command being executed.
set -x
python3 "$PYTHON_SCRIPT" 2>&1 | tee -a "$LOG_FILE"
EXECUTION_STATUS=$?
set +x

# 5. Check execution status
if [ $EXECUTION_STATUS -eq 0 ]; then
    echo "--- Evaluation Completed Successfully ---" | tee -a "$LOG_FILE"
    echo "Results (confusion matrix image, reports, predictions) are available." | tee -a "$LOG_FILE"
else
    echo "--- Evaluation FAILED with exit code $EXECUTION_STATUS ---" | tee -a "$LOG_FILE"
    echo "Please check the log file ($LOG_FILE) for detailed errors." | tee -a "$LOG_FILE"
fi

# 6. Deactivate the Conda environment (Good practice)
conda deactivate