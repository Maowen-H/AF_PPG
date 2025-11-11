#!/bin/bash

# --- Script Name: setup_conda_env.sh ---
# Purpose: Creates or updates a Conda environment using an environment.yml file.

# 1. Define the path to the environment definition file
ENV_FILE="environment.yml"

# 2. Check if the 'conda' command is available (i.e., Conda is installed and in PATH)
if ! command -v conda &> /dev/null
then
    echo "ERROR: Conda command not detected."
    echo "Please ensure Anaconda or Miniconda is installed and added to your PATH."
    exit 1
fi

# 3. Check if the environment file exists
if [ ! -f "$ENV_FILE" ]; then
    echo "ERROR: Environment definition file $ENV_FILE not found."
    echo "Please ensure $ENV_FILE is in the current directory."
    exit 1
fi

# 4. Extract the environment name from the YAML file (assumes name: is on the first line)
ENV_NAME=$(grep "name:" "$ENV_FILE" | awk '{print $2}')

if [ -z "$ENV_NAME" ]; then
    echo "ERROR: Could not extract environment name from $ENV_FILE."
    echo "Please ensure the file contains a 'name: <env_name>' definition."
    exit 1
fi

echo "--- Starting Conda environment installation: $ENV_NAME ---"

# 5. Attempt to create the environment using conda env create
# -f: Specifies the environment file
conda env create -f "$ENV_FILE"

# 6. Check the exit status of the previous command ($?)
if [ $? -eq 0 ]; then
    echo "============================================================"
    echo " Conda environment '$ENV_NAME' created successfully!"
    echo "Activate it using: conda activate $ENV_NAME"
    echo "============================================================"
else
    # If creation failed (likely because the environment already exists), attempt to update it
    echo "WARNING: Conda environment creation failed (possibly already exists). Attempting to update..."
    conda env update -f "$ENV_FILE"

    if [ $? -eq 0 ]; then
        echo "============================================================"
        echo " Conda environment '$ENV_NAME' updated successfully!"
        echo "Activate it using: conda activate $ENV_NAME"
        echo "============================================================"
    else
        echo "============================================================"
        echo " Conda environment creation and update both failed. Check logs and network connection."
        echo "============================================================"
        exit 1
    fi
fi