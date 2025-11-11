# Implement Steps

This section outlines the two main steps required to set up the environment and run the model evaluation script.

## Step-by-Step Execution
1. **Download Dataset to H5** Run `npz_to_h5.py`
2.  **Install Conda Environment:** Run the setup script to create and install the necessary dependencies defined in `environment.yml`.
3.  **Run Evaluation:** Execute the main script to load the trained model, predict on the test set, and generate evaluation reports.

---

## Bash Commands

Use the following commands in your terminal. Ensure both `.sh` files are executable (`chmod +x <filename>`).

### 1. Install Environment

```bash
bash ./setup_conda_env.sh 
# This command installs the Conda environment based on your YAML file.
# You only need to run this once.
bash ./run_evaluation.sh
# This command activates the newly installed environment and runs the evaluate.py script.
