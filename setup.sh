#!/bin/bash



# --- Configuration ---
PYTHON_ENV_NAME="venv"
REQUIREMENTS_FILE="requirements.txt"
NOTEBOOK_FILE="notebook/notebook_main.ipynb" 

# --- Functions ---
# Function to print messages
print_message() {
    echo "======================================================"
    echo "$1"
    echo "======================================================"
}

# --- Script Start ---
print_message "Starting environment setup and notebook execution"

# 1. Check for Python 3 and pip
if ! command -v python3 &> /dev/null || ! command -v pip3 &> /dev/null; then
    echo "Error: Python 3 and/or pip3 are not installed. Please install them to continue."
    exit 1
fi

# 2. Check if requirements.txt exists
if [ ! -f "$REQUIREMENTS_FILE" ]; then
    echo "Error: '$REQUIREMENTS_FILE' not found in the current directory."
    echo "Please create it with the necessary libraries."
    exit 1
fi

# 3. Check if notebook file exists
if [ ! -f "$NOTEBOOK_FILE" ]; then
    echo "Error: Notebook file '$NOTEBOOK_FILE' not found."
    echo "Please update the NOTEBOOK_FILE variable in this script."
    exit 1
fi

# 4. Create a Python virtual environment
if [ ! -d "$PYTHON_ENV_NAME" ]; then
    print_message "Creating Python virtual environment: $PYTHON_ENV_NAME"
    python3 -m venv "$PYTHON_ENV_NAME"
    if [ $? -ne 0 ]; then
        echo "Error: Failed to create virtual environment."
        exit 1
    fi
else
    print_message "Virtual environment '$PYTHON_ENV_NAME' already exists."
fi

# 5. Activate the virtual environment
print_message "Activating virtual environment"
source "$PYTHON_ENV_NAME/bin/activate"

# 6. Install required packages
print_message "Installing dependencies from $REQUIREMENTS_FILE"
pip install --upgrade pip
pip install -r "$REQUIREMENTS_FILE"
if [ $? -ne 0 ]; then
    echo "Error: Failed to install dependencies. Please check '$REQUIREMENTS_FILE'."
    deactivate
    exit 1
fi

# 7. Execute the Jupyter Notebook
print_message "Executing the Jupyter Notebook: $NOTEBOOK_FILE"
# The --execute flag runs the notebook from top to bottom.
# --to notebook --inplace will save the output back to the original file.
jupyter nbconvert --to notebook --execute "$NOTEBOOK_FILE" --inplace
if [ $? -ne 0 ]; then
    echo "Error: Failed to execute the notebook. Check the notebook for errors."
    deactivate
    exit 1
fi

# 8. Deactivate the environment and finish
print_message "Notebook execution complete. Deactivating environment."
deactivate

echo "Process finished successfully."
exit 0
