#!/bin/bash

PYTHON_FILE="convertFoodIngrMap.py"
ENV_NAME="temp_env_$(date +%s)" # Unique environment name
REQUIREMENTS_FILE="requirements_oldPandas.txt"

echo "Creating temporary conda environment: $ENV_NAME"
conda create -n "$ENV_NAME" python=3.7 -y

echo "Activating conda environment: $ENV_NAME"
source activate "$ENV_NAME"

echo "Installing a compatible setuptools version"
pip install setuptools==65.6.3

echo "Installing packages from $REQUIREMENTS_FILE"
if [ -f "$REQUIREMENTS_FILE" ]; then
    pip install -r "$REQUIREMENTS_FILE"
else
    echo "Warning: $REQUIREMENTS_FILE not found. Proceeding without installing packages."
fi

echo "Running Python file: $PYTHON_FILE"
python "$PYTHON_FILE"

echo "Deactivating conda environment"
conda deactivate

echo "Removing temporary conda environment: $ENV_NAME"
conda env remove -n "$ENV_NAME" -y

echo "Script finished."
