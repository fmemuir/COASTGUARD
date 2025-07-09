#!/bin/bash

# Name of the environment
ENV_NAME="coastguard"

# Path to environment.yml (adjust if needed)
ENV_YML="coastguard_env.yml"

# Step 2: create env
echo "[COASTGUARD] Updating conda..."
conda update -n base conda

echo "[COASTGUARD] Creating Conda environment: $ENV_NAME..."
conda env create -f "$ENV_YML" --name "$ENV_NAME" || {
    echo "[COASTGUARD] ❌ Failed to create environment."
    exit 1
}

echo "[COASTGUARD] Activating environment: $ENV_NAME"

# Step 3: activate env
# Conda activation in a script requires this setup:
eval "$(conda shell.bash hook)"
conda activate "$ENV_NAME" || {
    echo "[COASTGUARD] ❌ Failed to activate environment."
    exit 1
}

# Step 4: authenticate GEE
earthengine authenticate

# Step 5: install pip-only packages (copernicusmarine)
echo "[COASTGUARD] Installing pip-only packages..."
pip install "copernicusmarine>=1.0,<=2.0" || {
    echo "❌ pip install failed."
    exit 1
}

echo "[COASTGUARD] ✅ Environment '$ENV_NAME' is ready!"




