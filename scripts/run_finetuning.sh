#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# ---------------------------
# Function to display usage
# ---------------------------
usage() {
    echo "Usage: $0 [--subscription-id SUBSCRIPTION_ID] [--resource-group RESOURCE_GROUP] [--workspace-name WORKSPACE_NAME] [--project-path PROJECT_PATH]"
    echo ""
    echo "Alternatively, you can set the following environment variables:"
    echo "  AZURE_SUBSCRIPTION_ID"
    echo "  AZURE_RESOURCE_GROUP"
    echo "  AZURE_WORKSPACE_NAME"
    echo "  PROJECT_PATH"
    echo ""
    echo "Example:"
    echo "  $0 --subscription-id your-subscription-id --resource-group your-resource-group --workspace-name your-workspace-name --project-path /path/to/project"
    exit 1
}

# ---------------------------
# Parse command-line arguments
# ---------------------------
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --subscription-id) AZURE_SUBSCRIPTION_ID="$2"; shift ;;
        --resource-group) AZURE_RESOURCE_GROUP="$2"; shift ;;
        --workspace-name) AZURE_WORKSPACE_NAME="$2"; shift ;;
        --project-path) PROJECT_PATH="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; usage ;;
    esac
    shift
done

# ---------------------------
# Validate inputs
# ---------------------------
if [[ -z "$AZURE_SUBSCRIPTION_ID" ]]; then
    echo "Error: AZURE_SUBSCRIPTION_ID is not set."
    usage
fi

if [[ -z "$AZURE_RESOURCE_GROUP" ]]; then
    echo "Error: AZURE_RESOURCE_GROUP is not set."
    usage
fi

if [[ -z "$AZURE_WORKSPACE_NAME" ]]; then
    echo "Error: AZURE_WORKSPACE_NAME is not set."
    usage
fi

if [[ -z "$PROJECT_PATH" ]]; then
    echo "Error: PROJECT_PATH is not set."
    usage
fi

# ---------------------------
# Step 1: Login to Azure (Skip if already logged in)
# ---------------------------
if ! az account show > /dev/null 2>&1; then
    echo "Logging into Azure..."
    az login
else
    echo "Already logged into Azure."
fi

# ---------------------------
# Step 2: Set Azure Subscription
# ---------------------------
echo "Setting Azure subscription to $AZURE_SUBSCRIPTION_ID..."
az account set --subscription "$AZURE_SUBSCRIPTION_ID"

# ---------------------------
# Step 3: Configure Default Resource Group and Workspace
# ---------------------------
echo "Configuring default resource group and workspace..."
az configure --defaults group="$AZURE_RESOURCE_GROUP" workspace="$AZURE_WORKSPACE_NAME"

# Verify the default configuration
echo "Default configurations set:"
az configure -l | grep -E "group|workspace"

# ---------------------------
# Step 4: Navigate to Project Directory
# ---------------------------
echo "Navigating to project directory at $PROJECT_PATH..."
cd "$PROJECT_PATH"

# ---------------------------
# Step 5: Ensure Azure ML CLI Extension is Installed
# ---------------------------
echo "Checking for Azure ML CLI extension..."
if ! az extension show -n ml -o none 2>/dev/null; then
    echo "Azure ML CLI extension not found. Installing..."
    az extension add -n ml -y
else
    echo "Azure ML CLI extension is already installed."
fi

# ---------------------------
# Step 6: Run the Finetuning Pipeline
# ---------------------------
PIPELINE_FILE="pipelines/finetuning_pipeline.yaml"

echo "Running the finetuning pipeline using $PIPELINE_FILE..."
az ml job create --file "$PIPELINE_FILE"

echo "Finetuning pipeline submitted successfully."