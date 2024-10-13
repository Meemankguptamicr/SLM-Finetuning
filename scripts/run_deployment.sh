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
# Step 1: Export Environment Variables
# ---------------------------
echo "Exporting environment variables..."
export AZURE_SUBSCRIPTION_ID="$AZURE_SUBSCRIPTION_ID"
export AZURE_RESOURCE_GROUP="$AZURE_RESOURCE_GROUP"
export AZURE_WORKSPACE_NAME="$AZURE_WORKSPACE_NAME"

# ---------------------------
# Step 2: Login to Azure (Skip if already logged in)
# ---------------------------
if ! az account show > /dev/null 2>&1; then
    echo "Logging into Azure..."
    az login
else
    echo "Already logged into Azure."
fi

# ---------------------------
# Step 3: Set Azure Subscription
# ---------------------------
echo "Setting Azure subscription to $AZURE_SUBSCRIPTION_ID..."
az account set --subscription "$AZURE_SUBSCRIPTION_ID"

# ---------------------------
# Step 4: Configure Default Resource Group and Workspace
# ---------------------------
echo "Configuring default resource group and workspace..."
az configure --defaults group="$AZURE_RESOURCE_GROUP" workspace="$AZURE_WORKSPACE_NAME"

# Verify the default configuration
echo "Default configurations set:"
az configure -l | grep -E "group|workspace"

# ---------------------------
# Step 5: Navigate to Project Directory
# ---------------------------
echo "Navigating to project directory at $PROJECT_PATH..."
cd "$PROJECT_PATH"

# ---------------------------
# Step 6: Run Deployment Pipeline
# ---------------------------
PIPELINE_FILE="pipelines/deployment_pipeline.yml"
echo "Running the deployment pipeline using $PIPELINE_FILE..."
az ml job create --file "$PIPELINE_FILE"

echo "Deployment pipeline submitted successfully."