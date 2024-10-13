#!/bin/bash

set -e

ENV_NAME=${ENV_NAME:-"gpu-environment"}
FILE=${FILE:-"./infrastructure/azure_ml/gpu_docker/gpu-environment.yml"}
RESOURCE_GROUP=${RESOURCE_GROUP:-"my-resource-group"}
WORKSPACE_NAME=${WORKSPACE_NAME:-"my-workspace"}

if ! command -v az &> /dev/null; then
    echo "Azure CLI is not installed. Please install it first."
    exit 1
fi

if ! az extension show -n ml -o none 2>/dev/null; then
    echo "Azure ML CLI extension is not installed. Installing..."
    az extension add -n ml
fi

echo "Registering environment '$ENV_NAME' in Azure ML..."

if [ -f "$FILE" ]; then
    echo "Using Dockerfile for environment creation..."
    az ml environment create \
        --name "$ENV_NAME" \
        --file "$FILE" \
        --resource-group "$RESOURCE_GROUP" \
        --workspace "$WORKSPACE_NAME"
else
    echo "Dockerfile not found at '$FILE'. Exiting."
    exit 1
fi

echo "Environment '$ENV_NAME' created successfully."