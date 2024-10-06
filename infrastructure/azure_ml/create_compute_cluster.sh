#!/bin/bash

set -e

COMPUTE_NAME=${COMPUTE_NAME:-"gpu-cluster"}
VM_SIZE=${VM_SIZE:-"Standard_NC24ads_A100_v4"}
COMPUTE_TYPE=${COMPUTE_TYPE:-"AmlCompute"}
MIN_NODES=${MIN_NODES:-0}
MAX_NODES=${MAX_NODES:-1}
RESOURCE_GROUP=${RESOURCE_GROUP:-"my-resource-group"}
WORKSPACE_NAME=${WORKSPACE_NAME:-"my-workspace"}

if ! command -v az &> /dev/null; then
    echo "Azure CLI is not installed. Please install it first."
    exit 1
fi

echo "Creating Azure ML Compute Clusters..."

az ml compute create \
    --name "$COMPUTE_NAME" \
    --size "$VM_SIZE" \
    --type "$COMPUTE_TYPE" \
    --min-instances "$MIN_NODES" \
    --max-instances "$MAX_NODES" \
    --resource-group "$RESOURCE_GROUP" \
    --workspace "$WORKSPACE_NAME"

echo "Compute Cluster '$COMPUTE_NAME' created successfully."