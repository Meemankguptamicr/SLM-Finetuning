ENV_NAME="my-gpu-environment" \
RESOURCE_GROUP="my-resource-group" \
WORKSPACE_NAME="my-workspace" \
./infrastructure/create_environment.sh


COMPUTE_NAME="gpu-environment" \
VM_SIZE="Standard_NC24ads_A100_v4" \
MIN_NODES=0 \
MIN_NODES=4 \
RESOURCE_GROUP="my-resource-group" \
WORKSPACE_NAME="my-workspace" \
./infrastructure/create_compute_cluster.sh


## Example Usages

### Option 1:
./scripts/run_finetuning_pipeline.sh \
  --subscription-id "your-subscription-id" \
  --resource-group "your-resource-group" \
  --workspace-name "your-workspace-name"

### Option 2:
export AZURE_SUBSCRIPTION_ID="your-subscription-id"
export AZURE_RESOURCE_GROUP="your-resource-group"
export AZURE_WORKSPACE_NAME="your-workspace-name"

./scripts/run_finetuning_pipeline.sh