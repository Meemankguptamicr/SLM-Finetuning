# SLM-Finetuning

![Work in Progress](https://img.shields.io/badge/status-work%20in%20progress-yellow)

This repository contains the necessary scripts and infrastructure setup for fine-tuning models in Azure Machine Learning.

## Clone the repository

```bash
git clone https://github.com/Meemankguptamicr/SLM-Finetuning
git checkout repo-restruct2
```

## Setup
Navigate to the repository directory:

```bash
cd SLM-Finetuning
```

Log in to Azure account:

```bash
az login
```

## Environment Creation in Azure ML

Create Azure ML environment:

```bash
ENV_NAME="gpu-finetuning-environment" \
RESOURCE_GROUP="your-resource-group" \
WORKSPACE_NAME="your-workspace-name" \
./scripts/helper/create_environment.sh
```

```bash
ENV_NAME="gpu-quantization-environment" \
RESOURCE_GROUP="your-resource-group" \
WORKSPACE_NAME="your-workspace-name" \
./scripts/helper/create_environment.sh
```

Create GPU compute cluster:

```bash
COMPUTE_NAME="gpu-fintetuning-compute" \
VM_SIZE="Standard_NC24ads_A100_v4" \
COMPUTE_TYPE="AmlCompute" \
MIN_NODES=0 \
MAX_NODES=1 \
RESOURCE_GROUP="your-resource-group" \
WORKSPACE_NAME="your-workspace-name" \
./scripts/helper/create_compute_cluster.sh
```

Create CPU compute cluster

```bash
COMPUTE_NAME="cpu-compute" \
VM_SIZE="Standard_DS11_v2" \
COMPUTE_TYPE="AmlCompute" \
MIN_NODES=0 \
MAX_NODES=1 \
RESOURCE_GROUP="your-resource-group" \
WORKSPACE_NAME="your-workspace-name" \
./scripts/helper/create_compute_cluster.sh
```

## End-to-end flow for RAFT

Run the script by passing in the required parameters:

```bash
./scripts/jobs/run_raft_flow.sh \
  --project-path "./" \
  --subscription-id "your-subscription-id" \
  --resource-group "your-resource-group" \
  --workspace-name "your-workspace-name"
```