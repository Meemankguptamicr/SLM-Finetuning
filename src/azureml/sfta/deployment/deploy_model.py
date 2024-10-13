import os
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential
from managed.utils import deploy_model as deploy_finetuned_to_managed_compute, deploy_base_to_managed_compute
from serverless.utils import deploy_model as deploy_to_serverless_compute

def deploy_model(args):
    subscription_id = os.getenv("SUBSCRIPTION_ID")
    resource_group = os.getenv("RESOURCE_GROUP")
    workspace = os.getenv("WORKSPACE_NAME")

    ml_client = MLClient(
        DefaultAzureCredential(), subscription_id, resource_group, workspace
    )

    if args.deployment_type == "finetuned-managed":
        deploy_finetuned_to_managed_compute(ml_client, args.model_dir, args.base_model_name, args.endpoint_name, args.vm_type, args.instance_type)
    elif args.deployment_type == "base-managed":
        deploy_base_to_managed_compute(ml_client, args.registry_name, args.base_model_name, args.endpoint_name, args.deployment_name, args.instance_typ)
    elif args.deployment_type == "base-serverless":
        deploy_to_serverless_compute()