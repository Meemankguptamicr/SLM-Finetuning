import os
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential
from managed.utils import (
    deploy_finetuned_model_to_managed_endpoint,
    deploy_base_model_to_managed_endpoint,
)
from serverless.utils import deploy_model_to_serverless_endpoint


def deploy_model(args):
    subscription_id = os.getenv("SUBSCRIPTION_ID")
    resource_group = os.getenv("RESOURCE_GROUP")
    workspace = os.getenv("WORKSPACE_NAME")

    ml_client = MLClient(
        DefaultAzureCredential(), subscription_id, resource_group, workspace
    )

    if args.nlu_task == "chat-completion":
        sample_input_data = {
            "input_data": {
                "input_string": [
                    {"role": "user", "content": "I am going to Paris, what should I see?"},
                    {
                        "role": "assistant",
                        "content": "Paris, the capital of France, is known for its stunning architecture...",
                    },
                    {"role": "user", "content": "What is so great about #1?"},
                ],
                "parameters": {
                    "temperature": 0.8,
                    "top_p": 0.8,
                    "do_sample": True,
                    "max_new_tokens": 2048,
                },
            }
        }
    else:
        raise ValueError(f"Unsupported NLU task: {args.nlu_task}")

    if args.deployment_type == "finetuned-managed":
        deploy_finetuned_model_to_managed_endpoint(
            ml_client=ml_client,
            sample_input_data=sample_input_data,
            model_dir=args.model_dir,
            model_name=args.base_model_name,
            endpoint_name=args.endpoint_name,
            vm_type=args.vm_type,
            instance_type=args.instance_type,
            instance_count=args.instance_count,
            auth_mode=args.auth_mode,
        )
    elif args.deployment_type == "base-managed":
        deploy_base_model_to_managed_endpoint(
            ml_client=ml_client,
            sample_input_data=sample_input_data,
            registry_name=args.registry_name,
            model_name=args.base_model_name,
            endpoint_name=args.endpoint_name,
            deployment_name=args.deployment_name,
            instance_type=args.instance_type,
            instance_count=args.instance_count,
            auth_mode=args.auth_mode,
        )
    elif args.deployment_type == "base-serverless":
        deploy_model_to_serverless_endpoint(
            ml_client=ml_client,
            registry_name=args.registry_name,
            endpoint_name=args.endpoint_name,
            model_name=args.base_model_name,
            nlu_task=args.nlu_task,
        )
    else:
        raise ValueError(f"Unknown deployment type: {args.deployment_type}")