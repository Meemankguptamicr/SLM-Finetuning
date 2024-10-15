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

    if args.nlu_task == "chat-completion":
        sample_input_data = {
            "input_data": {
                "input_string": [
                {
                    "role": "user",
                    "content": "I am going to Paris, what should I see?"
                },
                {
                    "role": "assistant",
                    "content": "Paris, the capital of France, is known for its stunning architecture, art museums, historical landmarks, and romantic atmosphere. Here are some of the top attractions to see in Paris:\n\n1. The Eiffel Tower: The iconic Eiffel Tower is one of the most recognizable landmarks in the world and offers breathtaking views of the city.\n2. The Louvre Museum: The Louvre is one of the world's largest and most famous museums, housing an impressive collection of art and artifacts, including the Mona Lisa.\n3. Notre-Dame Cathedral: This beautiful cathedral is one of the most famous landmarks in Paris and is known for its Gothic architecture and stunning stained glass windows.\n\nThese are just a few of the many attractions that Paris has to offer. With so much to see and do, it's no wonder that Paris is one of the most popular tourist destinations in the world."
                },
                {
                    "role": "user",
                    "content": "What is so great about #1?"
                }
                ],
                "parameters": {
                "temperature": 0.8,
                "top_p": 0.8,
                "do_sample": True,
                "max_new_tokens": 2048
                }
            }
        }
    else:
        ValueError(f"Unsupported NLU task: {args.nlu_task}")

    if args.deployment_type == "finetuned-managed":
        deploy_finetuned_to_managed_compute(ml_client, sample_input_data, args.model_dir, args.base_model_name, args.endpoint_name, args.vm_type, args.instance_type, args.nlu_task)
    elif args.deployment_type == "base-managed-online":
        deploy_base_to_managed_compute(ml_client, sample_input_data, args.registry_name, args.base_model_name, args.endpoint_name, args.deployment_name, args.instance_type, args.nlu_task)
    elif args.deployment_type == "base-serverless":
        deploy_to_serverless_compute(ml_client, sample_input_data, args.registry_name, args.endpoint_name, args.base_model_name, args.nlu_task)
    else:
        raise ValueError(f"Unknown deployment type: {args.deployment_type}")