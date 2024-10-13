import os
import json
import argparse
from azure.ai.ml import MLClient
from azure.ai.ml.entities import (
    MarketplaceSubscription,
    ServerlessEndpoint
)
from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import SystemMessage, UserMessage
from azure.identity import DefaultAzureCredential
#from azureml.sfta.deployment import load_model_tokenizer, train_model


# Argument parser configuration
def parse_args():
    parser = argparse.ArgumentParser(description="Argument parser for deploying a model.")
    parser.add_argument("--run-name", type=str, help="Name of the deployment run", required=True)
    parser.add_argument("--registry-name", type=str, help="Name of the registry", default="azureml",required=True)
    parser.add_argument("--base-model-name", type=str, help="Name of the base model", default="Phi-3-mini-4k-instruct", required=True)
    parser.add_argument("--endpoint-name", type=str, help="Name of the endpoint", required=True)


def create_marketplace_subscription(ml_client, model_id, model_name):
    marketplace_subscription = MarketplaceSubscription(
        model_id=model_id,
        name=model_name,
    )

    marketplace_subscription = ml_client.marketplace_subscriptions.begin_create_or_update(
        marketplace_subscription
    ).result()

# Deploy model as online endpoint
def deploy_model(ml_client, model_id, endpoint_name):
    serverless_endpoint = ServerlessEndpoint(
        name=endpoint_name,
        model_id=model_id
    )

    created_endpoint = ml_client.serverless_endpoints.begin_create_or_update(
        serverless_endpoint
    ).result()

    endpoint_keys = ml_client.serverless_endpoints.get_keys(endpoint_name)
    print(endpoint_keys.primary_key)


def test_deployment():
    client = ChatCompletionsClient(
        endpoint=os.environ["AZURE_INFERENCE_ENDPOINT"],
        credential=DefaultAzureCredential(),
    )
    model_info = client.get_model_info()
    print("Model name:", model_info.model_name)
    print("Model type:", model_info.model_type)
    print("Model provider name:", model_info.model_provider_name)

    response = client.complete(
        messages=[
            SystemMessage(content="You are a helpful assistant."),
            UserMessage(content="How many languages are in the world?"),
        ],
    )

    print("Response:", response.choices[0].message.content)
    print("Model:", response.model)
    print("Usage:")
    print("\tPrompt tokens:", response.usage.prompt_tokens)
    print("\tTotal tokens:", response.usage.total_tokens)
    print("\tCompletion tokens:", response.usage.completion_tokens)


# Test the deployment
def test_deployment(ml_client, endpoint_name):
    scoring_file = "./sample_score.json"
    with open(scoring_file, "w") as outfile:
        outfile.write('{"inputs": ["Paris is the [MASK] of France.", "The goal of life is [MASK]."]}')   
    response = ml_client.online_endpoints.invoke(
        endpoint_name=endpoint_name,
        deployment_name="demo",
        request_file=scoring_file,
    )
    response_json = json.loads(response)
    print(json.dumps(response_json, indent=2))


def main():
    args = parse_args()

    subscription_id = os.getenv("SUBSCRIPTION_ID")
    resource_group = os.getenv("RESOURCE_GROUP")
    workspace = os.getenv("WORKSPACE_NAME")

    ml_client = MLClient(
        DefaultAzureCredential(), subscription_id, resource_group, workspace
    )

    model_id = f"azureml://registries/{args.registry_name}/models/{args.base_model_name}/labels/latest"
    
    if args.registry_name != "azureml":
        print("For non-Microsoft models it's necessary to create the subscription first")
        create_marketplace_subscription(ml_client, model_id, args.base_model_name)
        
    deploy_model(ml_client, model_id, args.endpoint_name)
    test_deployment()

    print("Finished Model Deployment")


if __name__ == "__main__":
    main()