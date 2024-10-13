import os
import json
import argparse
from azure.ai.ml import MLClient
from azure.ai.ml.entities import (
    ManagedOnlineEndpoint,
    ManagedOnlineDeployment,
)
from azure.identity import DefaultAzureCredential
#from azureml.sfta.deployment import load_model_tokenizer, train_model


# Argument parser configuration
def parse_args():
    parser = argparse.ArgumentParser(description="Argument parser for deploying a model.")
    parser.add_argument("--run-name", type=str, help="Name of the deployment run", required=True)
    parser.add_argument("--registry-name", type=str, help="Name of the registry", default="azureml",required=True)
    parser.add_argument("--base-model-name", type=str, help="Name of the base model", default="Phi-3-mini-4k-instruct", required=True)
    parser.add_argument("--endpoint-name", type=str, help="Name of the endpoint", required=True)
    parser.add_argument("--deployment-name", type=str, help="Name of the deployment", required=True)
    parser.add_argument("--instance_type", type=str, help="Instance compute type", required=True)


# Deploy model as online endpoint
def deploy_model(ml_client, registry_name, model_name, endpoint_name, deployment_name, instance_type):
    managed_endpoint = ManagedOnlineEndpoint(
        name=endpoint_name,
        auth_mode="key",
        tags={"model_name": model_name},
    )
    ml_client.begin_create_or_update().wait(managed_endpoint)

    model_id = f"azureml://registries/{registry_name}/models/{model_name}/labels/latest"
    deployment = ManagedOnlineDeployment(
        name=deployment_name,
        endpoint_name=endpoint_name,
        model=model_id,
        instance_type=instance_type,
        instance_count=1,
    )
    ml_client.online_deployments.begin_create_or_update(deployment).wait()

    endpoint = ml_client.online_endpoints.get(endpoint_name)
    endpoint.traffic = {deployment_name: 100}
    ml_client.begin_create_or_update(endpoint).result()


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

    deploy_model(ml_client, args.registry_name, args.base_model_name, args.endpoint_name, args.deployment_name, args.instance_type)
    test_deployment(ml_client, args.endpoint_name)

    print("Finished Model Deployment")


if __name__ == "__main__":
    main()