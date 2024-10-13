import os
import datetime
import argparse
from azure.ai.ml import MLClient
from azure.ai.ml.entities import (
    ManagedOnlineEndpoint,
    ManagedOnlineDeployment,
    Model,
    Environment,
    CodeConfiguration
)
from azure.core.exceptions import ResourceNotFoundError
!!!!!import onnx
from azure.identity import DefaultAzureCredential
#from azureml.sfta.deployment import load_model_tokenizer, train_model


# Argument parser configuration
def parse_args():
    parser = argparse.ArgumentParser(description="Argument parser for deploying a model.")
    parser.add_argument("--run-name", type=str, help="Name of the deployment run", required=True)
    parser.add_argument("--model-dir", type=str, help="Path to the model that should be deployed", required=True)
    parser.add_argument("--base_model_id", type=str, help="Model name (used for tagging the deployment)", required=True)
    parser.add_argument("--endpoint-type", type=str, help="Type of the endpoint ('batch' or 'online')", required=True, choices=["batch", "online"])
    parser.add_argument("--endpoint-name", type=str, help="Name of the endpoint", required=True)
    parser.add_argument("--deployment-name", type=str, help="Name of the deployment", required=True)
    parser.add_argument("--instance_type", type=str, help="Instance compute type", required=True)


def deploy_model_to_online_endpoint(ml_client, endpoint_name, model, code_path, instance_type):  
    # Create or get the endpoint
    try:
        endpoint = ml_client.online_endpoints.get(endpoint_name)
        print(f"Endpoint '{endpoint_name}' already exists.")
    except ResourceNotFoundError:
        endpoint = ManagedOnlineEndpoint(
            name=endpoint_name,
            description="Online endpoint for phi-3 model",
            auth_mode="key",
        )
        ml_client.online_endpoints.begin_create_or_update(endpoint).result()
        print(f"Created endpoint '{endpoint_name}'.")

    # Define the code configuration
    code_config = CodeConfiguration(
        code=code_path,  # Path to directory containing score.py
        scoring_script="score.py",
    )

    # Define the deployment
    deployment = ManagedOnlineDeployment(
        name="default",
        endpoint_name=endpoint_name,
        model=model,
        environment=env,
        code_configuration=code_config,
        instance_type=instance_type,
        instance_count=1,
    )

    # Create or update the deployment
    ml_client.online_deployments.begin_create_or_update(deployment).result()
    print(f"Deployment 'default' created for endpoint '{endpoint_name}'.")

    # Set traffic to direct 100% to this deployment
    endpoint.traffic = {"default": 100}
    ml_client.online_endpoints.begin_create_or_update(endpoint).result()
    print(f"Traffic updated for endpoint '{endpoint_name}'.")


def test_deployment(ml_client, endpoint_name):
    import json
    import requests

    # Get the endpoint details
    endpoint = ml_client.online_endpoints.get(endpoint_name)
    scoring_uri = endpoint.scoring_uri

    if endpoint.auth_mode == "key":
        # Retrieve the primary key
        keys = ml_client.online_endpoints.list_keys(name=endpoint.name)
        key = keys.primary_key
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {key}"
        }
    else:
        headers = {
            "Content-Type": "application/json"
        }

    # Prepare input data
    input_data = {
        "input": "Your test input here"
    }

    # Send POST request
    response = requests.post(scoring_uri, headers=headers, json=input_data)

    # Print the response
    print("Response status code:", response.status_code)
    print("Response body:", response.json())



def main():
    args = parse_args()

    subscription_id = os.getenv("SUBSCRIPTION_ID")
    resource_group = os.getenv("RESOURCE_GROUP")
    workspace = os.getenv("WORKSPACE_NAME")

    ml_client = MLClient(
        DefaultAzureCredential(), subscription_id, resource_group, workspace
    )

    model = Model(
        path=args.model_dir,
        type="custom_model",
        name=!!model_name!!,
    )


    !! MUST BE A CHOICE BETWEEN CPU AND GPU DEPENDING ON INSTANCE TYPE
    env = Environment(
        name="inference_env",
        description="Environment for phi-3 model inference",
        conda_file="environment.yml",
        image="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04:latest",
    )

    deploy_model_to_online_endpoint(
        ml_client=ml_client,
        endpoint_name=args.endpoint_name,
        model=model,
        code_path="./code",
        instance_type=args.instance_type
    )

    test_deployment(ml_client, args.endpoint_name)

    print("Finished Model Deployment")


if __name__ == "__main__":
    main()