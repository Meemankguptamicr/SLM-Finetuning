import json
import requests
from azure.ai.ml.entities import (
    ManagedOnlineEndpoint,
    ManagedOnlineDeployment,
    Model,
    Environment,
    CodeConfiguration
)
from azure.core.exceptions import ResourceNotFoundError


# Deploy model as online endpoint
def deploy_model_online_endpoint(ml_client, registry_name, model_name, endpoint_name, deployment_name, instance_type):
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


def deploy_custom_to_online_endpoint(ml_client, endpoint_name, model, env, code_path, instance_type):  
    try:
        endpoint = ml_client.online_endpoints.get(endpoint_name)
        print(f"Endpoint '{endpoint_name}' already exists.")
    except ResourceNotFoundError:
        endpoint = ManagedOnlineEndpoint(
            name=endpoint_name,
            description="Online endpoint for SLM model",
            auth_mode="key",
        )
        ml_client.online_endpoints.begin_create_or_update(endpoint).result()
        print(f"Created endpoint '{endpoint_name}'.")

    code_config = CodeConfiguration(
        code=code_path,  # Path to directory containing score.py
        scoring_script="score.py",
    )
    deployment = ManagedOnlineDeployment(
        name="default",
        endpoint_name=endpoint_name,
        model=model,
        environment=env,
        code_configuration=code_config,
        instance_type=instance_type,
        instance_count=1,
    )
    ml_client.online_deployments.begin_create_or_update(deployment).result()
    print(f"Deployment 'default' created for endpoint '{endpoint_name}'.")
    endpoint.traffic = {"default": 100}
    ml_client.online_endpoints.begin_create_or_update(endpoint).result()
    print(f"Traffic updated for endpoint '{endpoint_name}'.")


def test_online_endpoint_deployment(ml_client, endpoint_name):
    endpoint = ml_client.online_endpoints.get(endpoint_name)
    scoring_uri = endpoint.scoring_uri
    if endpoint.auth_mode == "key":
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
    input_data = {
        "input": "Testing prompt"
    }
    response = requests.post(scoring_uri, headers=headers, json=input_data)
    print("Response status code:", response.status_code)
    print("Response body:", response.json())



def deploy_base_to_managed_compute(ml_client, registry_name, base_model_name, endpoint_name, deployment_name, instance_type):
    deploy_model_online_endpoint(ml_client, registry_name, base_model_name, endpoint_name, deployment_name, instance_type)
    test_online_endpoint_deployment(ml_client, endpoint_name)
    print("Finished Model Deployment")


def deploy_finetuned_to_managed_compute(ml_client, model_dir, model_name, endpoint_name, vm_type, instance_type):
    model = Model(
        path=model_dir,
        type="custom_model",
        name=model_name,
    )

    if vm_type == "cpu":
        env = Environment(
            name="cpu_inference_env",
            description="Environment for CPU inference",
            conda_file="cpu.yml",
            image="mcr.microsoft.com/azureml/minimal-py311-inference:latest",
        )
    else:
        env = Environment(
            name="gpu_inference_env",
            description="Environment for GPU inference",
            image="mcr.microsoft.com/azureml/acpt-pytorch-2.2-cuda12.1:latest",
        )

    deploy_custom_to_online_endpoint(
        ml_client=ml_client,
        endpoint_name=endpoint_name,
        model=model,
        env=env,
        code_path="./",
        instance_type=instance_type
    )

    test_online_endpoint_deployment(ml_client, endpoint_name)

    print("Finished Model Deployment")
