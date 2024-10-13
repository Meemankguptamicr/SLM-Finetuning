import argparse
import logging
from azure.identity import AzureCliCredential
from azure.ai.ml import MLClient
from azure.ai.ml.entities import ManagedOnlineDeployment, ManagedOnlineEndpoint, Model
from azure.ai.ml.constants import AssetTypes
import os

# Logging setup
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.DEBUG
)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Deploy a model to an online endpoint.")
    parser.add_argument("--endpoint-name", type=str, required=True, help="Name of the online endpoint")
    parser.add_argument("--model-name", type=str, required=True, help="Name of the registered model")
    parser.add_argument("--compute-instance", type=str, required=True, help="Compute instance type")
    return parser.parse_args()

def get_ml_client():
    """Initialize and return the ML Client."""
    credential = AzureCliCredential()
    subscription_id = os.getenv("AZURE_SUBSCRIPTION_ID")
    resource_group = os.getenv("AZURE_RESOURCE_GROUP")
    workspace_name = os.getenv("AZURE_WORKSPACE_NAME")
    
    return MLClient(credential, subscription_id, resource_group, workspace_name)

def deploy_model(args):
    ml_client = get_ml_client()

    # Check if the endpoint already exists, if not create a new one
    try:
        endpoint = ml_client.online_endpoints.get(args.endpoint_name)
        logger.info(f"Using existing endpoint: {args.endpoint_name}")
    except Exception as e:
        logger.info(f"Endpoint {args.endpoint_name} not found. Creating new endpoint.")
        endpoint = ManagedOnlineEndpoint(
            name=args.endpoint_name,
            description="Endpoint for serving the deployed model."
        )
        ml_client.online_endpoints.begin_create_or_update(endpoint).result()
        logger.info(f"Created new endpoint: {args.endpoint_name}")

    # Create or update the deployment
    deployment = ManagedOnlineDeployment(
        name="deployment",
        endpoint_name=args.endpoint_name,
        model=args.model_name,
        instance_type=args.compute_instance,
        instance_count=1,
        code_path="./",
        entry_script="score_hf.py",
    )
    
    deployment_result = ml_client.online_deployments.begin_create_or_update(deployment).result()
    logger.info(f"Deployment created for endpoint {args.endpoint_name}")

if __name__ == "__main__":
    args = parse_args()
    deploy_model(args)