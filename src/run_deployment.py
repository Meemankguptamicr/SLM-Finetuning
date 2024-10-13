import argparse
from azureml.sfta.deployment.deploy_model import deploy_model

parser = argparse.ArgumentParser(description="Unified Deployment Script")

def parse_args():
    parser = argparse.ArgumentParser(description="Argument parser for deploying a model.")
    parser.add_argument("--run-name", type=str, help="Name of the deployment run", required=True)
    parser.add_argument("--endpoint-name", type=str, help="Name of the endpoint", required=True)
    parser.add_argument("--deployment-type", type=str, help="Type of deployment", required=True, choices=["finetuned-managed", "base-serverless", "base-managed"])

    # Arguments specific to finetuned-managed deployment
    parser.add_argument("--model-dir", type=str, help="Path to the finetuned model")
    parser.add_argument("--base_model_id", type=str, help="Base model ID for tagging")
    parser.add_argument("--endpoint-type", type=str, help="Type of the endpoint ('batch' or 'online')", choices=["batch", "online"])
    parser.add_argument("--deployment-name", type=str, help="Name of the deployment")
    parser.add_argument("--vm_type", type=str, help="VM type ('cpu' or 'gpu')")
    parser.add_argument("--instance_type", type=str, help="Instance compute type")

    # Arguments specific to base-serverless and base-managed deployments
    parser.add_argument("--registry-name", type=str, help="Name of the registry", default="azureml")
    parser.add_argument("--base-model-name", type=str, help="Name of the base model", default="Phi-3-mini-4k-instruct")


def enforce_required_args(args, required_args_list):
    missing_args = []
    for arg in required_args_list:
        if getattr(args, arg) is None:
            missing_args.append(f"--{arg.replace('_', '-')}")
    if missing_args:
        parser.error(f"The following arguments are required for {args.deployment_type} deployment: {', '.join(missing_args)}")


def main():
    args = parse_args()

    if args.deployment_type == "finetuned-managed":
        required_args = ["model_dir", "base_model_id", "endpoint_type", "deployment_name", "vm_type", "instance_type"]
        enforce_required_args(required_args)
    elif args.deployment_type == "base-managed":
        required_args = ["deployment_name", "instance_type"]
        enforce_required_args(required_args)
    elif args.deployment_type == "base-serverless":
        # registry_name and base_model_name have defaults, so they are always present
        pass
    else:
        parser.error("Invalid deployment type specified.")

    deploy_model(args)

    print("Finished Model Deployment")


if __name__ == "__main__":
    main()
