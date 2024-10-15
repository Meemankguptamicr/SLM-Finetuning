import os
from azure.ai.ml.entities import MarketplaceSubscription, ServerlessEndpoint
from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import SystemMessage, UserMessage
from azure.identity import DefaultAzureCredential


def create_marketplace_subscription(ml_client, model_id, model_name):
    marketplace_subscription = MarketplaceSubscription(
        model_id=model_id,
        name=model_name,
    )

    marketplace_subscription = ml_client.marketplace_subscriptions.begin_create_or_update(
        marketplace_subscription
    ).result()


def deploy_model_to_serverless_endpoint(
    ml_client, registry_name, endpoint_name, model_name="Phi-3-mini-4k-instruct", nlu_task="chat-completion"
):
    model_id = f"azureml://registries/{registry_name}/models/{model_name}/labels/latest"
    # For non-Microsoft models, uncomment the following line
    # create_marketplace_subscription(ml_client, model_id, model_name)
    deploy_serverless_endpoint(ml_client, model_id, endpoint_name)
    test_deployment(nlu_task)
    print("Finished Model Deployment")


def deploy_serverless_endpoint(ml_client, model_id, endpoint_name):
    serverless_endpoint = ServerlessEndpoint(
        name=endpoint_name,
        model_id=model_id,
    )

    created_endpoint = ml_client.serverless_endpoints.begin_create_or_update(
        serverless_endpoint
    ).result()

    endpoint_keys = ml_client.serverless_endpoints.get_keys(endpoint_name)
    print("Primary Key:", endpoint_keys.primary_key)


def test_deployment(nlu_task="chat-completion"):
    if nlu_task == "chat-completion":
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
    else:
        raise ValueError(f"Unsupported NLU task: {nlu_task}")