# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from strenum import StrEnum


# Required env vars for AOAI
AZURE_OPENAI_ENDPOINT="AZURE_OPENAI_ENDPOINT"
AZURE_OPENAI_KEY="AZURE_OPENAI_KEY"
AZURE_OPENAI_DEPLOYMENT="AZURE_OPENAI_DEPLOYMENT"
AZURE_OPENAI_API_VERSION="AZURE_OPENAI_API_VERSION"

EMBEDDING_AZURE_OPENAI_KEY="EMBEDDING_AZURE_OPENAI_KEY"
EMBEDDING_AZURE_OPENAI_ENDPOINT="EMBEDDING_AZURE_OPENAI_ENDPOINT"
EMBEDDING_AZURE_OPENAI_API_VERSION="EMBEDDING_AZURE_OPENAI_API_VERSION"

AZURE_OPENAI_REQUIRED_VARS = [AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_KEY]


class EndpointType(StrEnum):
    """Endpoint Type."""

    AOAI = "AOAI"
    Serverless = 'Serverless'
    Null = "Null"


class AuthenticationType(StrEnum):
    """Authentication Type."""

    Unknown = 'unknown'
    ManagedIdentity = 'managed_identity'
    ApiKey = 'api_key'
    WorkspaceConnection = 'azureml_workspace_connection'
