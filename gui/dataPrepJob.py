import streamlit as st
import requests


def new_job_request(subscription_id="72c03bf3-4e69-41af-9532-dfcdc3eefef4", resource_group="shared-finetuning-rg",
                    workspace_name="ayushmishra-6237", id="job_id", api_version="2024-04-01", body=None):
    url = f"https://management.azure.com/subscriptions/{subscription_id}/resourceGroups/{resource_group}/providers/Microsoft.MachineLearningServices/workspaces/{workspace_name}/jobs/{id}?api-version={api_version}"

    # set request body
    if body is None:
        body = {}
    request_body = st.text_area("Request Body (JSON)", "{}")

    # auth
    headers = {
        "Authorization": "AZURE_ACCESS_TOKEN",  # Replace with your actual access token
        "Content-Type": "application/json"
    }

    return requests.put(url, headers=headers, json=body)
