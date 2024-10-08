# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Authentication provider."""

import json
import os
import logging
import requests

from abc import abstractmethod
from azureml.core import Run, Workspace
from azureml.core.run import _OfflineRun
from raft.logconf import log_setup
from requests.adapters import HTTPAdapter
from urllib3 import Retry


log_setup()
logger = logging.getLogger("raft")


class AuthProvider:
    """Authentication provider."""

    @abstractmethod
    def get_auth_headers(self) -> dict:
        """Get auth headers."""
        pass


class WorkspaceConnectionAuthProvider(AuthProvider):
    """Workspace connection auth provider."""

    def __init__(self, connection_name, endpoint_type) -> None:
        """Initialize WorkspaceConnectionAuthProvider."""
        self._current_workspace = None
        self._connection_name = connection_name
        self._endpoint_type = endpoint_type

    @property
    def current_workspace(self) -> Workspace:
        """Get the current workspace."""
        if self._current_workspace is None:
            run: Run = Run.get_context()
            self._current_workspace = (
                Workspace.from_config()
                if isinstance(run, _OfflineRun)
                else run.experiment.workspace
            )
        return self._current_workspace

    def get_auth_headers(self) -> dict:
        """Get the auth headers."""
        resp = self._get_workspace_connection_by_name()
        return {
            "category": resp['properties']['category'],
            "target": resp['properties']['target'],
            "key": resp['properties']['credentials']['key'],
            "metadata": resp['properties']['metadata'],
        }

    def _get_workspace_connection_by_name(self) -> dict:
        """Get a workspace connection from the workspace."""
        if hasattr(self.current_workspace._auth, "get_token"):
            bearer_token = self.current_workspace._auth.get_token(
                "https://management.azure.com/.default").token
        else:
            bearer_token = self.current_workspace._auth.token

        endpoint = self.current_workspace.service_context._get_endpoint("api")

        url_list = [
            endpoint,
            "rp/workspaces/subscriptions",
            self.current_workspace.subscription_id,
            "resourcegroups",
            self.current_workspace.resource_group,
            "providers",
            "Microsoft.MachineLearningServices",
            "workspaces",
            self.current_workspace.name,
            "connections",
            self._connection_name,
            "listsecrets?api-version=2023-02-01-preview"
        ]
        response = self._send_post_request('/'.join(url_list), {
            "Authorization": f"Bearer {bearer_token}",
            "content-type": "application/json"
        }, {})

        return response.json()

    def _send_post_request(self, url: str, headers: dict, payload: dict):
        """Send a POST request."""
        with self._create_session_with_retry() as session:
            response = session.post(url, data=json.dumps(payload), headers=headers)
            response.raise_for_status()

        return response

    def _create_session_with_retry(self, retry: int = 3) -> requests.Session:
        """Create requests.session with retry."""
        retry_policy = self._get_retry_policy(num_retry=retry)

        session = requests.Session()
        session.mount("https://", HTTPAdapter(max_retries=retry_policy))
        session.mount("http://", HTTPAdapter(max_retries=retry_policy))
        return session

    def _get_retry_policy(self, num_retry: int = 3) -> Retry:
        """Request retry policy with increasing backoff."""
        status_forcelist = [413, 429, 500, 502, 503, 504]
        backoff_factor = 0.4
        retry_policy = Retry(
            total=num_retry,
            read=num_retry,
            connect=num_retry,
            backoff_factor=backoff_factor,
            status_forcelist=status_forcelist,
            raise_on_status=False
        )
        return retry_policy
