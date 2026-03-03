"""Shared test fixtures."""

import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

# Set policy path before importing the app
POLICY_PATH = Path(__file__).parent.parent / "policies" / "finance.json"
os.environ["POLICY_PATH"] = str(POLICY_PATH)


@pytest.fixture
def client():
    from intentguard.server import app
    with TestClient(app) as c:
        yield c


@pytest.fixture
def finance_policy():
    from intentguard.policy import Policy
    return Policy.from_file(POLICY_PATH)
