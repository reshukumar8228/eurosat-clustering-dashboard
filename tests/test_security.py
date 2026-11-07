"""
Security regression tests for EuroSAT FastAPI backend:
1. Path traversal prevention on static data endpoint.
2. Arbitrary file read prevention on preset inference endpoint.
3. File extension and payload restrictions on image uploads.
"""

import os
import io
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from backend.api import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_api_health(client):
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_static_data_path_traversal_blocked(client):
    # Attempt directory traversal to read config.yaml
    response = client.get("/static/data/..%2f..%2f/config.yaml")
    # Must be 403 (blocked) or 404 (not found), NEVER 200 with yaml content
    assert response.status_code in (403, 404)


def test_static_data_valid_access(client):
    # Access a valid EuroSAT image
    response = client.get("/static/data/Highway/Highway_1.jpg")
    assert response.status_code == 200
    assert "image" in response.headers.get("content-type", "")


def test_predict_preset_path_traversal_blocked(client):
    # Attempt to pass an arbitrary server file as preset_path
    response = client.post("/api/predict", data={"preset_path": "config.yaml"})
    assert response.status_code == 400
    assert "Invalid or unauthorized preset path" in response.json().get("error", "")

    response2 = client.post("/api/predict", data={"preset_path": "../pipeline.py"})
    assert response2.status_code == 400


def test_predict_invalid_extension_blocked(client):
    # Attempt to upload an executable or script
    fake_file = io.BytesIO(b"echo 'malicious payload'")
    response = client.post(
        "/api/predict",
        files={"file": ("malicious.sh", fake_file, "application/x-sh")}
    )
    assert response.status_code == 400
    assert "Unsupported format" in response.json().get("error", "")


def test_predict_valid_image(client):
    # Upload a small valid 64x64 RGB JPEG
    img = Image.new("RGB", (64, 64), color=(100, 150, 200))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)

    response = client.post(
        "/api/predict",
        files={"file": ("test_patch.jpg", buf, "image/jpeg")}
    )
    assert response.status_code == 200
    data = response.json()
    assert "unsupervised" in data
    assert "predicted_cluster" in data["unsupervised"]
    assert "nearest_neighbors" in data
