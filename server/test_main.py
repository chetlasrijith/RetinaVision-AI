"""Backend smoke tests (run without TensorFlow via a stubbed model).

    pip install fastapi uvicorn python-multipart pillow numpy httpx pytest
    pytest server/test_main.py
"""

import io
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
from fastapi.testclient import TestClient
from PIL import Image

import main as app_module


def make_png(size=(200, 200), color=(10, 200, 30)):
    buf = io.BytesIO()
    Image.new("RGB", size, color=color).save(buf, format="PNG")
    buf.seek(0)
    return buf


class StubModel:
    output_shape = (None, 7)

    def __init__(self, vec):
        self._vec = np.asarray(vec, dtype=np.float32)

    def predict(self, batch, verbose=0):
        assert batch.shape == (1, 224, 224, 3), batch.shape
        return np.tile(self._vec, (batch.shape[0], 1))


client = TestClient(app_module.app)


def test_health_without_tf():
    res = client.get("/health")
    assert res.status_code == 200
    assert res.json()["status"] == "ok"
    assert res.json()["classes"] == 7


def test_classes():
    res = client.get("/classes")
    keys = [c["key"] for c in res.json()["classes"]]
    assert keys == app_module.CLASS_NAMES


def test_predict_with_stub():
    app_module._MODEL = StubModel([0.02, 0.03, 0.05, 0.10, 0.05, 0.70, 0.05])
    try:
        res = client.post(
            "/predict", files={"file": ("fundus.png", make_png(), "image/png")}
        )
        assert res.status_code == 200, res.text
        body = res.json()
        assert body["predicted_class"] == "Normal_Fundus"
        assert body["probs_verified"] is True
        assert len(body["scores"]) == 7
        assert body["scores"][0]["key"] == "Normal_Fundus"
        assert abs(sum(s["score"] for s in body["scores"]) - 1.0) < 0.01
    finally:
        app_module._MODEL = None


def test_predict_rejects_bad_extension():
    res = client.post("/predict", files={"file": ("a.txt", b"hello", "text/plain")})
    assert res.status_code == 400


def test_predict_rejects_corrupt_image():
    res = client.post(
        "/predict", files={"file": ("a.png", b"not an image", "image/png")}
    )
    assert res.status_code == 400


def test_predict_warns_on_tiny_image():
    app_module._MODEL = StubModel([0.1, 0.1, 0.1, 0.1, 0.1, 0.4, 0.1])
    try:
        res = client.post(
            "/predict", files={"file": ("t.jpg", make_png((40, 40)), "image/jpeg")}
        )
        assert res.status_code == 200, res.text
        assert len(res.json()["image_warnings"]) >= 1
    finally:
        app_module._MODEL = None
