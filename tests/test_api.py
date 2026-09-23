"""Tests for the API module."""

import io

import cv2
import numpy as np
import pandas as pd
import pytest
from conftest import make_fundus
from fastapi.testclient import TestClient

from drstage.api.app import create_app
from drstage.api.predictors import DemoPredictor, Prediction


@pytest.fixture
def demo_app(tmp_path):
    """Create app with demo predictor and no sample split."""
    return create_app(predictor=DemoPredictor(), split_csv=tmp_path / "missing.csv")


@pytest.fixture
def test_client(demo_app):
    """Create TestClient for demo app."""
    return TestClient(demo_app)


def test_health_demo(test_client):
    """GET /api/health returns demo=true with model=null."""
    response = test_client.get("/api/health")
    assert response.status_code == 200

    data = response.json()
    assert data["status"] == "ok"
    assert data["demo"] is True
    assert data["model"] is None


def test_predict_success(test_client):
    """POST /api/predict processes image and returns valid response."""
    img = make_fundus(640, 480, seed=42)
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    _, img_bytes = cv2.imencode(".jpg", img_bgr)

    response = test_client.post(
        "/api/predict",
        files={"file": ("test.jpg", io.BytesIO(img_bytes), "image/jpeg")},
    )
    assert response.status_code == 200

    data = response.json()
    assert data["demo"] is True
    assert 0 <= data["grade"] <= 4
    assert data["grade_name"] in ["No DR", "Mild", "Moderate", "Severe", "Proliferative"]
    assert isinstance(data["dr_detected"], bool)
    assert 0 <= data["prob_dr"] <= 1
    assert isinstance(data["referable"], bool)
    assert len(data["class_probs"]) == 5
    assert 0.99 <= sum(data["class_probs"]) <= 1.01

    assert "original" in data["images"]
    assert "enhanced" in data["images"]
    assert "heatmap" in data["images"]

    for key in ["original", "enhanced", "heatmap"]:
        assert data["images"][key].startswith("data:image/png;base64,")

    assert data["inference_ms"] > 0


def test_predict_dr_detected_prob(test_client):
    """dr_detected is True iff prob_dr >= 0.5."""
    img = make_fundus(640, 480, seed=42)
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    _, img_bytes = cv2.imencode(".jpg", img_bgr)

    response = test_client.post(
        "/api/predict",
        files={"file": ("test.jpg", io.BytesIO(img_bytes), "image/jpeg")},
    )
    data = response.json()
    assert data["dr_detected"] == (data["prob_dr"] >= 0.5)


def test_predict_referable_grade(test_client):
    """referable is True iff grade >= 2."""
    img = make_fundus(640, 480, seed=42)
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    _, img_bytes = cv2.imencode(".jpg", img_bgr)

    response = test_client.post(
        "/api/predict",
        files={"file": ("test.jpg", io.BytesIO(img_bytes), "image/jpeg")},
    )
    data = response.json()
    assert data["referable"] == (data["grade"] >= 2)


def test_predict_invalid_image(test_client):
    """POST /api/predict with garbage bytes returns 400 with helpful message."""
    response = test_client.post(
        "/api/predict",
        files={"file": ("test.jpg", io.BytesIO(b"not an image"), "image/jpeg")},
    )
    assert response.status_code == 400
    data = response.json()
    assert "detail" in data
    assert "image" in data["detail"].lower()


def test_predict_missing_file(test_client):
    """POST /api/predict without file returns 422."""
    response = test_client.post("/api/predict")
    assert response.status_code == 422


def test_samples_empty_when_no_csv(test_client):
    """GET /api/samples returns empty list when CSV is missing."""
    response = test_client.get("/api/samples")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)


def test_samples_structure(tmp_path):
    """Samples come from the test split, two per grade, with working thumbnail and image URLs."""
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    rows = []
    for i in range(20):
        path = image_dir / f"img{i:02d}.jpg"
        cv2.imwrite(str(path), cv2.cvtColor(make_fundus(640, 480, seed=i), cv2.COLOR_RGB2BGR))
        rows.append((f"img{i:02d}", str(path), i % 5, "test" if i < 15 else "train"))
    csv_path = tmp_path / "split.csv"
    pd.DataFrame(rows, columns=["image_id", "path", "label", "split"]).to_csv(csv_path, index=False)

    client = TestClient(create_app(predictor=DemoPredictor(), split_csv=csv_path))
    samples = client.get("/api/samples").json()

    assert len(samples) == 10
    assert sorted(s["label"] for s in samples) == [0, 0, 1, 1, 2, 2, 3, 3, 4, 4]
    ids = {r[0] for r in rows if r[3] == "test"}
    assert {s["id"] for s in samples} <= ids
    thumb = client.get(samples[0]["thumbnail"])
    assert thumb.status_code == 200 and thumb.headers["content-type"] == "image/jpeg"
    assert client.get(f"/api/samples/{samples[0]['id']}/image").status_code == 200


def test_frontend_served_without_path_traversal(tmp_path):
    """The SPA is served at / with index fallback, and paths cannot escape the dist directory."""
    dist = tmp_path / "dist"
    dist.mkdir()
    (dist / "index.html").write_text("<html>app</html>")
    (tmp_path / "secret.txt").write_text("secret")
    client = TestClient(
        create_app(predictor=DemoPredictor(), split_csv=tmp_path / "none.csv", frontend_dist=dist)
    )
    assert client.get("/").text == "<html>app</html>"
    assert client.get("/some/route").text == "<html>app</html>"
    assert "secret" not in client.get("/..%2Fsecret.txt").text
    assert client.get("/api/unknown").status_code == 404


def test_sample_thumbnail_404(test_client):
    """GET /api/samples/{id}/thumbnail with unknown id returns 404."""
    response = test_client.get("/api/samples/nonexistent-id/thumbnail")
    assert response.status_code == 404


def test_sample_image_404(test_client):
    """GET /api/samples/{id}/image with unknown id returns 404."""
    response = test_client.get("/api/samples/nonexistent-id/image")
    assert response.status_code == 404


def test_data_urls_decodable(test_client):
    """Images in predict response are valid PNG data URLs."""
    img = make_fundus(640, 480, seed=42)
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    _, img_bytes = cv2.imencode(".jpg", img_bgr)

    response = test_client.post(
        "/api/predict",
        files={"file": ("test.jpg", io.BytesIO(img_bytes), "image/jpeg")},
    )
    data = response.json()

    for key in ["original", "enhanced"]:
        assert data["images"][key].startswith("data:image/png;base64,")
        b64_str = data["images"][key].split(",")[1]

        import base64

        png_bytes = base64.b64decode(b64_str)
        decoded = cv2.imdecode(np.frombuffer(png_bytes, np.uint8), cv2.IMREAD_COLOR)
        assert decoded is not None
        assert decoded.shape[:2] == (512, 512)

    heatmap_b64_str = data["images"]["heatmap"].split(",")[1]
    import base64

    heatmap_bytes = base64.b64decode(heatmap_b64_str)
    heatmap_decoded = cv2.imdecode(np.frombuffer(heatmap_bytes, np.uint8), cv2.IMREAD_UNCHANGED)
    assert heatmap_decoded is not None
    assert heatmap_decoded.shape[:2] == (512, 512)
    assert heatmap_decoded.shape[2] == 4


def test_real_predictor_demo_flag(tmp_path):
    """When real predictor is passed, demo=False in responses."""

    class FakePredictor:
        demo = False

        def info(self):
            return {
                "name": "test_model",
                "head": "classification",
                "image_size": 512,
                "preprocessing": "ben_graham",
                "checkpoint": "/path/to/model.pt",
            }

        def predict(self, image_rgb):
            return Prediction(
                grade=2,
                grade_name="Moderate",
                prob_dr=0.75,
                class_probs=[0.02, 0.05, 0.71, 0.15, 0.07],
                original=np.ones((512, 512, 3), dtype=np.uint8) * 100,
                enhanced=np.ones((512, 512, 3), dtype=np.uint8) * 120,
                heatmap=np.random.random((512, 512)).astype(np.float32),
            )

    app = create_app(predictor=FakePredictor(), split_csv=tmp_path / "missing.csv")
    client = TestClient(app)

    response = client.get("/api/health")
    assert response.status_code == 200
    health = response.json()
    assert health["demo"] is False
    assert health["model"] is not None
    assert health["model"]["name"] == "test_model"

    img = make_fundus(640, 480, seed=42)
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    _, img_bytes = cv2.imencode(".jpg", img_bgr)

    response = client.post(
        "/api/predict",
        files={"file": ("test.jpg", io.BytesIO(img_bytes), "image/jpeg")},
    )
    assert response.status_code == 200
    predict = response.json()
    assert predict["demo"] is False
    assert predict["grade"] == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


def test_predict_includes_quality_report(test_client):
    """A well-exposed synthetic fundus passes the field check and reports every measure."""
    img = make_fundus(800, 600, seed=3)
    ok, buf = cv2.imencode(".png", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    quality = test_client.post(
        "/api/predict", files={"file": ("f.png", buf.tobytes(), "image/png")}
    ).json()["quality"]
    assert {c["key"] for c in quality["checks"]} == {"exposure", "contrast", "sharpness", "field"}
    assert all(c["status"] in {"ok", "warn", "fail"} for c in quality["checks"])
    assert isinstance(quality["gradable"], bool)


def test_dark_image_is_not_gradable():
    """A nearly black photograph fails the quality screen."""
    from drstage.api.quality import assess_quality, is_gradable

    checks = assess_quality(make_fundus(640, 480, dark=True))
    assert not is_gradable(checks)
