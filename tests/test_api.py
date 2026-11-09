from fastapi.testclient import TestClient

from src.app import app

# No "with": startup (model loading) is skipped, so tests run without the .pt file
client = TestClient(app)


def test_rejects_text_file():
    r = client.post("/predict", files={"file": ("a.txt", b"hello", "text/plain")})
    assert r.status_code == 400


def test_rejects_corrupted_image():
    r = client.post("/predict", files={"file": ("a.png", b"not an image", "image/png")})
    assert r.status_code == 400


def test_rejects_large_file():
    big = b"0" * 5_000_001
    r = client.post("/predict", files={"file": ("a.png", big, "image/png")})
    assert r.status_code == 413