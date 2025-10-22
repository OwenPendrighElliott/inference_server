import pytest
import requests
import numpy as np
from PIL import Image
from io import BytesIO
import base64
import open_clip
from sentence_transformers import SentenceTransformer


INFERENCE_BASE_URL = "http://127.0.0.1:8686"
MODEL_BASE_URL = "http://127.0.0.1:8687"

# test models
SENTENCE_TRANSFORMER_MODEL = "intfloat/e5-small-v2"
OPENCLIP_MODEL = "hf-hub:laion/CLIP-ViT-B-32-laion2B-s34B-b79K"


def check_server_running():
    try:
        response = requests.get(f"{INFERENCE_BASE_URL}/health")
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        raise RuntimeError(
            "Server is not running on localhost:8686. Please start the server and try again."
        ) from e


def load_openclip_model():
    response = requests.post(
        f"{MODEL_BASE_URL}/load_model",
        json={"name": OPENCLIP_MODEL, "library": "open_clip"},
    )
    response.raise_for_status()


def load_sentence_transformer_model():
    response = requests.post(
        f"{MODEL_BASE_URL}/load_model",
        json={"name": SENTENCE_TRANSFORMER_MODEL, "library": "sentence_transformers"},
    )
    response.raise_for_status()


@pytest.mark.integration
def test_health():
    check_server_running()
    response = requests.get(f"{INFERENCE_BASE_URL}/health")
    assert response.status_code == 200
    assert response.json() == {"message": "The inference server is running."}


@pytest.mark.integration
def test_infer_text():
    check_server_running()
    load_sentence_transformer_model()
    test_text = "This is a test sentence."
    response = requests.post(
        f"{INFERENCE_BASE_URL}/embed_text",
        json={"name": SENTENCE_TRANSFORMER_MODEL, "text": test_text},
    )
    assert response.status_code == 200
    assert "embeddings" in response.json()
    assert "processingTimeMs" in response.json()

    ingrain_embeddings = response.json()["embeddings"]

    model = SentenceTransformer(SENTENCE_TRANSFORMER_MODEL)
    model.eval()
    model_embeddings = model.encode([test_text])

    print("Max abs diff:", np.max(np.abs(ingrain_embeddings - model_embeddings)))
    print("Mean abs diff:", np.mean(np.abs(ingrain_embeddings - model_embeddings)))
    print(
        "Allclose (1e-5):", np.allclose(ingrain_embeddings, model_embeddings, atol=1e-5)
    )
    print(
        "Allclose (1e-4):", np.allclose(ingrain_embeddings, model_embeddings, atol=1e-4)
    )
    assert np.allclose(ingrain_embeddings, model_embeddings, atol=1e-5)


@pytest.mark.integration
def test_infer_openclip_text():
    check_server_running()
    load_openclip_model()
    test_text = "This is a test sentence."
    response = requests.post(
        f"{INFERENCE_BASE_URL}/embed_text",
        json={
            "name": OPENCLIP_MODEL,
            "text": test_text,
        },
    )
    assert response.status_code == 200
    assert "embeddings" in response.json()
    assert "processingTimeMs" in response.json()

    ingrain_embeddings = response.json()["embeddings"]

    model, _, _ = open_clip.create_model_and_transforms(OPENCLIP_MODEL)
    tokenizer = open_clip.get_tokenizer(OPENCLIP_MODEL)
    tokens = tokenizer([test_text])
    model_embeddings = model.encode_text(tokens)
    model_embeddings /= model_embeddings.norm(dim=-1, keepdim=True)
    assert np.allclose(
        ingrain_embeddings, model_embeddings.detach().cpu().numpy(), atol=1e-5
    )


@pytest.mark.integration
def test_infer_openclip_image():
    check_server_running()
    load_openclip_model()

    test_image = "data:image/jpeg;base64,iVBORw0KGgoAAAANSUhEUgAAAOAAAADgCAIAAACVT/22AAACkElEQVR4nOzUMQ0CYRgEUQ5wgwAUnA+EUKKJBkeowAHVJd/kz3sKtpjs9fH9nDjO/npOT1jKeXoA/CNQ0gRKmkBJEyhpAiVNoKQJlDSBkiZQ0gRKmkBJEyhpAiVNoKQJlDSBkiZQ0gRKmkBJEyhpAiVNoKQJlDSBkiZQ0gRKmkBJEyhpAiVNoKQJlDSBkiZQ0gRKmkBJEyhpAiVNoKQJlDSBkiZQ0gRKmkBJEyhpAiVNoKQJlDSBkiZQ0gRKmkBJEyhpAiVNoKQJlDSBkiZQ0gRKmkBJEyhpAiVNoKQJlDSBkiZQ0gRKmkBJEyhpAiVNoKQJlDSBkiZQ0gRKmkBJEyhpAiVNoKQJlDSBkiZQ0gRKmkBJEyhpAiVNoKQJlDSBkiZQ0gRKmkBJEyhpAiVNoKQJlDSBkiZQ0gRKmkBJEyhpAiVNoKQJlDSBkiZQ0gRKmkBJEyhpAiVNoKRtl/t7esNSbvs2PWEpHpQ0gZImUNIESppASRMoaQIlTaCkCZQ0gZImUNIESppASRMoaQIlTaCkCZQ0gZImUNIESppASRMoaQIlTaCkCZQ0gZImUNIESppASRMoaQIlTaCkCZQ0gZImUNIESppASRMoaQIlTaCkCZQ0gZImUNIESppASRMoaQIlTaCkCZQ0gZImUNIESppASRMoaQIlTaCkCZQ0gZImUNIESppASRMoaQIlTaCkCZQ0gZImUNIESppASRMoaQIlTaCkCZQ0gZImUNIESppASRMoaQIlTaCkCZQ0gZImUNIESppASRMoaQIlTaCkCZQ0gZImUNIESppASRMoaQIlTaCkCZQ0gZImUNIESppASRMoaQIlTaCkCZQ0gZImUNIESppASRMoaQIl7RcAAP//iL8GbQ2nM1wAAAAASUVORK5CYII="
    response = requests.post(
        f"{INFERENCE_BASE_URL}/embed_image",
        json={
            "name": OPENCLIP_MODEL,
            "image": test_image,
        },
    )
    assert response.status_code == 200
    assert "embeddings" in response.json()
    assert "processingTimeMs" in response.json()

    ingrain_embeddings = response.json()["embeddings"]

    model, _, preprocess = open_clip.create_model_and_transforms(OPENCLIP_MODEL)
    model.eval()
    img = Image.open(BytesIO(base64.b64decode(test_image.split(",")[1]))).convert("RGB")
    processed_im = preprocess(img).unsqueeze(0)
    model_embeddings = model.encode_image(processed_im)
    model_embeddings /= model_embeddings.norm(dim=-1, keepdim=True)
    assert np.allclose(
        ingrain_embeddings, model_embeddings.detach().cpu().numpy(), atol=1e-5
    )
