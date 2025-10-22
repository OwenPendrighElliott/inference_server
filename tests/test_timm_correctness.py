import pytest
import requests
import numpy as np
from PIL import Image
from io import BytesIO
import base64
import timm


INFERENCE_BASE_URL = "http://127.0.0.1:8686"
MODEL_BASE_URL = "http://127.0.0.1:8687"

TIMM_MODEL = "hf_hub:timm/mobilenetv4_hybrid_medium.e200_r256_in12k_ft_in1k"


def check_server_running():
    try:
        response = requests.get(f"{INFERENCE_BASE_URL}/health")
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        raise RuntimeError(
            "Server is not running on localhost:8686. Please start the server and try again."
        ) from e


def load_timm_model():
    response = requests.post(
        f"{MODEL_BASE_URL}/load_model",
        json={"name": TIMM_MODEL, "library": "timm"},
    )
    response.raise_for_status()


def unload_timm_model():
    response = requests.post(
        f"{MODEL_BASE_URL}/unload_model",
        json={"name": TIMM_MODEL},
    )
    response.raise_for_status()


@pytest.mark.integration
def test_health():
    check_server_running()
    response = requests.get(f"{INFERENCE_BASE_URL}/health")
    assert response.status_code == 200
    assert response.json() == {"message": "The inference server is running."}


@pytest.mark.integration
def test_infer_timm_image():
    check_server_running()
    load_timm_model()

    test_image = "data:image/jpeg;base64,iVBORw0KGgoAAAANSUhEUgAAAOAAAADgCAIAAACVT/22AAACkElEQVR4nOzUMQ0CYRgEUQ5wgwAUnA+EUKKJBkeowAHVJd/kz3sKtpjs9fH9nDjO/npOT1jKeXoA/CNQ0gRKmkBJEyhpAiVNoKQJlDSBkiZQ0gRKmkBJEyhpAiVNoKQJlDSBkiZQ0gRKmkBJEyhpAiVNoKQJlDSBkiZQ0gRKmkBJEyhpAiVNoKQJlDSBkiZQ0gRKmkBJEyhpAiVNoKQJlDSBkiZQ0gRKmkBJEyhpAiVNoKQJlDSBkiZQ0gRKmkBJEyhpAiVNoKQJlDSBkiZQ0gRKmkBJEyhpAiVNoKQJlDSBkiZQ0gRKmkBJEyhpAiVNoKQJlDSBkiZQ0gRKmkBJEyhpAiVNoKQJlDSBkiZQ0gRKmkBJEyhpAiVNoKQJlDSBkiZQ0gRKmkBJEyhpAiVNoKQJlDSBkiZQ0gRKmkBJEyhpAiVNoKQJlDSBkiZQ0gRKmkBJEyhpAiVNoKRtl/t7esNSbvs2PWEpHpQ0gZImUNIESppASRMoaQIlTaCkCZQ0gZImUNIESppASRMoaQIlTaCkCZQ0gZImUNIESppASRMoaQIlTaCkCZQ0gZImUNIESppASRMoaQIlTaCkCZQ0gZImUNIESppASRMoaQIlTaCkCZQ0gZImUNIESppASRMoaQIlTaCkCZQ0gZImUNIESppASRMoaQIlTaCkCZQ0gZImUNIESppASRMoaQIlTaCkCZQ0gZImUNIESppASRMoaQIlTaCkCZQ0gZImUNIESppASRMoaQIlTaCkCZQ0gZImUNIESppASRMoaQIlTaCkCZQ0gZImUNIESppASRMoaQIlTaCkCZQ0gZImUNIESppASRMoaQIlTaCkCZQ0gZImUNIESppASRMoaQIl7RcAAP//iL8GbQ2nM1wAAAAASUVORK5CYII="
    response = requests.post(
        f"{INFERENCE_BASE_URL}/classify_image",
        json={
            "name": TIMM_MODEL,
            "image": test_image,
        },
    )
    assert response.status_code == 200
    assert "probabilities" in response.json()
    assert "processingTimeMs" in response.json()

    ingrain_embeddings = response.json()["probabilities"]

    model = timm.create_model(TIMM_MODEL, pretrained=True)
    model.eval()

    data_config = timm.data.resolve_model_data_config(model)
    transforms = timm.data.create_transform(**data_config, is_training=False)

    img = Image.open(BytesIO(base64.b64decode(test_image.split(",")[1]))).convert("RGB")
    processed_im = transforms(img).unsqueeze(0)
    model_embeddings = model(processed_im).softmax(dim=-1)

    assert np.allclose(
        ingrain_embeddings, model_embeddings.detach().cpu().numpy(), atol=1e-5
    )

    unload_timm_model()


@pytest.mark.integration
def test_get_timm_classes():
    check_server_running()
    load_timm_model()

    response = requests.get(
        f"{MODEL_BASE_URL}/model_classification_labels",
        params={"name": TIMM_MODEL},
    )
    assert response.status_code == 200
    assert "labels" in response.json()
    assert len(response.json()["labels"]) == 1000
    assert response.json()["labels"][0] == "tench, Tinca tinca"

    unload_timm_model()
