# Ingrain Server

![Test Status](https://github.com/OwenPendrighElliott/ingrain_server/actions/workflows/tests.yml/badge.svg)
![Build Status](https://github.com/OwenPendrighElliott/ingrain_server/actions/workflows/test-build.yml/badge.svg)
![Style](https://github.com/OwenPendrighElliott/ingrain_server/actions/workflows/code-style-check.yml/badge.svg)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![GitHub release](https://img.shields.io/github/v/release/owenpendrighelliott/ingrain_server)](https://github.com/owenpendrighelliott/ingrain_server)


> __This project is under active development, expect breaking changes between versions until v1.0.0.__

This is a wrapper for [Triton Inference Server](https://developer.nvidia.com/triton-inference-server) that makes using it with sentence transformers and open CLIP models easy.

Ingrain server works in tandem with Triton to automate much of the process for serving OpenCLIP, sentence transformers, and timm models. 

The easiest way to get started is with a docker compose file, which will run Triton and the Ingrain server.

```yml
services:
  ingrain-models:
    image: owenelliottdev/ingrain-models:latest
    container_name: ingrain-models
    ports:
      - "8687:8687"
    environment:
      - TRITON_GRPC_URL=triton:8001
      - MAX_BATCH_SIZE=16
      - MODEL_INSTANCES=1
      - INSTANCE_KIND=KIND_GPU # Change to KIND_CPU if using a CPU
    depends_on:
      - triton
    volumes:
      - ./model_repository:/app/model_repository 
      - ${HOME}/.cache/huggingface:/app/model_cache/
  ingrain-inference:
    image: owenelliottdev/ingrain-inference:latest
    container_name: ingrain-inference
    ports:
      - "8686:8686"
    environment:
      - TRITON_GRPC_URL=triton:8001
    depends_on:
      - triton
    volumes:
      - ./model_repository:/app/model_repository 
  triton:
    image: nvcr.io/nvidia/tritonserver:25.09-py3
    container_name: triton
    runtime: nvidia # Remove if using a CPU
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
      - LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libjemalloc.so
    shm_size: "256m"
    command: >
      tritonserver
      --model-repository=/models
      --model-control-mode=explicit
    ports:
      - "8000:8000"
      - "8001:8001"
      - "8002:8002"
    volumes:
      - ./model_repository:/models
    restart:
      unless-stopped
```

To run without a GPU comment out the `runtime: nvidia` in the triton container.

Spin up the server with:

```bash
docker compose up
```

## What does it do?

This server handles all the model loading, ONNX conversion, memory management, parallelisation, dynamic batching, input pre-processing, image handling, and other complexities of running a model in production. The API is very simple but lets you serve models in a performant manner.

Open CLIP models and sentence transformers are both converted to ONNX and served by Triton. The server can handle multiple models at once.

## How does it perform?

It retains all the performance of Triton. On 12 cores at 4.3 GHz with a 2080 SUPER 8GB card running in Docker using WSL2, it can serve `intfloat/e5-small-v2` to 100 clients at ~1310 QPS, or `intfloat/e5-base-v2` to 100 clients at ~1158 QPS.

## How compatible is it?

Most models work out of the box, it is intractable to test every sentence transformers model and every CLIP models but most main architectures are tested and work. If you have a model that doesn't work, please open an issue.

## Usage

The easiest way to get started is the use the optimised Python Client:

```bash
pip install ingrain
```

```python
import ingrain

ingrn = ingrain.Client()

model = ingrn.load_sentence_transformer_model(name="intfloat/e5-small-v2")

response = model.infer_text(text=["I am a sentence.", "I am another sentence.", "I am a third sentence."])

print(f"Processing Time (ms): {response['processingTimeMs']}")
print(f"Text Embeddings: {response['embeddings']}")
```

You can also have the embeddings automatically be returned as a numpy array:

```python
import ingrain

ingrn = ingrain.Client(return_numpy=True)

model = client.load_sentence_transformer_model(name="intfloat/e5-small-v2")

response = model.infer_text(text=["I am a sentence.", "I am another sentence.", "I am a third sentence."])

print(type(response['embeddings']))
```

## Build Container Locally

You can build and run Ingrain with docker locally by using the included docker compose file.

```bash
docker compose up --build
```

## Performance test

You can run the benchmark script to test the performance of the server:

Install the python client:
```bash
pip install ingrain
```

Run the benchmark script:
```bash
python benchmark.py
```

It will output some metrics about the inference speed of the server.

```
{"message":"Model intfloat/e5-small-v2 is already loaded."}
Benchmark results:
Concurrent threads: 500
Requests per thread: 20
Total requests: 10000
Total benchmark time: 9.31 seconds
QPS: 1074.66
Mean response time: 0.3595 seconds
Median response time: 0.3495 seconds
Standard deviation of response times: 0.1174 seconds
Mean inference time: 235.5968 ms
Median inference time: 227.6743 ms
Standard deviation of inference times: 84.8669 ms
```

## Development Setup

Requires Docker and Python to be installed.

### Install dependencies

This project uses `uv` for development, so you need to install it first.

Setup project:
```bash
uv sync --dev
```

### Run the Triton Inference Server

```bash
bash run_triton_server_dev.sh
```

### Run the FastAPI server

```bash
uv run uvicorn --app-dir ingrain_inference_server inference_server:app --host 127.0.0.1 --port 8686 --reload
```

```bash
uv run uvicorn --app-dir ingrain_model_server model_server:app --host 127.0.0.1 --port 8687 --reload
```

### Testing

#### Unit tests

```bash
uv run pytest
```

#### Integration tests and unit tests

```bash
uv run pytest --integration
```
