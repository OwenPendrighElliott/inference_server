import os
import torch
from ingrain_models.models.model_optimisation import (
    generate_tensorrt_config,
    convert_to_float16,
    optimize_onnx_model,
)
from ingrain_common.common import (
    MAX_BATCH_SIZE,
    DYNAMIC_BATCHING,
    MODEL_INSTANCES,
    INSTANCE_KIND,
    TENSORRT_ENABLED,
    FP16_ENABLED,
)
from sentence_transformers import SentenceTransformer


class SentenceTransformerWrapper(torch.nn.Module):
    def __init__(self, model: SentenceTransformer):
        super(SentenceTransformerWrapper, self).__init__()
        self.model = model

    def forward(self, input_ids, attention_mask):
        return self.model({"input_ids": input_ids, "attention_mask": attention_mask})[
            "sentence_embedding"
        ]


def generate_text_sentence_transformer_config(
    cfg_path: str,
    name: str,
    embedding_dim: int,
) -> None:
    config = f"""name: "{name}"
platform: "onnxruntime_onnx"
max_batch_size: {MAX_BATCH_SIZE}
input [
  {{
    name: "input_ids"
    data_type: TYPE_INT64
    dims: [ -1 ]
  }},
  {{
    name: "attention_mask"
    data_type: TYPE_INT64
    dims: [ -1 ]
  }}
]
output [
  {{
    name: "sentence_embedding"
    data_type: TYPE_FP32
    dims: [ {embedding_dim} ]
  }}
]
"""
    if DYNAMIC_BATCHING:
        config += "\n\ndynamic_batching {}"

    if MODEL_INSTANCES > 0 and INSTANCE_KIND:
        config += f"""\n\ninstance_group [
    {{
        count: {MODEL_INSTANCES}
        kind: {INSTANCE_KIND}
    }}
]
        """
    if TENSORRT_ENABLED:
        tensorrt_config = generate_tensorrt_config(
            {"input_ids": [256], "attention_mask": [256]}, "INT64"
        )
        config += f"\n\n{tensorrt_config}"

    with open(os.path.join(cfg_path, "config.pbtxt"), "w") as f:
        f.write(config)


def onnx_transformer_model(
    model: SentenceTransformer, output_path: str
) -> torch.jit.ScriptModule:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    wrapped_model = SentenceTransformerWrapper(model)
    wrapped_model.eval().to(device)
    with torch.inference_mode():
        dummy_input = {
            "input_ids": torch.tensor(
                [[101, 2023, 2003, 1037, 1398, 102]], dtype=torch.int64, device=device
            ),
            "attention_mask": torch.tensor(
                [[1, 1, 1, 1, 1, 1]], dtype=torch.int64, device=device
            ),
        }

        torch.onnx.export(
            model=wrapped_model,
            args=(dummy_input["input_ids"], dummy_input["attention_mask"]),
            f=output_path,
            opset_version=24,
            dynamo=False,
            input_names=["input_ids", "attention_mask"],
            output_names=["sentence_embedding"],
            dynamic_axes={
                "input_ids": {0: "batch_size", 1: "sequence_length"},
                "attention_mask": {0: "batch_size", 1: "sequence_length"},
                "sentence_embedding": {0: "batch_size"},
            },
        )

    optimize_onnx_model(output_path, output_path)

    if FP16_ENABLED and not TENSORRT_ENABLED:
        convert_to_float16(output_path, output_path)
