from torchvision.transforms.functional import InterpolationMode
from torchvision.transforms import Compose, Resize, CenterCrop
import torch
from PIL import Image

from typing import List


def convert_interpolation(interp: InterpolationMode) -> int:
    mapping = {
        InterpolationMode.NEAREST: Image.Resampling.NEAREST,
        InterpolationMode.NEAREST_EXACT: Image.Resampling.NEAREST,
        InterpolationMode.BILINEAR: Image.Resampling.BILINEAR,
        InterpolationMode.BICUBIC: Image.Resampling.BICUBIC,
        InterpolationMode.BOX: Image.Resampling.BOX,
        InterpolationMode.HAMMING: Image.Resampling.HAMMING,
        InterpolationMode.LANCZOS: Image.Resampling.LANCZOS,
    }
    return mapping[interp]


def image_transform_dict_from_torch_transforms(transforms: Compose) -> List[dict]:
    transform_dict = []
    last_resize_size = None
    for transform in transforms.transforms:
        if isinstance(transform, Resize):
            size = transform.size
            if isinstance(size, int):
                size = (size, size)
            transform_data = {
                "type": "ResizeImage",
                "size": size,
                "method": convert_interpolation(transform.interpolation),
            }
            last_resize_size = size
            transform_dict.append(transform_data)
        elif isinstance(transform, CenterCrop):
            size = transform.size
            if isinstance(size, int):
                size = (size, size)

            # skip center crop if it is a no-op (same size as last resize)
            if size == last_resize_size:
                continue

            transform_data = {
                "type": "CenterCropImage",
                "size": size,
            }
            transform_dict.append(transform_data)
        elif hasattr(transform, "__name__") and transform.__name__ == "_convert_to_rgb":
            transform_data = {"type": "ConvertToRGB"}
            transform_dict.append(transform_data)

    return transform_dict


class DynamoFriendlyNormalize(torch.nn.Module):
    """
    TODO: Note that this is not really used ATM because there are too many issues with the dynamo exports in 2.9.0
    and SDPA

    The Normalize provided in torchvision transforms uses `if (std == 0).any()` which dynamo
    cannot convert to ONNX, dynamo is not the default in 2.9.0. This is a small re-write of the Normalize
    transform that gets rid of a lot of the checks torch usually does, because the domain of this application
    is more constrained we don't need the same checks
    """

    def __init__(self, mean: list[float], std: list[float]):
        super().__init__()
        self.mean = mean
        self.std = std

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        mean = torch.as_tensor(self.mean, dtype=dtype, device=x.device)
        std = torch.as_tensor(self.std, dtype=dtype, device=x.device)
        if mean.ndim == 1:
            mean = mean.view(-1, 1, 1)
        if std.ndim == 1:
            std = std.view(-1, 1, 1)
        return x.sub_(mean).div_(std)
