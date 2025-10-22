from timm.data import MaybeToTensor, MaybePILToTensor
from torch import nn
from torchvision.transforms import Compose, ToTensor


class TimmClassifierWrapper(nn.Module):
    def __init__(self, visual: nn.Module, transforms: nn.Sequential):
        """
        A wrapper that encapsulates the image encoder part of the CLIP model.
        """
        super().__init__()

        self.visual = visual

        self.tensor_transforms = transforms

    def forward(self, image):
        """
        Forward pass to encode image.
        """

        x = self.tensor_transforms(image)
        x = self.visual(x)
        return x
