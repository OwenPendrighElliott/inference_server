import torch
from torch import nn
from torchvision.transforms import Compose, ToTensor
from open_clip.transformer import text_global_pool


class CLIPTextEncoderWrapper(nn.Module):
    def __init__(self, clip_model):
        """
        A wrapper that encapsulates the text encoder part of the CLIP model.
        """
        super().__init__()
        # Extract all text-related components from the CLIP model
        self.token_embedding = clip_model.token_embedding
        self.positional_embedding = clip_model.positional_embedding
        self.transformer = clip_model.transformer
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.text_pool_type = clip_model.text_pool_type
        self.attn_mask = clip_model.attn_mask
        self.context_length = clip_model.context_length
        self.vocab_size = clip_model.vocab_size

    def forward(self, text):
        """
        Forward pass to encode text.
        """
        cast_dtype = self.transformer.get_cast_dtype()

        # Token embedding and adding positional encoding
        x = self.token_embedding(text).to(cast_dtype)  # [batch_size, n_ctx, d_model]
        x = x + self.positional_embedding.to(cast_dtype)
        # Passing through the transformer
        x = self.transformer(x, attn_mask=self.attn_mask)

        # Final layer normalization
        x = self.ln_final(x)

        text = text.to(torch.int32)
        # Pooling to get final representation
        x = text_global_pool(x, text, self.text_pool_type)

        # Project the text embedding if the projection layer exists
        if self.text_projection is not None:
            if isinstance(self.text_projection, nn.Linear):
                x = self.text_projection(x)
            else:
                x = x @ self.text_projection

        return x


class CLIPImageEncoderWrapper(nn.Module):
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
