"""Module implements vision transfromer model from https://arxiv.org/pdf/2010.11929v2.pdf.

The module implements the vision transformer model. 
The module implements all the components of the vision transformer model.
Including VisionTransformer, TransformerEncoder, MultiHeadSelfAttention, PatchEmbedding.

Typical usage example:

  model = VisionTransformer(img_size=224, in_channels=3, patch_size=16, embed_size=16*16*3,
                              num_layers=12, num_heads=8, head_dim=256, mlp_dim=512,
                              num_classes=10, drop_prob=0.0)
  prediction = model(images)
"""

from typing import Tuple, Dict

import torch
from torch import nn
from einops.layers import torch as einops_torch


class PatchEmbedding(nn.Module):
    """This module divides the image into a sequence of patches and learns embedding for each patch.

    Divides the input image into a sequence of rectangular (S x S) patches, S=patch_size.
    Expects a tensor of size (N, C, H, W), where N, C, H, W is batch, channels, height, and width.
    Returns a tensor of size (N, P, E), where N, P, E is batch, num_patches, and embedding size.

    Attributes:
        unfold_op: nn.Unfold, divides the image into a set of patches of size (S x S), S=patch_size.
        linear_op: nn.Linear, projects patch feature of size (S x S x C) to embedding size.
        cls_token: nn.Parameter, learnable classification token of size (1, 1, E), E=embedding_size.
        postional_encodings: nn.Parameter, learnable positional encoding of size (P + 1, E).
            P=num_patches, E=embedding_size, +1 for class token.
    """
    def __init__(
        self,
        in_channels: int = 3,
        patch_size: int = 16,
        img_size: int = 224,
        emb_size: int = 768,
    ) -> None:
        """Initializes the PatchEmbedding module.

        Args:
            in_channel: int, the number of input image channels.
            patch_size: int, the size of each patch.
            img_size: int, the size of the input image.
            emb_size: int, the size of the output embedding of each patch.

        Returns:
            The method returns nothing.

        Raises:
            ValueError: If image_size is not perfectly divisible by patch_size.
        """
        super().__init__()
        if img_size % patch_size != 0:
            raise ValueError("The image size should be divisible by patch size.")
        self.unfold_op = nn.Unfold(kernel_size=patch_size, stride=patch_size)
        self.linear_op = nn.Linear(in_channels * patch_size * patch_size, emb_size)
        self.cls_token = nn.Parameter(torch.randn(1, 1, emb_size))
        self.postional_encodings = nn.Parameter(
            torch.randn((img_size // patch_size) ** 2 + 1, emb_size)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward propagates the input tensor through the module.

        Args:
            x: torch.Tensor, input tensor of size (B, C, H, W).
                Where N, C, H, W is batch, channels, height, and width.

        Returns:
            The output tensor of type torch.Tensor of size (B, P + 1, E).
                Where B, P, E is batch, num_patches, embedding_size.
        """
        b, _, _, _ = x.shape
        x = self.unfold_op(x)
        x = torch.permute(x, (0, 2, 1))
        x = self.linear_op(x)
        cls_tokens = self.cls_token.repeat((b, 1, 1))
        x = torch.cat([cls_tokens, x], dim=1)
        return x + self.postional_encodings


class MultiHeadSelfAttention(nn.Module):
    """The module implements multi-head self attention from https://arxiv.org/abs/1706.03762.
    
    Expects a tensor of size (N, P, E), where N, P, E is batch, num_patches, and embedding size.
    Returns a tensor of size (N, P, E), where N, P, E is batch, num_patches, and embedding size.

    Attributes:
        scale: float, the scale factor before applying softmax.
        key_layer: nn.Sequential(nn.Linear, einops.Rearrange), embedding for the keys of all heads.
        query_layer: nn.Sequential(nn.Linear, einops.Rearrange), embedding for the queries of all
            heads.
        value_layer: nn.Sequential(nn.Linear, einops.Rearrange), embedding for the values of all
            heads.
        attention_scores: nn.Softmax, calculates attention scores.
        out_layer: nn.Sequential(einops.Rearrange, nn.Linear, nn.Dropout), the output layer of the
            module.
    """
    def __init__(
        self, in_dim: int, att_dim: int, num_heads: int, drop_prob: float
    ) -> None:
        """Initializes the MultiHeadSelfAttention module.
        
        Args:
            in_dim: int, the input dimension of the module.
            att_dim: int, the dimension of query/key/value of each head.
            num_head: int, the number of head in the module.
            drop_prop: float, the dropout probability.

        Returns:
            The method returns nothing.
        """
        super().__init__()
        self.scale = att_dim**-0.5

        self.key_layer = self.get_key_query_value_layer(
            in_dim, att_dim * num_heads, num_heads
        )
        self.query_layer = self.get_key_query_value_layer(
            in_dim, att_dim * num_heads, num_heads
        )
        self.value_layer = self.get_key_query_value_layer(
            in_dim, att_dim * num_heads, num_heads
        )
        self.attention_scores = nn.Softmax(dim=-1)

        self.out_layer = nn.Sequential(
            einops_torch.Rearrange(
                "batch_size num_heads patches out_dim -> batch_size patches (num_heads out_dim)"
            ),
            nn.Linear(att_dim * num_heads, in_dim),
            nn.Dropout(drop_prob),
        )

    def get_key_query_value_layer(
        self, in_dim: int, out_dim: int, num_heads
    ) -> torch.nn.Module:
        """Builds an embedding layer for query/key/value for each head.
        
        Args:
            in_dim: int, input dimension for key/query/value layer.
            out_dim: int, the out dimension of the layer, should be attension_dim x num_heads.
            num_heads: int, the number of heads in the module.

        Returns:
            returns a nn.Sequential(nn.Linear, einops.Rearrange).
        """
        return nn.Sequential(
            nn.Linear(in_dim, out_dim, bias=False),
            einops_torch.Rearrange(
                "batch_size patches (num_heads out_dim) -> batch_size num_heads patches out_dim",
                num_heads=num_heads,
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward propagates the input tensor through the module.
        
        Args:
            x: torch.Tensor, an input tensor of size (B, P, E), where B, P, E is batch, num_patches,
                and embedding_size.

        Returns:
            returns the output as torch.Tensor of size (B, P, E), where B, P, E is batch,
                num_patches, and embedding_size.
        """
        key = self.key_layer(x)
        query = self.query_layer(x)
        value = self.value_layer(x)

        weights = self.attention_scores(
            torch.matmul(query, key.transpose(-1, -2)) * self.scale
        )
        value = torch.matmul(weights, value)
        out = self.out_layer(value)
        return out, weights


class TransformerEncoder(nn.Module):
    """The encoder module from the vision transformer https://arxiv.org/pdf/2010.11929v2.pdf.
    
    Expects a tensor of size (N, P, E), where N, P, E is batch, num_patches, and embedding size.
    Returns a tensor of size (N, P, E), where N, P, E is batch, num_patches, and embedding size.

    Attributes:
        sa_blocks: nn.ModuleList, a list of the self attention modules blocks, of size num_layers.
        mlp_blocks: nn.ModuleList, a list of the mlp blocks, of size num_layers.
    """
    def __init__(
        self,
        embed_dim: int,
        num_layers: int,
        num_heads: int,
        head_dim: int,
        mlp_dim: int,
        drop_prob: float,
    ) -> None:
        """Initializes the TransformerEncoder module.
        
        Args:
            embed_dim: int, the embedding dimension of input/output.
            num_layers: int, the number of sa_blocks/mlp_blocks.
            num_heads: int, the number heads in the MultiHeadSelfAttention module.
            head_dim: int, the attention_dim in the MultiHeadSelfAttention module.
            mlp_dim: int, the mlp dimension in the mlp blocks.
            drop_prob: float, the drop out probability.

        Returns:
            The method returns nothing.
        """
        super().__init__()

        self.sa_blocks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(embed_dim),
                    MultiHeadSelfAttention(
                        in_dim=embed_dim,
                        att_dim=head_dim,
                        num_heads=num_heads,
                        drop_prob=drop_prob,
                    ),
                )
                for _ in range(num_layers)
            ]
        )
        self.mlp_blocks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(embed_dim),
                    nn.Linear(embed_dim, mlp_dim),
                    nn.GELU(),
                    nn.Dropout(drop_prob),
                    nn.Linear(mlp_dim, embed_dim),
                    nn.GELU(),
                    nn.Dropout(drop_prob),
                )
                for _ in range(num_layers)
            ]
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[int, torch.Tensor]]:
        """Forward propagates the input tensor through the module.
        
        Args:
            x: torch.Tensor, an input tensor of size (B, P, E), where B, P, E is batch, num_patches,
                and embedding_size.

        Returns:
            A tuples T(torch.Tensor, Dict[int, torch.Tensor]).
                Where T[0] is the output of the module, and
                T[1] is the attention weights of each self attention block.
        """
        idx = 0
        att_weights = {}
        for sa_block, mlp_block in zip(self.sa_blocks, self.mlp_blocks):
            y, weights = sa_block(x)
            x = x + y
            x = x + mlp_block(x)
            att_weights[idx] = weights
            idx += 1
        return x, att_weights


class VisionTransformer(nn.Module):
    """The vision transformer from https://arxiv.org/pdf/2010.11929v2.pdf.
    
    Expects a tensor of size (N, C, H, W), where N, C, H, W is batch, channels, height, and width.
    Returns a tensor of size (N, O), where N, O is batch, number of output classes.

    Attributes:
        patch_emb_layer: PatchEmbedding, the patch embedding module/layer.
        transformer_encoder: TransformerEncoder, the encoder module/layer.
        out_layer: nn.Linear, the classification layer.
    """
    def __init__(
        self,
        img_size: int,
        in_channels: int,
        patch_size: int,
        embed_size: int,
        num_layers: int,
        num_heads: int,
        head_dim: int,
        mlp_dim: int,
        num_classes: int,
        drop_prob: float,
    ) -> None:
        """Initializes the VisionTransformer module.
        
        Args:
            img_size: int, the size of the input image.
            in_channels: int, the number of channels of input image.
            patch_size: int, the size of each patch.
            embed_size: int, the embedding size.
            num_layers: int, the number of layers in the encoder block.
            num_heads: int, the number of heads in the MultiHeadSelfAttention module.
            head_dim: int, the attention dimension in the MultiHeadSelfAttention module.
            mlp_dim: int, the dimension in the mlp block.
            num_classes: int, the number of classes.
            drop_prob: float, the drop out probability.

        Returns:
            The method returns nothing.
        """
        super().__init__()

        self.patch_emb_layer = PatchEmbedding(
            in_channels=in_channels,
            patch_size=patch_size,
            img_size=img_size,
            emb_size=embed_size,
        )
        self.transformer_encoder = TransformerEncoder(
            embed_dim=embed_size,
            num_layers=num_layers,
            num_heads=num_heads,
            head_dim=head_dim,
            mlp_dim=mlp_dim,
            drop_prob=drop_prob,
        )
        self.out_layer = nn.Linear(embed_size, num_classes)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[int, torch.Tensor]]:
        """Forward propagates the input tensor through the module.
        
        Args:
            x: torch.Tensor, tensor of size (N, C, H, W), where N, C, H, W is batch, channels,
                height, and width.

        Returns:
            A tuples T(torch.Tensor, Dict[int, torch.Tensor]).
                Where T[0] is the prediction of the module.
                T[1] is the attention weights of each self attention block.
        """
        x = self.patch_emb_layer(x)
        x, att_weights = self.transformer_encoder(x)

        cls_token = x[:, 0, :]
        out = self.out_layer(cls_token)
        return out, att_weights
