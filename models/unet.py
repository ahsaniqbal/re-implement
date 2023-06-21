"""The module implements UNet.
"""
from typing  import List
import torch
from torch import nn

class EncoderBlock(nn.Module):
    """The module defines the encoder block of UNet.
    
    Attributes:
        model: nn.Sequential, a sequence of modules to be applied to the input.
    """
    def __init__(self, img_size: int, input_channels: int,
                 output_channels: int, kernel_size: int,
                 padding: int) -> None:
        """Initializes the encoder block of the UNet.

        Args:
            img_size: int, the spatial size of the input.
            input_channels: int, the number of input channels.
            output_channels: int, the number of output channels of the model.
            kernel_size: int, the size of the convolution kernel.
            padding: the amount of padding while convolving.
        """
        super().__init__()
        self.model = nn.Sequential(nn.LayerNorm([input_channels, img_size, img_size]),
                                   nn.Conv2d(in_channels=input_channels,
                                             out_channels=output_channels,
                                             kernel_size=kernel_size,
                                             padding=padding),
                                   nn.SiLU(inplace=True),
                                   nn.Conv2d(in_channels=output_channels,
                                             out_channels=output_channels,
                                             kernel_size=kernel_size,
                                             padding=padding),
                                   nn.SiLU(inplace=True),
                                   nn.MaxPool2d(kernel_size=2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward propogates the input through model/block.
        
        Args:
            x: torch.Tensor, the input to the model of size N, C, H, W.

        Returns:
            The method returns the output tensor of size N, D, H//2, W//2.
        """
        return self.model(x)


class DecoderBlock(nn.Module):
    """The module defines the decoder block of UNet.
    
    Attributes:
        model: nn.Sequential, a sequence of modules to be applied to the input.
    """
    def __init__(self, img_size: int, input_channels: int,
                 output_channels: int, kernel_size: int,
                 padding: int) -> None:
        """Initializes the decoder block.
        
        Args:
            img_size: int, the spatial size of input.
            input_channels: int, the number of input channels.
            output_channels: int, the number of output channels.
            kernel_size: int, size of the convolutional kernel.
            padding: int, amount of padding while convolving.

        Returns:
            the model doesn't return anything.
        """
        super().__init__()
        self.model = nn.Sequential(nn.LayerNorm([input_channels, img_size, img_size]),
                                   nn.Conv2d(in_channels=input_channels,
                                             out_channels=output_channels,
                                             kernel_size=kernel_size,
                                             padding=padding),
                                   nn.SiLU(inplace=True),
                                   nn.Conv2d(in_channels=output_channels,
                                             out_channels=output_channels,
                                             kernel_size=kernel_size,
                                             padding=padding),
                                   nn.SiLU(inplace=True),
                                   nn.Upsample(scale_factor=2, mode='bilinear'))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward propogates the input through the block.
        
        Args:
            x: torch.Tensor, the input to the block of size N, C, H, W.

        Returns:
            A torch.Tensor of size N, D, Hx2, Wx2.
        """
        return self.model(x)

class UNet(nn.Module):
    """This module defines UNet model.

    The module encodes the given image of size C, H, W to 1024, H//32, W//32.
    Then decodes the encoded image to C, H, W.

    Attributes:
        encoder_blocks: nn.ModuleList, a list of encoder blocks.
        bottle_neck: nn.Sequential, sequence of modules called bottle neck.
        decoder_blocks: nn.ModuleList, a list of decoder blocks.
        out: nn.Sequential, sequence of modules to generate the output of the model.
    """
    def __init__(self, img_size: int) -> None:
        """Initializes the UNet model.

        Args:
            img_size: int, the size of input images to the model.

        Returns:
            the method doesn't return anything.
        """
        super().__init__()
        self.encoder_blocks = nn.ModuleList([
            # 256x256x3
            EncoderBlock(img_size=img_size,
                         input_channels=3,
                         output_channels=64,
                         kernel_size=3,
                         padding=1),
            # 128x128x64
            EncoderBlock(img_size=img_size//2,
                         input_channels=64,
                         output_channels=128,
                         kernel_size=3,
                         padding=1),
            # 64x64x128
            EncoderBlock(img_size=img_size//4,
                         input_channels=128,
                         output_channels=256,
                         kernel_size=3,
                         padding=1),
            # 32x32x256
            EncoderBlock(img_size=img_size//8,
                         input_channels=256,
                         output_channels=512,
                         kernel_size=3,
                         padding=1),
        ])

        # 16x16x512
        self.bottle_neck = nn.Sequential(
            EncoderBlock(img_size=img_size//16,
                         input_channels=512,
                         output_channels=1024,
                         kernel_size=3,
                         padding=1),
            nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=1),
            nn.SiLU(inplace=True)
        )


        self.decoder_blocks = nn.ModuleList([
            # 8x8x1024
            DecoderBlock(img_size=img_size//32,
                         input_channels=1024,
                         output_channels=256,
                         kernel_size=3,
                         padding=1),
            # 16x16x512
            DecoderBlock(img_size=img_size//16,
                         input_channels=512,
                         output_channels=128,
                         kernel_size=3,
                         padding=1),
            # 32x32x256
            DecoderBlock(img_size=img_size//8,
                         input_channels=256,
                         output_channels=64,
                         kernel_size=3,
                         padding=1),
            # 64x64x128
            DecoderBlock(img_size=img_size//4,
                         input_channels=128,
                         output_channels=32,
                         kernel_size=3,
                         padding=1)
        ])

        self.out = nn.Sequential(
            # 128x128x32
            DecoderBlock(img_size=img_size//2,
                         input_channels=32,
                         output_channels=3,
                         kernel_size=3,
                         padding=1),
            # 256x256x3
            nn.Conv2d(in_channels=3, out_channels=3, kernel_size=1),
            nn.SiLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward propogates the input through UNet.

        Args:
            x: torch.Tensor, a batch of images of size N, C, H, W.
                x could also be batch of random noise drawn from N(0, 1).

        Returns:
            returns a batch of tensor of size (N, C, H, W).
        """
        encoder_outputs: List[torch.Tensor] = []
        for encoder_block in self.encoder_blocks:
            x = encoder_block(x)
            encoder_outputs.append(x)

        x = self.bottle_neck(x)

        index = len(self.encoder_blocks) - 1
        for decoder_block in self.decoder_blocks:
            x = decoder_block(torch.cat([x, encoder_outputs[index]], axis=1))
            index -= 1

        x = self.out(x)
        return x
        