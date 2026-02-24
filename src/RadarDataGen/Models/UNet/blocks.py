# std lib imports
import math
from typing import Optional

# 3 party import
import torch

# projekt imports



class SelfAttentionBlock(torch.nn.Module):
    """
    2D Self-Attention block for U-Net backbones (multi-head).

    Parameters
        channels (int):
            Number of input/output channels C.
        num_heads (int):
            Number of attention heads. Must divide `channels`.
        norm_groups (int):
            GroupNorm groups; must divide `channels` or will fallback to 1 group.
        use_bias (bool):
            Whether to use bias in 1x1 projections.
        zero_init_out (bool):
            If True, zero-initialize the output projection for stability.
    """

    def __init__(
        self,
        in_channels: int,
        num_heads: int = 4,
        norm_groups: int = 8,
        use_bias: bool = True,
        zero_init_out: bool = True,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ) -> None:
        super().__init__()
        
        if in_channels % num_heads != 0:
            raise ValueError(f"channels ({in_channels}) must be divisible by num_heads ({num_heads}).")
        
        self.in_channels = in_channels
        self.num_heads = num_heads
        self.dim_head = in_channels // num_heads

        g = norm_groups if in_channels % norm_groups == 0 else 1
        self.norm = torch.nn.GroupNorm(g, in_channels, device=device)

        self.qkv = torch.nn.Conv2d(in_channels, in_channels * 3, kernel_size=1, bias=use_bias, device=device)   # we can save this in on Cov2d and split later 

        self.proj = torch.nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=use_bias, device=device)  # out conv
        if zero_init_out:
            torch.nn.init.zeros_(self.proj.weight)
            if self.proj.bias is not None:
                torch.nn.init.zeros_(self.proj.bias)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply self-attention over spatial positions.

        Parameters
            x (torch.Tensor):
                Input tensor of shape (B, C, H, W).

        Returns
            torch.Tensor
                Output tensor of shape (B, C, H, W).
        """
        B, C, H, W = x.shape
        h = self.norm(x)

        # QKV: (B, 3C, H, W) -> split into q,k,v each (B, C, H, W)
        qkv = self.qkv(h)
        q, k, v = qkv.chunk(3, dim=1)

        # Reshape to heads: (B, heads, HW, self.dim_head) from (B, C, H, W)
        q = q.view(B, self.num_heads, self.dim_head, H * W).transpose(2, 3)  # (B, heads, HW, self.dim_head)
        k = k.view(B, self.num_heads, self.dim_head, H * W).transpose(2, 3)  # (B, heads, HW, self.dim_head)
        v = v.view(B, self.num_heads, self.dim_head, H * W).transpose(2, 3)  # (B, heads, HW, self.dim_head)

        # Scaled dot-product attention
        scale = 1.0 / math.sqrt(self.dim_head)
        attn = (q @ k.transpose(-2, -1)) * scale            # (B, heads, HW, HW)
        attn = attn.softmax(dim=-1)
        out = attn @ v                                      # (B, heads, HW, self.dim_head)

        out = out.transpose(2, 3).contiguous().view(B, C, H, W)

        out = self.proj(out)    # (B, C, H, W)
        
        return x + out


class Nin(torch.nn.Module):
    """
        Network-in-Network (NiN) layer implemented as a 1x1 convolution.

        This layer performs a pointwise linear projection across the channel dimension
        at each spatial location (height x width), effectively transforming the input
        tensor from `input_dim` channels to `output_dim` channels.

        It is equivalent to a standard 1x1 convolution and is commonly used in U-Net
        architectures and diffusion models to adjust feature dimensionality without
        changing spatial resolution.

        Parameters
            input_dim (int):
                Number of input channels
            output_dim (int):
                Number of output channels
            device(optional str)
                Device on which the layer is initialized. Defaults to "cuda" if available, otherwise "cpu"

        Source:
            "Network In Network" Paper from Min Lin, Qiang Chen, Shuicheng Yan. Link: https://arxiv.org/pdf/1312.4400
    """

    def __init__(
        self, 
        input_dim: int, 
        output_dim: int, 
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ) -> None:
        super().__init__()
        self.proj = torch.nn.Conv2d(input_dim, output_dim, kernel_size=1, bias=True, device=device)
        torch.nn.init.kaiming_uniform_(self.proj.weight, a=0.0, mode="fan_in", nonlinearity="relu")
        torch.nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class ResBlock(torch.nn.Module):
    """
        Residual block used in diffusion U-Nets with optional time embedding.

        Structure (pre-activation style):
            x ----> [GN(in) -> SiLU -> Conv3x3] --(+ time_bias)--> 
                    [GN(out) -> SiLU -> Dropout -> Conv3x3(zero-init)] --> (+ skip) --> out

        - If `in_channels != out_channels`, a 1x1 projection (NiN) is applied on the skip path.
        - The time embedding (shape: [B, time_dim]) is projected to `out_channels` and added
        after the first convolution as a bias broadcast over spatial dims.

        Parameters
            in_channels (int):
                Number of input feature channels.
            out_channels (int):
                Number of output feature channels.
            kernel_size (int):
                Convolution kernel size (odd values expected).
            dropout (float):
                Dropout probability applied between the two convolutions.
            time_dim (int):
                Dimension of the input time embedding vector. If provided, a linear layer
                projects the time embedding to `out_channels` and adds it after the first conv.
            norm_groups (int):
                Number of groups for GroupNorm.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_embedding_dim: int,
        kernel_size: int = 3,
        dropout: float = 0,
        norm_groups: int = 8,
        init_conv2_zeros: bool = True,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        super().__init__()

        padding =  (kernel_size - 1) // 2
        self.time_embedding_dim = time_embedding_dim
        self.norm_groups = norm_groups if (in_channels % norm_groups  == 0 and norm_groups < in_channels and out_channels % norm_groups == 0 and norm_groups < out_channels) else 1

        self.norm_1 = torch.nn.GroupNorm(self.norm_groups, in_channels, device=device)
        self.conv_1 = torch.nn.Conv2d(in_channels=in_channels, out_channels= out_channels, kernel_size= kernel_size, padding=padding, stride=1, device=device)
        self.dense = torch.nn.Linear(time_embedding_dim, out_channels, device=device)
        self.norm_2 = torch.nn.GroupNorm(self.norm_groups, out_channels, device=device)
        self.conv_2 = torch.nn.Conv2d(in_channels=out_channels, out_channels= out_channels, kernel_size= kernel_size, padding=padding, stride=1, device=device)
        self.dropout = torch.nn.Dropout(p=dropout)
        self.activation_func = torch.nn.SiLU()

        if init_conv2_zeros:
            torch.nn.init.zeros_(self.conv_2.weight)
            torch.nn.init.zeros_(self.conv_2.bias)

        if not (in_channels == out_channels):
            self.nin = Nin(in_channels, out_channels, device=device)


    def forward(self, x: torch.Tensor, time_embedding: torch.Tensor = None):
        """
        forwards the given thensor and the given time embedding if the Res Block was __init__ with a given time embedding dim 

        Inputs
            x (torch.Tensor):
                Input tensor of shape (B, in_channels, H, W).
            time_embedding (Optional[torch.Tensor]):
                Tensor of shape (B, time_dim). Required if `time_dim` is not None.

        Returns
            torch.Tensor
                Output tensor of shape (B, out_channels, H, W).

        """        
        h = self.activation_func(self.norm_1(x))
        h = self.conv_1(h)
    
        if self.time_embedding_dim != None and time_embedding != None:
            h += self.dense(self.activation_func(time_embedding))[:, :, None,  None]    # add the time embedding
        
        h = self.activation_func(self.norm_2(h))
        h = self.dropout(h)
        h = self.conv_2(h)

        if not (x.shape[1] == h.shape[1]):
            x = self.nin(x)

        return x + h 



class DownSampleBlock(torch.nn.Module):
    
    """
        A simple downsampling block for U-Net architectures.

        Applies a 2D convolution with stride=2 to reduce the spatial resolution
        (height and width) by half, while keeping the number of channels constant.

        Parameters
        
        channels (int):
            Number of input and output channels
        kernel_size (optional int):
            Size of the convolution kernel default is 3
    """

    def __init__(
        self, 
        in_channels: int,
        out_channels: int, 
        kernel_size: Optional[int] = 3,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=2, padding=1, device=device)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the downsampling block.

        Parameters
            x (torch.Tensor):
                Input tensor of shape (B, C, H, W)

        Returns
            torch.Tensor
                Downsampled tensor of shape (B, C, H/2, W/2)
        """
        return self.conv(x)



class UpSampleBlock(torch.nn.Module):
    
    """
        A simple upsampling block for U-Net architectures

        Applies nearest-neighbor interpolation to double the spatial resolution,
        followed by a 2D convolution to refine the upsampled features

        Parameters
            channels (int):
                Number of output channels
    """

    def __init__(
        self, 
        in_channels: int,
        out_channels: int,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        super().__init__()
        self.conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, device=device)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the upsampling block.

        Parameters
            x (torch.Tensor):
                Input tensor of shape (B, C, H, W)

        Returns
            torch.Tensor
                Upsampled tensor of shape (B, C, 2H, 2W)
        """
        x = torch.nn.functional.interpolate(x, scale_factor=2, mode="nearest")
        x = self.conv(x)
        return x