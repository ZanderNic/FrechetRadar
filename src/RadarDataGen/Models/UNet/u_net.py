# std lib imports
from typing import List, Optional

# 3 party import
import torch

# projekt imports
from RadarDataGen.Models.UNet.blocks import ResBlock, DownSampleBlock, UpSampleBlock, SelfAttentionBlock


class U_Net(torch.nn.Module):
    def __init__(
        self, 
        input_chanels: int,
        time_embedding_dim: int, 
        channels_per_level: List[int] = [64, 128, 256, 512],
        attention_levels: List[bool] = [0, 0, 1, 1],
        resnet_blocks_per_depth: int = 2,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        
        """
            U-Net backbone for diffusion models with optional self-attention at selected scales.

            The architecture follows an encoder-bottleneck-decoder pattern:
            - Encoder (down path): per level: N Res blocks + optional self-attention + downsample
            - Middle (bottleneck): res block + self-attention + Res block
            - Decoder (up path): per level: concatenate with skip from encoder + N Res blocks + optional self-attention + upsample
            
            Self-attention can be enabled per level via `attention_levels`. Attention at the middle level
            is always applied.

            Parameters
                input_chanels (int):
                    Number of input channels 
                time_embedding_dim (int):
                    Dimension of the global time embedding vector provided to each ResBlock
                channels_per_level (List[int]):
                    Output channels for each encoder level (length defines the number of levels)
                attention_levels (List[bool]):
                    A boolean mask of the same length as channels_per_level indicating whether to apply
                    self-attention at the corresponding level (on both encoder and decoder sides).
                resnet_blocks_per_depth (int):
                    Number of ResBlocks at each level for both encoder and decoder.
        
            Source:
                The implementation is not 
                Paper U-Net: Convolutional Networks for Biomedical Image Segmentation https://arxiv.org/pdf/1505.04597
        """

        super().__init__()
        num_levels = len(channels_per_level)
        
        ###  In Conv  ################################   
        self.in_conv = torch.nn.Conv2d(input_chanels, channels_per_level[0], kernel_size=3, padding=1)

        ###  Encoder  ################################
        self.down = torch.nn.ModuleList()

        for level in range(num_levels - 1):   # build up self.down 
            in_ch = channels_per_level[level]
            out_ch = channels_per_level[level + 1]

            # resnet_blocks_per_depth * ResBlocks at this level
            res_blocks = torch.nn.ModuleList([
                ResBlock(in_channels=(in_ch),
                        out_channels=in_ch,
                        time_embedding_dim=time_embedding_dim,
                        device=device)
                for i in range(resnet_blocks_per_depth)
            ])
            attn = SelfAttentionBlock(in_ch, device=device) if attention_levels[level] else torch.nn.Identity()     # Optional attention at this level (encoder side)
            down = DownSampleBlock(in_channels=in_ch, out_channels=out_ch, device=device)                           # Downsample except at the bottom-most encoder level
            self.down.append(torch.nn.ModuleDict(dict(res_blocks=res_blocks, attn=attn, down=down)))

        ###  Middel  ################################
        self.middel = torch.nn.ModuleList([
            ResBlock(in_channels=channels_per_level[-1], out_channels=channels_per_level[-1], time_embedding_dim=time_embedding_dim, device=device),
            SelfAttentionBlock(in_channels= channels_per_level[-1], device=device),
            ResBlock(in_channels=channels_per_level[-1], out_channels=channels_per_level[-1], time_embedding_dim=time_embedding_dim, device=device)
        ])

        ###  Decoder  ################################
        self.up = torch.nn.ModuleList()
       
        for level in range(num_levels - 1, 0, -1):
            in_ch = channels_per_level[level]
            out_ch = channels_per_level[level - 1]
            
            up = UpSampleBlock(in_channels=in_ch, out_channels=out_ch)

            res_blocks = torch.nn.ModuleList([])
            res_blocks.append(ResBlock(2 * out_ch, out_ch, time_embedding_dim=time_embedding_dim, device=device))     # because we have x + skip as input
            for _ in range(1, resnet_blocks_per_depth):        
                res_blocks.append(ResBlock(out_ch, out_ch, time_embedding_dim=time_embedding_dim, device=device))     # because we have x 

            attn = SelfAttentionBlock(out_ch)  if attention_levels[level] else torch.nn.Identity()
            self.up.append(torch.nn.ModuleDict(dict(res_blocks=res_blocks, attn=attn, up=up)))

        ###  Out Conv  ############################### 
        norm_groups_out = 32 if channels_per_level[0] % 32 == 0 and channels_per_level[0] > 32 else 1
        self.out_conv = torch.nn.Sequential(
            torch.nn.GroupNorm(norm_groups_out, channels_per_level[0]),
            torch.nn.SiLU(),
            torch.nn.Conv2d(channels_per_level[0], input_chanels, kernel_size=3, padding=1)
        )

        
    def forward(
        self, 
        x: torch.Tensor,
        t_emb: torch.Tensor
    ) -> torch.Tensor:
        """
            forward pass of the U-Net

            Parameters
                x (torch.Tensor):
                    Input tensor of shape (B, C_in, H, W)
                t_emb (torch.Tensor):
                    Global time embedding of shape (B, time_embedding_dim)

            Returns
                torch.Tensor
                    Output tensor of shape (B, C_in, H, W)
        """

        ###  In Conv  ################################   
        h = self.in_conv(x)
        
        ###  Encoder  ################################
        pre_down_skips = [] # we save the features for the skip conection in the decoder path 
        for stage in self.down:
            for res_block in stage["res_blocks"]:
                h = res_block(h, t_emb)

            h = stage["attn"](h)
            pre_down_skips.append(h)
            h = stage["down"](h)
        
        ###  Middel  ################################
        h = self.middel[0](h, t_emb)
        h = self.middel[1](h)
        h = self.middel[2](h, t_emb)

        ###  Decoder  ################################
        for level, stage in enumerate(self.up):
            skip = pre_down_skips[-(level + 1)]
            h = stage["up"](h)
            h = torch.cat([h, skip], dim=1)

            for blk in stage["res_blocks"]:
                h = blk(h, t_emb)
            h = stage["attn"](h)

        ###  Out Conv  ############################### 
        return self.out_conv(h) 
   
    
    @torch.no_grad()
    def predict(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        """
            Inference helper that calls `forward` under `no_grad()`.

            Parameters
                x (torch.Tensor):
                    Input tensor of shape (B, C_in, H, W)
                t_emb (torch.Tensor):
                    Global time embedding of shape (B, time_embedding_dim)

            Returns
                torch.Tensor
                    Output tensor of shape (B, C_in, H, W)
        """
        self.eval()
        return self.forward(x, t_emb)

    
    def save(self, path: str) -> None:
        """
            Save the model weights to a file.

            Parameters
                path (str):
                    File path for saving (e.g., 'unet.pt').
        """
        torch.save(self.state_dict(), path)


    def load(self, path: str, map_location: Optional[str] = None) -> None:
        """
            Load the model weights from a file.

            Parameters
                path (str):
                    File path to load from.
                map_location (Optional[str]):
                    device mapping
        """
        state = torch.load(path, map_location=map_location or "cpu")
        self.load_state_dict(state)