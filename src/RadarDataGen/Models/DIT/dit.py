# std-lib import 
from typing import Dict, Any

# 3 Party imports
import numpy as np
import torch
import torch.nn as nn
from timm.models.vision_transformer import Mlp, Attention, PatchEmbed


# https://github.com/CompVis/tread/blob/master/dit.py#L84 import 
from RadarDataGen.Models.DIT.routing_model import Router



"""
    All components of this module (DiT architecture, Transformer blocks,
    positional embeddings, and modulation functions) are adapted from the
    official TReAD/DiT implementation released by CompVis:

        https://github.com/CompVis/tread

    Original authors and copyright holders: CompVis
    The implementation here follows the same structural logic with adjustments
    for integration into this project 

    Paper bibtex:
        @InProceedings{krause2025tread,
            author={Krause, Felix and Phan, Timy and Gui, Ming and Baumann, Stefan Andreas and Hu, Vincent Tao and Ommer, Bj\"orn},
            title={TREAD: Token Routing for Efficient Architecture-agnostic Diffusion Training},
            booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
            month={October},
            year={2025},
            pages={15703-15713}
        }
"""


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

class LabelEmbedder(nn.Module):
    def __init__(self, num_classes, hidden_size):
        super().__init__()
        use_cfg_embedding = True
        self.embedding_table = nn.Embedding(
            num_classes + use_cfg_embedding, hidden_size
        )
        self.num_classes = num_classes
        self._init()
        
    def token_drop(self, labels, cond_drop_prob, force_drop_ids=None):
        if force_drop_ids is None:
            drop_ids = (
                torch.rand(labels.shape[0], device=labels.device) < cond_drop_prob
            )
        else:
            drop_ids = force_drop_ids == 1
        labels = torch.where(drop_ids, self.num_classes, labels)
        return labels

    def forward(self, labels, class_drop_prob=0.1, force_drop_ids=None):
        if labels.dim() == 2 and labels.size(1) == self.num_classes:
            labels = labels.argmax(dim=1)
        elif labels.dim() != 1:
            raise ValueError(f"Expected labels to be of shape (batch_size,) or (batch_size, {self.num_classes}), but got {labels.shape}")
        assert labels.max() <= 999
        use_dropout = class_drop_prob > 0
        if use_dropout or (force_drop_ids is not None):
            labels = self.token_drop(labels, class_drop_prob, force_drop_ids)
        embeddings = self.embedding_table(labels)
        return embeddings
    
    def _init(self):
        nn.init.normal_(self.embedding_table.weight, std=0.02)
        
class TransformerBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, cond_mode=None, **block_kwargs):
        super().__init__()
        self.cond_mode = cond_mode
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)

        if cond_mode == "adaln":
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(hidden_size, 6 * hidden_size, bias=True)
            )
            self._init_conditional()
        else:
            self._init_standard()
            
    #@torch.compile
    def forward(self, x, c=None):
        if self.cond_mode == "adaln" and c is not None:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
            x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
            x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        else:
            x = x + self.attn(self.norm1(x))
            x = x + self.mlp(self.norm2(x))
        return x

    def _init_standard(self):
        pass

    def _init_conditional(self):
        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)

class FinalLayer(nn.Module):
    def __init__(self, hidden_size, patch_size, out_channels, cond_mode=None):
        super().__init__()
        self.cond_mode = cond_mode
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)

        if cond_mode == "adaln":
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(hidden_size, 2 * hidden_size, bias=True)
            )
            self._init_conditional()
        else:
            self._init_standard()

    #@torch.compile
    def forward(self, x, c=None):
        if self.cond_mode == "adaln" and c is not None:
            shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
            x = modulate(self.norm_final(x), shift, scale)
        else:
            x = self.norm_final(x)
        x = self.linear(x)
        return x

    def _init_standard(self):
        nn.init.constant_(self.linear.weight, 0)
        nn.init.constant_(self.linear.bias, 0)

    def _init_conditional(self):
        nn.init.constant_(self.linear.weight, 0)
        nn.init.constant_(self.linear.bias, 0)
        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)


class DiT(nn.Module):
    def __init__(
        self,
        input_size=32,
        patch_size=2,
        in_channels=4,
        hidden_size=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        num_classes=1000,
        learn_sigma=False,
        cond_mode="adaln",
        enable_routing=False,
        routes=None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        super().__init__()
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        self.input_size = input_size
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.num_classes = num_classes
        self.hidden_size = hidden_size
        self.enable_routing = enable_routing
        self.routes = routes if routes is not None else []
        
        self.x_embedder = PatchEmbed(image_size = input_size, patch_size= patch_size, in_chans=in_channels, embed_dim=hidden_size)

        num_patches = self.x_embedder.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size), requires_grad=False)

        self.blocks = nn.ModuleList([
            TransformerBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio, cond_mode=cond_mode)
            for _ in range(depth)
        ])

        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels, cond_mode=cond_mode)
        if enable_routing:
            self.router = Router()
        self.mask_token = None
        self.initialize_weights()

        self.to(device)

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.x_embedder.num_patches ** 0.5))
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        # nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        # nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

    def unpatchify(self, x):
        c = self.out_channels
        p = self.x_embedder.patch_size[0]
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
        return imgs


    def forward(self, x: torch.Tensor, t: torch.Tensor, **kwargs):
        force_routing = kwargs.get('force_routing', False)
        overwrite_selection_ratio = kwargs.get('overwrite_selection_ratio', None)

        x = self.x_embedder(x) + self.pos_embed
        c = t

        use_routing = (self.training and self.enable_routing and self.routes) or force_routing
        route_count = 0 if use_routing else None
        fp32_next = False

        for idx, block in enumerate(self.blocks):
            if use_routing and idx == self.routes[route_count]['start_layer_idx']:
                x_D_last = x.clone()
                ids_keep = self.router.get_mask(x, selection_rate=self.routes[route_count]['selection_ratio'] if overwrite_selection_ratio is None else overwrite_selection_ratio)
                x = self.router.start_route(x, ids_keep)

            if fp32_next:
                with torch.amp.autocast(device_type="cuda", enabled=False):
                    x = block(x, c)
                fp32_next = False
            else:
                x = block(x, c)

            if use_routing and idx == self.routes[route_count]['end_layer_idx']:
                x = self.router.end_route(x, ids_keep, original_x=x_D_last)
                fp32_next = True
                if route_count < len(self.routes) - 1:
                    route_count += 1
                    
        x = self.final_layer(x, c)
        return self.unpatchify(x)


def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb