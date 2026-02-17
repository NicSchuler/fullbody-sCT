# --------------------------------------------------------
# Swin Transformer V2 UNet Generator for Pix2Pix
# Combines Swin V2 attention with UNet encoder-decoder architecture
# For 256x256 image-to-image translation
# --------------------------------------------------------

import os
import copy
import math
import urllib.request
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
import numpy as np

# Pretrained weights URL
SWINV2_WEIGHTS_URL = "https://github.com/SwinTransformer/storage/releases/download/v2.0.0/swinv2_tiny_patch4_window8_256.pth"


# --------------------------------------------------------
# Utility functions (avoiding timm dependency)
# --------------------------------------------------------

def to_2tuple(x):
    """Convert a value to a 2-tuple."""
    if isinstance(x, (list, tuple)):
        return tuple(x)
    return (x, x)


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    """Truncated normal initialization.

    Fills the input Tensor with values drawn from a truncated normal distribution.
    """
    with torch.no_grad():
        # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
        def norm_cdf(x):
            return (1. + math.erf(x / math.sqrt(2.))) / 2.

        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)

        tensor.uniform_(2 * l - 1, 2 * u - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
        return tensor


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample."""

    def __init__(self, drop_prob=0., scale_by_keep=True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        if keep_prob > 0.0 and self.scale_by_keep:
            random_tensor.div_(keep_prob)
        return x * random_tensor


class Mlp(nn.Module):
    """MLP as used in Vision Transformer, MLP-Mixer and related networks."""
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def window_partition(x, window_size):
    """Partition into windows.

    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """Reverse window partition.

    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class WindowAttentionV2(nn.Module):
    """Window based multi-head self attention with Swin V2 improvements.

    Features:
    - Cosine similarity attention with learnable logit_scale
    - Continuous position bias via MLP (cpb_mlp)
    - Selective QKV bias (Q and V only)
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, attn_drop=0., proj_drop=0.,
                 pretrained_window_size=[0, 0]):
        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.pretrained_window_size = pretrained_window_size
        self.num_heads = num_heads

        # Learnable temperature for cosine attention
        self.logit_scale = nn.Parameter(torch.log(10 * torch.ones((num_heads, 1, 1))), requires_grad=True)

        # MLP to generate continuous relative position bias
        self.cpb_mlp = nn.Sequential(
            nn.Linear(2, 512, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(512, num_heads, bias=False)
        )

        # Get relative_coords_table with log-spaced normalization
        relative_coords_h = torch.arange(-(self.window_size[0] - 1), self.window_size[0], dtype=torch.float32)
        relative_coords_w = torch.arange(-(self.window_size[1] - 1), self.window_size[1], dtype=torch.float32)
        relative_coords_table = torch.stack(
            torch.meshgrid([relative_coords_h, relative_coords_w], indexing='ij')
        ).permute(1, 2, 0).contiguous().unsqueeze(0)  # 1, 2*Wh-1, 2*Ww-1, 2

        if pretrained_window_size[0] > 0:
            relative_coords_table[:, :, :, 0] /= (pretrained_window_size[0] - 1)
            relative_coords_table[:, :, :, 1] /= (pretrained_window_size[1] - 1)
        else:
            relative_coords_table[:, :, :, 0] /= (self.window_size[0] - 1)
            relative_coords_table[:, :, :, 1] /= (self.window_size[1] - 1)
        relative_coords_table *= 8  # normalize to -8, 8
        relative_coords_table = torch.sign(relative_coords_table) * torch.log2(
            torch.abs(relative_coords_table) + 1.0) / np.log2(8)

        self.register_buffer("relative_coords_table", relative_coords_table)

        # Get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        # QKV with selective bias (Q and V only, no K bias)
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(dim))
            self.v_bias = nn.Parameter(torch.zeros(dim))
        else:
            self.q_bias = None
            self.v_bias = None
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        qkv_bias = None
        if self.q_bias is not None:
            qkv_bias = torch.cat((self.q_bias, torch.zeros_like(self.v_bias, requires_grad=False), self.v_bias))
        qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
        qkv = qkv.reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Cosine attention
        attn = (F.normalize(q, dim=-1) @ F.normalize(k, dim=-1).transpose(-2, -1))
        logit_scale = torch.clamp(self.logit_scale, max=torch.log(torch.tensor(1. / 0.01, device=self.logit_scale.device))).exp()
        attn = attn * logit_scale

        # Continuous position bias
        relative_position_bias_table = self.cpb_mlp(self.relative_coords_table).view(-1, self.num_heads)
        relative_position_bias = relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        relative_position_bias = 16 * torch.sigmoid(relative_position_bias)
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SwinTransformerBlockV2(nn.Module):
    """Swin Transformer Block with V2 improvements (post-norm structure)."""

    def __init__(self, dim, input_resolution, num_heads, window_size=8, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, pretrained_window_size=0):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttentionV2(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop,
            pretrained_window_size=to_2tuple(pretrained_window_size))

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if self.shift_size > 0:
            H, W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1))
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            mask_windows = window_partition(img_mask, self.window_size)
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = x.view(B, H, W, C)

        # Cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # Partition windows
        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=self.attn_mask)

        # Merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)

        # Reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(B, H * W, C)

        # Post-norm structure (V2 style)
        x = shortcut + self.drop_path(self.norm1(x))

        # FFN
        x = x + self.drop_path(self.norm2(self.mlp(x)))

        return x


class PatchMergingV2(nn.Module):
    """Patch Merging Layer (Swin V2 version - reduction then norm)."""

    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(2 * dim)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x = x.view(B, H, W, C)

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

        x = self.reduction(x)
        x = self.norm(x)

        return x


class PatchExpandV2(nn.Module):
    """Patch Expanding Layer for decoder upsampling."""

    def __init__(self, input_resolution, dim, dim_scale=2, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.expand = nn.Linear(dim, 2 * dim, bias=False) if dim_scale == 2 else nn.Identity()
        self.norm = norm_layer(dim // dim_scale)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        x = self.expand(x)
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        x = x.view(B, H, W, C)
        # Equivalent to: rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=2, p2=2, c=C // 4)
        c = C // 4
        x = x.view(B, H, W, 2, 2, c)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()  # B, H, p1, W, p2, c
        x = x.view(B, H * 2, W * 2, c)
        x = x.view(B, -1, c)
        x = self.norm(x)

        return x


class FinalPatchExpand_X4(nn.Module):
    """Final 4x patch expansion for output."""

    def __init__(self, input_resolution, dim, dim_scale=4, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.dim_scale = dim_scale
        self.expand = nn.Linear(dim, 16 * dim, bias=False)
        self.output_dim = dim
        self.norm = norm_layer(self.output_dim)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        x = self.expand(x)
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        x = x.view(B, H, W, C)
        # Equivalent to: rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=dim_scale, p2=dim_scale, c=...)
        p = self.dim_scale
        c = C // (p * p)
        x = x.view(B, H, W, p, p, c)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()  # B, H, p, W, p, c
        x = x.view(B, H * p, W * p, c)
        x = x.view(B, -1, self.output_dim)
        x = self.norm(x)

        return x


class PatchEmbedV2(nn.Module):
    """Image to Patch Embedding with variable input channels."""

    def __init__(self, img_size=256, patch_size=4, in_chans=1, embed_dim=96, norm_layer=nn.LayerNorm):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer is not None else None

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)  # B Ph*Pw C
        if self.norm is not None:
            x = self.norm(x)
        return x


class BasicLayerV2(nn.Module):
    """Basic Swin V2 layer for encoder."""

    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False,
                 pretrained_window_size=0):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # Build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlockV2(
                dim=dim, input_resolution=input_resolution,
                num_heads=num_heads, window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop, attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer,
                pretrained_window_size=pretrained_window_size)
            for i in range(depth)])

        # Downsample layer
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x, use_reentrant=False)
            else:
                x = blk(x)
        if self.downsample is not None:
            x = self.downsample(x)
        return x

    def _init_respostnorm(self):
        """Initialize post-norm layers to zero for residual connection."""
        for blk in self.blocks:
            nn.init.constant_(blk.norm1.bias, 0)
            nn.init.constant_(blk.norm1.weight, 0)
            nn.init.constant_(blk.norm2.bias, 0)
            nn.init.constant_(blk.norm2.weight, 0)


class BasicLayerV2_up(nn.Module):
    """Basic Swin V2 layer for decoder with upsampling."""

    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, upsample=None, use_checkpoint=False,
                 pretrained_window_size=0):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # Build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlockV2(
                dim=dim, input_resolution=input_resolution,
                num_heads=num_heads, window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop, attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer,
                pretrained_window_size=pretrained_window_size)
            for i in range(depth)])

        # Upsample layer
        if upsample is not None:
            self.upsample = PatchExpandV2(input_resolution, dim=dim, dim_scale=2, norm_layer=norm_layer)
        else:
            self.upsample = None

    def forward(self, x):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x, use_reentrant=False)
            else:
                x = blk(x)
        if self.upsample is not None:
            x = self.upsample(x)
        return x

    def _init_respostnorm(self):
        """Initialize post-norm layers to zero for residual connection."""
        for blk in self.blocks:
            nn.init.constant_(blk.norm1.bias, 0)
            nn.init.constant_(blk.norm1.weight, 0)
            nn.init.constant_(blk.norm2.bias, 0)
            nn.init.constant_(blk.norm2.weight, 0)


class SwinV2UNet256Generator(nn.Module):
    """Swin Transformer V2 UNet Generator for 256x256 image-to-image translation.

    Combines Swin V2's attention mechanism with UNet's encoder-decoder architecture.

    Args:
        input_nc (int): Number of input channels (e.g., 1 for grayscale CT)
        output_nc (int): Number of output channels
        embed_dim (int): Patch embedding dimension. Default: 96
        depths (tuple[int]): Depth of each encoder stage. Default: [2, 2, 6, 2]
        depths_decoder (tuple[int]): Depth of each decoder stage. Default: [2, 2, 2, 2]
        num_heads (tuple[int]): Number of attention heads. Default: [3, 6, 12, 24]
        window_size (int): Window size. Default: 8
        mlp_ratio (float): MLP ratio. Default: 4.0
        qkv_bias (bool): QKV bias. Default: True
        drop_rate (float): Dropout rate. Default: 0.0
        attn_drop_rate (float): Attention dropout rate. Default: 0.0
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        use_checkpoint (bool): Use gradient checkpointing. Default: False
        pretrained_window_sizes (tuple[int]): Pretrained window sizes. Default: [0,0,0,0]
        use_dropout (bool): Use dropout (pix2pix compatibility). Default: False
        pretrained_path (str): Path to pretrained checkpoint. Required.
    """

    def __init__(
        self,
        input_nc=1,
        output_nc=1,
        embed_dim=96,
        depths=[2, 2, 6, 2],
        depths_decoder=[2, 2, 2, 2],
        num_heads=[3, 6, 12, 24],
        window_size=8,
        mlp_ratio=4.,
        qkv_bias=True,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.1,
        norm_layer=nn.LayerNorm,
        patch_norm=True,
        use_checkpoint=False,
        pretrained_window_sizes=[0, 0, 0, 0],
        use_dropout=False,
        pretrained_path="checkpoints/swinv2_tiny_patch4_window8_256.pth",
        freeze_encoder_except_first=False,
        **kwargs
    ):
        super().__init__()

        self.input_nc = input_nc
        self.output_nc = output_nc
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.mlp_ratio = mlp_ratio

        # Patch embedding
        self.patch_embed = PatchEmbedV2(
            img_size=256, patch_size=4, in_chans=input_nc,
            embed_dim=embed_dim, norm_layer=norm_layer if patch_norm else None
        )
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        self.pos_drop = nn.Dropout(p=drop_rate)

        # Stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        # Build encoder
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayerV2(
                dim=int(embed_dim * 2 ** i_layer),
                input_resolution=(patches_resolution[0] // (2 ** i_layer),
                                  patches_resolution[1] // (2 ** i_layer)),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer,
                downsample=PatchMergingV2 if (i_layer < self.num_layers - 1) else None,
                use_checkpoint=use_checkpoint,
                pretrained_window_size=pretrained_window_sizes[i_layer]
            )
            self.layers.append(layer)

        # Build decoder
        self.layers_up = nn.ModuleList()
        self.concat_back_dim = nn.ModuleList()
        for i_layer in range(self.num_layers):
            concat_linear = nn.Linear(
                2 * int(embed_dim * 2 ** (self.num_layers - 1 - i_layer)),
                int(embed_dim * 2 ** (self.num_layers - 1 - i_layer))
            ) if i_layer > 0 else nn.Identity()

            if i_layer == 0:
                # First decoder layer: just upsample
                layer_up = PatchExpandV2(
                    input_resolution=(
                        patches_resolution[0] // (2 ** (self.num_layers - 1)),
                        patches_resolution[1] // (2 ** (self.num_layers - 1))
                    ),
                    dim=int(embed_dim * 2 ** (self.num_layers - 1)),
                    dim_scale=2,
                    norm_layer=norm_layer
                )
            else:
                layer_up = BasicLayerV2_up(
                    dim=int(embed_dim * 2 ** (self.num_layers - 1 - i_layer)),
                    input_resolution=(
                        patches_resolution[0] // (2 ** (self.num_layers - 1 - i_layer)),
                        patches_resolution[1] // (2 ** (self.num_layers - 1 - i_layer))
                    ),
                    depth=depths_decoder[self.num_layers - 1 - i_layer],
                    num_heads=num_heads[self.num_layers - 1 - i_layer],
                    window_size=window_size,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[sum(depths[:(self.num_layers - 1 - i_layer)]):
                                  sum(depths[:(self.num_layers - 1 - i_layer) + 1])],
                    norm_layer=norm_layer,
                    upsample=PatchExpandV2 if (i_layer < self.num_layers - 1) else None,
                    use_checkpoint=use_checkpoint,
                    pretrained_window_size=pretrained_window_sizes[self.num_layers - 1 - i_layer]
                )

            self.layers_up.append(layer_up)
            self.concat_back_dim.append(concat_linear)

        # Normalization layers
        self.norm = norm_layer(self.num_features)
        self.norm_up = norm_layer(embed_dim)

        # Final upsampling (4x to get back to original resolution)
        self.up = FinalPatchExpand_X4(
            input_resolution=(patches_resolution[0], patches_resolution[1]),
            dim_scale=4,
            dim=embed_dim
        )

        # Output head with Tanh for image-to-image translation
        self.output = nn.Sequential(
            nn.Conv2d(embed_dim, output_nc, kernel_size=1, bias=False),
            nn.Tanh()
        )

        # Initialize weights
        self.apply(self._init_weights)
        for bly in self.layers:
            bly._init_respostnorm()

        # Load pretrained weights (mandatory)
        self._load_pretrained(pretrained_path)

        # Freeze encoder layers (keep patch_embed trainable)
        if freeze_encoder_except_first:
            for i in range(0, self.num_layers):
                for param in self.layers[i].parameters():
                    param.requires_grad = False
            print(f"Froze encoder layers 0-{self.num_layers - 1} "
                  f"(patch_embed remain trainable)")

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def _load_pretrained(self, pretrained_path):
        """Load pretrained SwinV2 encoder weights and mirror to decoder."""
        # Resolve path relative to this file's directory
        if not os.path.isabs(pretrained_path):
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            pretrained_path = os.path.join(base_dir, pretrained_path)

        if not os.path.exists(pretrained_path):
            # Auto-download pretrained weights
            print(f"Pretrained weights not found at {pretrained_path}")
            print(f"Downloading from {SWINV2_WEIGHTS_URL}...")
            os.makedirs(os.path.dirname(pretrained_path), exist_ok=True)
            try:
                urllib.request.urlretrieve(SWINV2_WEIGHTS_URL, pretrained_path)
                print(f"Successfully downloaded pretrained weights to {pretrained_path}")
            except Exception as e:
                raise RuntimeError(
                    f"Failed to download pretrained weights: {e}\n"
                    f"Please manually download from {SWINV2_WEIGHTS_URL} "
                    f"and save to {pretrained_path}"
                )

        print(f"Loading pretrained weights from: {pretrained_path}")
        device = next(self.parameters()).device if len(list(self.parameters())) > 0 else 'cpu'
        pretrained_dict = torch.load(pretrained_path, map_location=device)

        # Handle nested 'model' key
        if 'model' in pretrained_dict:
            pretrained_dict = pretrained_dict['model']

        model_dict = self.state_dict()

        # Build full dict with encoder weights mirrored to decoder
        full_dict = copy.deepcopy(pretrained_dict)
        for k, v in pretrained_dict.items():
            if "layers." in k:
                # Mirror encoder layers to decoder: layers.0 -> layers_up.3, etc.
                layer_num = int(k.split('.')[1])
                decoder_layer_num = 3 - layer_num
                if decoder_layer_num > 0:  # Skip bottleneck
                    current_k = "layers_up." + str(decoder_layer_num) + k[8:]
                    full_dict.update({current_k: v})

        # Filter weights that match our model
        filtered_dict = {}
        skipped_keys = []
        for k, v in full_dict.items():
            # Skip classification head
            if 'head.' in k or 'avgpool' in k:
                continue

            # Skip patch_embed.proj if input channels don't match
            if 'patch_embed.proj' in k and self.input_nc != 3:
                skipped_keys.append(f"{k} (input_nc mismatch)")
                continue

            # Only load if key exists and shape matches
            if k in model_dict:
                if model_dict[k].shape == v.shape:
                    filtered_dict[k] = v
                else:
                    skipped_keys.append(f"{k} (shape mismatch: {v.shape} vs {model_dict[k].shape})")

        # Load filtered weights
        msg = self.load_state_dict(filtered_dict, strict=False)
        print(f"Loaded {len(filtered_dict)} pretrained parameters")
        if skipped_keys:
            print(f"Skipped {len(skipped_keys)} parameters due to mismatches")
        if msg.missing_keys:
            # Filter out expected missing keys (decoder-specific, output head)
            unexpected_missing = [k for k in msg.missing_keys
                                  if not any(x in k for x in ['output', 'up.', 'norm_up', 'concat_back_dim'])]
            if unexpected_missing:
                print(f"Note: {len(unexpected_missing)} encoder keys not found in checkpoint")

    def forward_encoder(self, x):
        """Encoder forward pass with skip connections."""
        x = self.patch_embed(x)
        x = self.pos_drop(x)

        x_downsample = []
        for layer in self.layers:
            x_downsample.append(x)
            x = layer(x)

        x = self.norm(x)
        return x, x_downsample

    def forward_decoder(self, x, x_downsample):
        """Decoder forward pass with skip connections."""
        for inx, layer_up in enumerate(self.layers_up):
            if inx == 0:
                x = layer_up(x)
            else:
                x = torch.cat([x, x_downsample[self.num_layers - 1 - inx]], dim=-1)
                x = self.concat_back_dim[inx](x)
                x = layer_up(x)

        x = self.norm_up(x)
        return x

    def forward_output(self, x):
        """Final output projection."""
        H, W = self.patches_resolution
        B, L, C = x.shape
        assert L == H * W

        x = self.up(x)
        x = x.view(B, 4 * H, 4 * W, -1)
        x = x.permute(0, 3, 1, 2)  # B, C, H, W
        x = self.output(x)
        return x

    def forward(self, x):
        """Full forward pass."""
        x, x_downsample = self.forward_encoder(x)
        x = self.forward_decoder(x, x_downsample)
        x = self.forward_output(x)
        return x

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {"cpb_mlp", "logit_scale", 'relative_position_bias_table'}
