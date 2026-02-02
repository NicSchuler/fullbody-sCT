"""
Swin Transformer networks for MR-to-CT synthesis.

Architecture:
- SwinEncoder: Pretrained SwinV2-T adapted for grayscale input
- PatchExpanding: Upsampling layer for decoder
- SwinGenerator: Complete encoder-decoder architecture
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import timm
except ImportError:
    raise ImportError("Please install timm: pip install timm>=0.9.0")


class SwinEncoder(nn.Module):
    """SwinV2-Tiny encoder adapted for 1-channel grayscale medical images.

    Uses pretrained ImageNet weights with averaged RGB patch embedding.

    Output feature maps:
        - Stage 0: [B, 96, 64, 64]
        - Stage 1: [B, 192, 32, 32]
        - Stage 2: [B, 384, 16, 16]
        - Stage 3: [B, 768, 8, 8] (bottleneck)
    """

    def __init__(self, pretrained=True):
        super().__init__()

        # Load pretrained SwinV2-T with features_only mode
        self.backbone = timm.create_model(
            'swinv2_tiny_window8_256',
            pretrained=pretrained,
            features_only=True,
            out_indices=(0, 1, 2, 3)
        )

        # Adapt patch embedding for grayscale (1 channel instead of 3)
        # Original: Conv2d(3, 96, kernel_size=4, stride=4)
        original_patch_embed = self.backbone.patch_embed.proj
        new_patch_embed = nn.Conv2d(
            in_channels=1,
            out_channels=96,
            kernel_size=original_patch_embed.kernel_size,
            stride=original_patch_embed.stride,
            padding=original_patch_embed.padding
        )

        # Initialize by averaging RGB weights
        if pretrained:
            with torch.no_grad():
                new_patch_embed.weight.copy_(
                    original_patch_embed.weight.mean(dim=1, keepdim=True)
                )
                new_patch_embed.bias.copy_(original_patch_embed.bias)

        self.backbone.patch_embed.proj = new_patch_embed

        # Feature channels at each stage
        self.feature_channels = [96, 192, 384, 768]

    def forward(self, x):
        """Forward pass.

        Args:
            x: Input tensor [B, 1, 256, 256]

        Returns:
            List of feature maps at each stage
        """
        features = self.backbone(x)
        return features


class PatchExpanding(nn.Module):
    """Patch Expanding layer for upsampling in decoder.

    Doubles spatial resolution, optionally changes channel dimension.
    Based on Swin-UNet paper.
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Linear expansion for pixel shuffle
        self.expand = nn.Linear(in_channels, out_channels * 4, bias=False)
        self.norm = nn.LayerNorm(out_channels)

    def forward(self, x):
        """Upsample by 2x.

        Args:
            x: [B, C, H, W]

        Returns:
            [B, out_channels, H*2, W*2]
        """
        B, C, H, W = x.shape

        # Convert to channel-last for linear layer
        x = x.permute(0, 2, 3, 1)  # [B, H, W, C]

        # Expand channels
        x = self.expand(x)  # [B, H, W, C*4]

        # Reshape for pixel shuffle: [B, H, W, 2, 2, C'] -> [B, H*2, W*2, C']
        x = x.view(B, H, W, 2, 2, self.out_channels)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        x = x.view(B, H * 2, W * 2, self.out_channels)

        # Normalize
        x = self.norm(x)

        # Back to channel-first
        x = x.permute(0, 3, 1, 2)  # [B, C', H*2, W*2]

        return x


class DecoderBlock(nn.Module):
    """Decoder block with skip connection.

    Upsamples input, concatenates with skip features, and refines.
    """

    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()

        # Upsample
        self.upsample = PatchExpanding(in_channels, in_channels // 2)

        # Fusion after skip connection
        fusion_in = in_channels // 2 + skip_channels
        self.fusion = nn.Sequential(
            nn.Conv2d(fusion_in, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU()
        )

    def forward(self, x, skip):
        """Forward pass.

        Args:
            x: Decoder features [B, C, H, W]
            skip: Encoder skip features [B, C_skip, H*2, W*2]

        Returns:
            Upsampled and refined features
        """
        x = self.upsample(x)
        x = torch.cat([x, skip], dim=1)
        x = self.fusion(x)
        return x


class FinalUpsample(nn.Module):
    """Final upsampling from 64x64 to 256x256."""

    def __init__(self, in_channels, out_channels):
        super().__init__()

        # 64 -> 128
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(in_channels // 2),
            nn.GELU()
        )
        # 128 -> 256
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(in_channels // 2, in_channels // 4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(in_channels // 4),
            nn.GELU()
        )
        # Final projection
        self.proj = nn.Conv2d(in_channels // 4, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.up1(x)
        x = self.up2(x)
        x = self.proj(x)
        return x


class SwinGenerator(nn.Module):
    """Complete Swin Transformer U-Net for MR-to-CT synthesis.

    Architecture:
        Encoder (SwinV2-T pretrained):
            Input:  [B, 1, 256, 256]
            Stage0: [B, 96, 64, 64]
            Stage1: [B, 192, 32, 32]
            Stage2: [B, 384, 16, 16]
            Stage3: [B, 768, 8, 8] (bottleneck)

        Decoder (Symmetric with skip connections):
            Dec3:   [B, 768, 8, 8] + skip2 -> [B, 384, 16, 16]
            Dec2:   [B, 384, 16, 16] + skip1 -> [B, 192, 32, 32]
            Dec1:   [B, 192, 32, 32] + skip0 -> [B, 96, 64, 64]
            Final:  [B, 96, 64, 64] -> [B, 1, 256, 256]
    """

    def __init__(self, input_nc=1, output_nc=1, pretrained=True):
        super().__init__()

        self.input_nc = input_nc
        self.output_nc = output_nc

        # Encoder
        self.encoder = SwinEncoder(pretrained=pretrained)

        # Decoder blocks
        # Dec3: 768 -> 384 (skip from stage2: 384)
        self.decoder3 = DecoderBlock(
            in_channels=768,
            skip_channels=384,
            out_channels=384
        )
        # Dec2: 384 -> 192 (skip from stage1: 192)
        self.decoder2 = DecoderBlock(
            in_channels=384,
            skip_channels=192,
            out_channels=192
        )
        # Dec1: 192 -> 96 (skip from stage0: 96)
        self.decoder1 = DecoderBlock(
            in_channels=192,
            skip_channels=96,
            out_channels=96
        )

        # Final upsampling: 64x64 -> 256x256
        self.final = FinalUpsample(
            in_channels=96,
            out_channels=output_nc
        )

        # Output activation
        self.tanh = nn.Tanh()

    def forward(self, x):
        """Forward pass.

        Args:
            x: Input MR image [B, 1, 256, 256] in range [-1, 1]

        Returns:
            Synthesized CT image [B, 1, 256, 256] in range [-1, 1]
        """
        # Encoder - outputs are in channel-last format [B, H, W, C]
        enc_features = self.encoder(x)
        skip0, skip1, skip2, bottleneck = enc_features

        # Convert to channel-first format [B, C, H, W]
        skip0 = skip0.permute(0, 3, 1, 2).contiguous()  # [B, 96, 64, 64]
        skip1 = skip1.permute(0, 3, 1, 2).contiguous()  # [B, 192, 32, 32]
        skip2 = skip2.permute(0, 3, 1, 2).contiguous()  # [B, 384, 16, 16]
        bottleneck = bottleneck.permute(0, 3, 1, 2).contiguous()  # [B, 768, 8, 8]

        # Decoder with skip connections
        x = self.decoder3(bottleneck, skip2)  # [B, 384, 16, 16]
        x = self.decoder2(x, skip1)           # [B, 192, 32, 32]
        x = self.decoder1(x, skip0)           # [B, 96, 64, 64]

        # Final upsampling
        x = self.final(x)  # [B, 1, 256, 256]

        # Output activation
        out = self.tanh(x)

        return out


def count_parameters(model):
    """Count trainable parameters in model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    # Test the architecture
    print("Testing SwinGenerator...")

    model = SwinGenerator(input_nc=1, output_nc=1, pretrained=True)
    print(f"Total parameters: {count_parameters(model):,}")

    # Test forward pass
    x = torch.randn(2, 1, 256, 256)
    with torch.no_grad():
        y = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
    print(f"Output range: [{y.min():.3f}, {y.max():.3f}]")
