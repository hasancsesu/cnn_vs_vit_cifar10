import torch
import torch.nn as nn

# 1. THE CUTTER
class PatchEmbedding(nn.Module):
    def __init__(self, img_size=64, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2)
        x = x.transpose(1, 2)
        return x

# 2. THE LABELER
class ViTPrep(nn.Module):
    def __init__(self, n_patches, embed_dim):
        super().__init__()
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + n_patches, embed_dim))

    def forward(self, x):
        batch_size = x.shape[0]
        cls_token = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x = x + self.pos_embed
        return x

# 3. THE BRAIN (Including Attention & MLP)
class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, int(embed_dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(embed_dim * mlp_ratio), embed_dim),
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.mlp(self.norm2(x))
        return x

# 4. THE FACTORY
class VisionTransformer(nn.Module):
    def __init__(self, img_size=64, n_classes=10, embed_dim=768, depth=12, num_heads=12):
        super().__init__()
        self.patch_embed = PatchEmbedding(img_size=img_size, embed_dim=embed_dim)
        self.prep = ViTPrep(self.patch_embed.n_patches, embed_dim)
        self.blocks = nn.ModuleList([TransformerBlock(embed_dim, num_heads) for _ in range(depth)])
        self.head = nn.Linear(embed_dim, n_classes)

    def forward(self, x):
        x = self.patch_embed(x)
        x = self.prep(x)
        for block in self.blocks:
            x = block(x)
        return self.head(x[:, 0])