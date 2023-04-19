import torch
import torch.nn as nn
from collections import OrderedDict

def drop_path(x, drop_prob=0., training=False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0], ) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    output = x.div(keep_prob) * random_tensor
    return output

class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)
    
class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, embed_dim=768):
        super().__init__()
        img_size = (img_size, img_size)
        patch_size = (patch_size, patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1]) # 14 * 14
        self.num_patches = self.grid_size[0] * self.grid_size[1] # 196

        self.proj = nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.Identity()

    def forward(self, x):
        # [b, c, h, w]->[b, hw, c]
        x = self.proj(x).flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x
    
class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop_ratio=0., proj_drop_ratio=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop_ratio)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop_ratio)

    def forward(self, x):
        #[b, 196+1, 768]
        B, N, C = x.shape
        # [b, n, 3, num_heads, c//num_heads]->[3, b, num_heads, n, c//num_heads]
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1) # 对每一行进行softmax
        attn = self.attn_drop(attn)

        # [b, num_heads, n, c//num_heads]->[b, n, c]
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class MLP(nn.Module):
    def __init__(self, input, output, drop=0.):
        super().__init__()
        self.fc1 = nn.Linear(input, 4 * input)
        self.gelu = nn.GELU()
        self.fc2 = nn.Linear(4 * input, output)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.gelu(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
    
class Encoder_Block(nn.Module):
    def __init__(self, dim, num_heads, qkv_bias=False, qk_scale=None, drop_ratio=0., attn_drop_ratio=0., drop_path_ratio=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = Attention(dim, num_heads, qkv_bias, qk_scale, attn_drop_ratio, drop_ratio)
        self.drop_path = DropPath(drop_path_ratio) if drop_path_ratio > 0 else nn.Identity()
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = MLP(dim, dim, drop_ratio)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class ViT(nn.Module):
    def __init__(self, img_size=224, patch_size=16, num_classes=1000, embed_dim=768, depth=12, num_heads=12, qkv_bias=True, qk_scale=None,
                 representation_size=None, drop_ratio=0., attn_drop_ratio=0., drop_path_ratio=0.):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim
        self.num_tokens = 1
        self.patch_embed = PatchEmbed(img_size, patch_size, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))
        self.pos_drop = nn.Dropout(drop_ratio)

        dpr = [x.item() for x in torch.linspace(0, drop_path_ratio, depth)]
        self.blocks = nn.Sequential(*[Encoder_Block(dim=embed_dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                                                    drop_ratio=drop_ratio, attn_drop_ratio=attn_drop_ratio, drop_path_ratio=dpr[i])
                                                    for i in range(depth)])
        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)
        if representation_size:
            self.has_logits = True
            self.num_features = representation_size
            self.pre_logits = nn.Sequential(OrderedDict([("fc", nn.Linear(embed_dim, representation_size)),
                                                         ("act", nn.Tanh())]))
        else:
            self.has_logits = False
            self.pre_logits = nn.Identity()

        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
        self.head_dist = None

    def forward(self, x):
        x = self.patch_embed(x)
        # [1, 1, 768]->[b, 1, 768]
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x = self.pos_drop(x + self.pos_embed)
        x = self.blocks(x)
        x = self.norm(x)
        x = self.pre_logits(x[:, 0]) # 提取cls_token
        x = self.head(x)
        return x

def vit_base_patch16_224(num_classes=1000):
    """
    ViT-Base model (ViT-B/16)
    链接: https://pan.baidu.com/s/1zqb08naP0RPqqfSXfkB2EA  密码: eu9f
    """
    model = ViT(img_size=224, patch_size=16, embed_dim=768, depth=12, num_heads=12, representation_size=None, num_classes=num_classes)
    return model


def vit_base_patch16_224_in21k(num_classes=21843, has_logits=True):
    """
    ViT-Base model (ViT-B/16)
    https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_patch16_224_in21k-e5005f0a.pth
    """
    model = ViT(img_size=224, patch_size=16, embed_dim=768, depth=12, num_heads=12, representation_size=768 if has_logits else None, num_classes=num_classes)
    return model


def vit_base_patch32_224(num_classes=1000):
    """
    ViT-Base model (ViT-B/32)
    链接: https://pan.baidu.com/s/1hCv0U8pQomwAtHBYc4hmZg  密码: s5hl
    """
    model = ViT(img_size=224, patch_size=32, embed_dim=768, depth=12, num_heads=12, representation_size=None, num_classes=num_classes)
    return model


def vit_base_patch32_224_in21k(num_classes=21843, has_logits=True):
    """
    ViT-Base model (ViT-B/32)
    https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_patch32_224_in21k-8db57226.pth
    """
    model = ViT(img_size=224, patch_size=32, embed_dim=768, depth=12, num_heads=12, representation_size=768 if has_logits else None, num_classes=num_classes)
    return model


def vit_large_patch16_224(num_classes=1000):
    """
    ViT-Large model (ViT-L/16)
    链接: https://pan.baidu.com/s/1cxBgZJJ6qUWPSBNcE4TdRQ  密码: qqt8
    """
    model = ViT(img_size=224, patch_size=16, embed_dim=1024, depth=24, num_heads=16, representation_size=None, num_classes=num_classes)
    return model


def vit_large_patch16_224_in21k(num_classes=21843, has_logits=True):
    """
    ViT-Large model (ViT-L/16)
    https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_large_patch16_224_in21k-606da67d.pth
    """
    model = ViT(img_size=224, patch_size=16, embed_dim=1024, depth=24, num_heads=16, representation_size=1024 if has_logits else None, num_classes=num_classes)
    return model


def vit_large_patch32_224_in21k(num_classes=21843, has_logits=True):
    """
    ViT-Large model (ViT-L/32)
    https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_large_patch32_224_in21k-9046d2e7.pth
    """
    model = ViT(img_size=224, patch_size=32, embed_dim=1024, depth=24, num_heads=16, representation_size=1024 if has_logits else None, num_classes=num_classes)
    return model


def vit_huge_patch14_224_in21k(num_classes=21843, has_logits=True):
    """
    ViT-Huge model (ViT-H/14)
    """
    model = ViT(img_size=224, patch_size=14, embed_dim=1280, depth=32, num_heads=16, representation_size=1280 if has_logits else None, num_classes=num_classes)
    return model