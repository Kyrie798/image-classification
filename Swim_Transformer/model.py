import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

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

def window_partition(x, window_size):
    """
    将feature map划分成windows
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    # [B, H//w, w, W//w, w, C]->[B*num_window, w, w, C]
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows

def window_reverse(windows, window_size, H, W):
    """
    将windows还原成feature map
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class PathchEmbed(nn.Module):
    def __init__(self, patch_size=4, embed_dim=96):
        super().__init__()
        patch_size = (patch_size, patch_size)
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.proj = nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)
    def forward(self, x):
        _, _, H, W = x.shape
        # 如果H,W部署patch_size的整数倍,需要进行padding
        pad_input = (H % self.patch_size[0] != 0) or (W % self.patch_size[1] != 0)
        if pad_input:
            x = F.pad(x, (0, self.patch_size[1] - W % self.patch_size[1],
                          0, self.patch_size[0] - H % self.patch_size[0],
                          0, 0))
        x = self.proj(x)
        _, _, H, W = x.shape
        # [B, C, H, W]->[B, HW, C]
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x, H, W

class PatchMerging(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = nn.LayerNorm(4 * dim)
    def forward(self, x, H, W):
        # x:[B, HW, C]
        B, L, C = x.shape
        x = x.view(B, H, W, C)
        pad_input = (H % 2 == 1) or (W % 2 == 1)
        if pad_input:
            x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))

        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3], -1) # [B, H/2, W/2, 4*C]
        x = x.view(B, -1, 4 * C) # [B, H/2*W/2, 4*C]
        x = self.norm(x)
        x = self.reduction(x) # [B, H/2*W/2, 2*C]
        return x

class WindowAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.window_size = window_size  # [w, w]
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.relative_position_bias_table = nn.Parameter(torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # [2*w-1 * 2*w-1, nH]
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing="ij"))  # [2, w, w]
        coords_flatten = torch.flatten(coords, 1)  # [2, w*w]
        # [2, w*w, 1] - [2, 1, w*w]
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # [2, w*w, w*w]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # [w*w, w*w, 2]
        relative_coords[:, :, 0] += self.window_size[0] - 1 
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # [w*w, w*w]
        self.register_buffer("relative_position_index", relative_position_index)
        
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        nn.init.trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(-1)
        
    def forward(self, x, mask=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)
        relative_position_bias = relative_position_bias.permute(2, 0 , 1).contiguous()
        attn += relative_position_bias.unsqueeze(0)
        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)
        attn = self.attn_drop(attn)
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

class SwinTransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, window_size=7, shift_size=0,
                 qkv_bias=True, drop=0., attn_drop=0., drop_path=0.):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, window_size=(window_size, window_size), num_heads=num_heads, qkv_bias=qkv_bias,
                                    attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, dim, drop=drop)

    def forward(self, x, attn_mask):
        H, W = self.H, self.W
        B, L, C = x.shape
        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)
        # 把feature map给pad到window size的整数倍
        pad_l = pad_t = 0
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
        _, Hp, Wp, _ = x.shape

        # SW-MSA/W-MSA
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x
            attn_mask = None
        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=attn_mask)
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, Hp, Wp)
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        # 去除padding
        if pad_r > 0 or pad_b > 0:
            x = x[:, :H, :W, :].contiguous()
        x = x.view(B, H * W, C)
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class BasicLayer(nn.Module):
    def __init__(self, dim, depth, num_heads, window_size, qkv_bias=True, 
                 drop=0., attn_drop=0., drop_path=0., downsample=None):
        super().__init__()
        self.dim = dim
        self.depth = depth
        self.window_size = window_size
        self.shift_size = window_size // 2
        self.blocks = nn.ModuleList([SwinTransformerBlock(dim, num_heads, window_size, shift_size=0 if (i % 2 == 0) else self.shift_size,
                                                          qkv_bias=qkv_bias, drop=drop, attn_drop=attn_drop, drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path)
                                                          for i in range(depth)])
        if downsample is not None:
            self.downsample = downsample(dim)
        else:
            self.downsample = None

    def create_mask(self, x, H, W):
        Hp = int(np.ceil(H / self.window_size)) * self.window_size
        Wp = int(np.ceil(W / self.window_size)) * self.window_size
        img_mask = torch.zeros((1, Hp, Wp, 1), device=x.device)
        h_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        # 给每个区域打标签[0,..., 8]
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt +=1 
        mask_windows = window_partition(img_mask, self.window_size) # [1*num_window, w, w, 1]
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size) # [num_window, w*w]
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2) # [num_window, w*w, w*w]
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        return attn_mask
        
    def forward(self, x, H, W):
        attn_mask = self.create_mask(x, H, W)
        for blk in self.blocks:
            blk.H, blk.W = H, W
            x = blk(x, attn_mask)
        if self.downsample is not None:
            x = self.downsample(x, H, W)
            H, W = (H + 1) // 2, (W + 1) // 2
        return x, H, W


class SwinTransformer(nn.Module):
    def __init__(self, patch_size=4, num_classes=1000, embed_dim=96, depths=(2, 2, 6, 2), num_heads=(3, 6, 12, 24), 
                 window_size=7, qkv_bias=True, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1, patch_norm=True):
        super().__init__()
        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        # stage4输出特征矩阵的channels
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.patch_embed = PathchEmbed(patch_size=patch_size, embed_dim=embed_dim)
        self.pos_drop = nn.Dropout(drop_rate)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layers = BasicLayer(dim=int(embed_dim * 2 ** i_layer), depth=depths[i_layer], num_heads=num_heads[i_layer],
                                window_size=window_size, qkv_bias=qkv_bias, drop=drop_rate, attn_drop=attn_drop_rate,
                                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],downsample=PatchMerging if (i_layer < self.num_layers - 1) else None)
            self.layers.append(layers)
        self.norm = nn.LayerNorm(self.num_features)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
    def forward(self, x):
        x, H, W = self.patch_embed(x)
        x = self.pos_drop(x)
        for layer in self.layers:
            x, H, W = layer(x, H, W)
        x = self.norm(x)
        x = self.avgpool(x.transpose(1, 2))
        x = torch.flatten(x, 1)
        x = self.head(x)
        return x
    
def swin_tiny_patch4_window7_224(num_classes=1000):
    # trained ImageNet-1K
    # https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth
    model = SwinTransformer(patch_size=4, window_size=7, embed_dim=96, depths=(2, 2, 6, 2), num_heads=(3, 6, 12, 24), num_classes=num_classes)
    return model


def swin_small_patch4_window7_224(num_classes=1000):
    # trained ImageNet-1K
    # https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_small_patch4_window7_224.pth
    model = SwinTransformer(patch_size=4, window_size=7, embed_dim=96, depths=(2, 2, 18, 2), num_heads=(3, 6, 12, 24), num_classes=num_classes)
    return model


def swin_base_patch4_window7_224(num_classes=1000):
    # trained ImageNet-1K
    # https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window7_224.pth
    model = SwinTransformer(patch_size=4, window_size=7, embed_dim=128, depths=(2, 2, 18, 2), num_heads=(4, 8, 16, 32), num_classes=num_classes)
    return model


def swin_base_patch4_window12_384(num_classes=1000):
    # trained ImageNet-1K
    # https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window12_384.pth
    model = SwinTransformer(patch_size=4, window_size=12, embed_dim=128, depths=(2, 2, 18, 2), num_heads=(4, 8, 16, 32), num_classes=num_classes)
    return model


def swin_base_patch4_window7_224_in22k(num_classes=1000):
    # trained ImageNet-22K
    # https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window7_224_22k.pth
    model = SwinTransformer(patch_size=4, window_size=7, embed_dim=128, depths=(2, 2, 18, 2), num_heads=(4, 8, 16, 32), num_classes=num_classes)
    return model


def swin_base_patch4_window12_384_in22k(num_classes=1000):
    # trained ImageNet-22K
    # https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window12_384_22k.pth
    model = SwinTransformer(patch_size=4, window_size=12, embed_dim=128, depths=(2, 2, 18, 2), num_heads=(4, 8, 16, 32), num_classes=num_classes)
    return model


def swin_large_patch4_window7_224_in22k(num_classes=1000):
    # trained ImageNet-22K
    # https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_large_patch4_window7_224_22k.pth
    model = SwinTransformer(patch_size=4, window_size=7, embed_dim=192, depths=(2, 2, 18, 2), num_heads=(6, 12, 24, 48), num_classes=num_classes)
    return model


def swin_large_patch4_window12_384_in22k(num_classes=1000):
    # trained ImageNet-22K
    # https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_large_patch4_window12_384_22k.pth
    model = SwinTransformer(patch_size=4, window_size=12, embed_dim=192, depths=(2, 2, 18, 2), num_heads=(6, 12, 24, 48), num_classes=num_classes)
    return model
