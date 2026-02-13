"""
UNet architecture for Diffusion Model on CIFAR10
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalPositionEmbeddings(nn.Module):
    """时间步的正弦位置编码"""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class Block(nn.Module):
    """基础卷积块"""
    def __init__(self, in_ch, out_ch, time_emb_dim, up=False):
        super().__init__()
        self.time_mlp = nn.Linear(time_emb_dim, out_ch)
        if up:
            self.conv1 = nn.Conv2d(2 * in_ch, out_ch, 3, padding=1)
            self.transform = nn.ConvTranspose2d(out_ch, out_ch, 4, 2, 1)
        else:
            self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
            self.transform = nn.Conv2d(out_ch, out_ch, 4, 2, 1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.bnorm1 = nn.GroupNorm(8, out_ch)
        self.bnorm2 = nn.GroupNorm(8, out_ch)
        self.relu = nn.SiLU()

    def forward(self, x, t):
        # 第一次卷积
        h = self.bnorm1(self.relu(self.conv1(x)))
        # 时间嵌入
        time_emb = self.relu(self.time_mlp(t))
        time_emb = time_emb[(...,) + (None,) * 2]
        h = h + time_emb
        # 第二次卷积
        h = self.bnorm2(self.relu(self.conv2(h)))
        # 上/下采样
        return self.transform(h)


class SelfAttention(nn.Module):
    """自注意力模块"""
    def __init__(self, channels, size):
        super().__init__()
        self.channels = channels
        self.size = size
        self.mha = nn.MultiheadAttention(channels, 4, batch_first=True)
        self.ln = nn.LayerNorm([channels])
        self.ff_self = nn.Sequential(
            nn.LayerNorm([channels]),
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels),
        )

    def forward(self, x):
        x = x.view(-1, self.channels, self.size * self.size).swapaxes(1, 2)
        x_ln = self.ln(x)
        attention_value, _ = self.mha(x_ln, x_ln, x_ln)
        attention_value = attention_value + x
        attention_value = self.ff_self(attention_value) + attention_value
        return attention_value.swapaxes(2, 1).view(-1, self.channels, self.size, self.size)


class UNet(nn.Module):
    """
    UNet架构用于Diffusion Model
    输入: (batch, 3, 32, 32) 图像 + (batch,) 时间步
    输出: (batch, 3, 32, 32) 预测的噪声
    """
    def __init__(self, c_in=3, c_out=3, time_dim=256):
        super().__init__()

        # 时间嵌入
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )

        # 初始卷积
        self.conv0 = nn.Conv2d(c_in, 64, 3, padding=1)

        # 下采样路径
        self.down1 = Block(64, 128, time_dim)
        self.sa1 = SelfAttention(128, 16)
        self.down2 = Block(128, 256, time_dim)
        self.sa2 = SelfAttention(256, 8)
        self.down3 = Block(256, 256, time_dim)
        self.sa3 = SelfAttention(256, 4)

        # Bottleneck
        self.bot1 = nn.Conv2d(256, 512, 3, padding=1)
        self.bot2 = nn.Conv2d(512, 512, 3, padding=1)
        self.bot3 = nn.Conv2d(512, 256, 3, padding=1)

        # 上采样路径
        self.up1 = Block(256, 128, time_dim, up=True)
        self.sa4 = SelfAttention(128, 8)
        self.up2 = Block(128, 64, time_dim, up=True)
        self.sa5 = SelfAttention(64, 16)
        self.up3 = Block(64, 64, time_dim, up=True)
        self.sa6 = SelfAttention(64, 32)

        # 输出层
        self.outc = nn.Conv2d(64, c_out, 1)

    def forward(self, x, t):
        # 时间嵌入
        t = self.time_mlp(t)

        # 初始卷积
        x1 = self.conv0(x)

        # 下采样
        x2 = self.down1(x1, t)
        x2 = self.sa1(x2)
        x3 = self.down2(x2, t)
        x3 = self.sa2(x3)
        x4 = self.down3(x3, t)
        x4 = self.sa3(x4)

        # Bottleneck
        x4 = F.silu(self.bot1(x4))
        x4 = F.silu(self.bot2(x4))
        x4 = F.silu(self.bot3(x4))

        # 上采样 + skip connections
        x = self.up1(torch.cat([x4, x4], dim=1), t)
        x = self.sa4(x)
        x = self.up2(torch.cat([x, x3], dim=1), t)
        x = self.sa5(x)
        x = self.up3(torch.cat([x, x2], dim=1), t)
        x = self.sa6(x)

        # 输出
        output = self.outc(x)
        return output


class UNetImproved(nn.Module):
    """
    改进版UNet，具有更好的结构
    """
    def __init__(self, in_channels=3, model_channels=128, out_channels=3,
                 num_res_blocks=2, attention_resolutions=(16, 8),
                 channel_mult=(1, 2, 2, 2), time_embed_dim=512):
        super().__init__()

        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels

        # 时间嵌入
        self.time_embed = nn.Sequential(
            SinusoidalPositionEmbeddings(model_channels),
            nn.Linear(model_channels, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

        # 输入层
        self.input_blocks = nn.ModuleList([
            nn.Conv2d(in_channels, model_channels, 3, padding=1)
        ])

        # 下采样
        channels = [model_channels]
        ch = model_channels
        ds = 1

        for level, mult in enumerate(channel_mult):
            out_ch = model_channels * mult
            for _ in range(num_res_blocks):
                layers = [ResBlock(ch, out_ch, time_embed_dim)]
                ch = out_ch
                if ds in attention_resolutions:
                    layers.append(AttentionBlock(ch))
                self.input_blocks.append(nn.ModuleList(layers))
                channels.append(ch)
            if level != len(channel_mult) - 1:
                self.input_blocks.append(nn.ModuleList([Downsample(ch)]))
                channels.append(ch)
                ds *= 2

        # 中间层
        self.middle_block = nn.ModuleList([
            ResBlock(ch, ch, time_embed_dim),
            AttentionBlock(ch),
            ResBlock(ch, ch, time_embed_dim),
        ])

        # 上采样
        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            out_ch = model_channels * mult
            for i in range(num_res_blocks + 1):
                skip_ch = channels.pop()
                layers = [ResBlock(ch + skip_ch, out_ch, time_embed_dim)]
                ch = out_ch
                if ds in attention_resolutions:
                    layers.append(AttentionBlock(ch))
                if level and i == num_res_blocks:
                    layers.append(Upsample(ch))
                    ds //= 2
                self.output_blocks.append(nn.ModuleList(layers))

        # 输出层
        self.out = nn.Sequential(
            nn.GroupNorm(32, ch),
            nn.SiLU(),
            nn.Conv2d(ch, out_channels, 3, padding=1),
        )

    def forward(self, x, t):
        emb = self.time_embed(t)

        hs = []
        h = x
        for module in self.input_blocks:
            if isinstance(module, nn.Conv2d):
                h = module(h)
            else:
                for layer in module:
                    if isinstance(layer, ResBlock):
                        h = layer(h, emb)
                    else:
                        h = layer(h)
            hs.append(h)

        for layer in self.middle_block:
            if isinstance(layer, ResBlock):
                h = layer(h, emb)
            else:
                h = layer(h)

        for module in self.output_blocks:
            h = torch.cat([h, hs.pop()], dim=1)
            for layer in module:
                if isinstance(layer, ResBlock):
                    h = layer(h, emb)
                else:
                    h = layer(h)

        return self.out(h)


class ResBlock(nn.Module):
    """残差块"""
    def __init__(self, in_channels, out_channels, time_emb_dim):
        super().__init__()
        self.norm1 = nn.GroupNorm(32, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.time_emb = nn.Linear(time_emb_dim, out_channels)
        self.norm2 = nn.GroupNorm(32, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)

        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x, emb):
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.time_emb(F.silu(emb))[:, :, None, None]
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.shortcut(x)


class AttentionBlock(nn.Module):
    """注意力块"""
    def __init__(self, channels):
        super().__init__()
        self.norm = nn.GroupNorm(32, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj = nn.Conv2d(channels, channels, 1)
        self.scale = channels ** -0.5

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.qkv(self.norm(x))
        q, k, v = qkv.reshape(b, 3, c, h * w).unbind(1)
        attn = torch.einsum('bci,bcj->bij', q, k) * self.scale
        attn = attn.softmax(dim=-1)
        out = torch.einsum('bij,bcj->bci', attn, v)
        out = out.reshape(b, c, h, w)
        return x + self.proj(out)


class Downsample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        return self.conv(x)
