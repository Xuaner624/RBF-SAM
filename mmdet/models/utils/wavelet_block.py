import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from .common import LayerNorm2d
import math
from ..builder import BACKBONES


class DWT_Function(Function):
    @staticmethod
    def forward(ctx, x, w_ll, w_lh, w_hl, w_hh):
        # 保存小波滤波器权重，用于反向传播
        ctx.save_for_backward(w_ll, w_lh, w_hl, w_hh)
        # 保存输入的形状（反向时需要拆分通道）
        ctx.shape = x.shape

        dim = x.shape[1]
        # 分别用 LL/LH/HL/HH 4 个小波核做 stride=2 的卷积
        # expand(dim, -1, -1, -1) 表示按通道复制，使每个通道独立卷积（groups=dim）
        x_ll = torch.nn.functional.conv2d(x, w_ll.expand(dim, -1, -1, -1), stride=2, groups=dim)
        x_lh = torch.nn.functional.conv2d(x, w_lh.expand(dim, -1, -1, -1), stride=2, groups=dim)
        x_hl = torch.nn.functional.conv2d(x, w_hl.expand(dim, -1, -1, -1), stride=2, groups=dim)
        x_hh = torch.nn.functional.conv2d(x, w_hh.expand(dim, -1, -1, -1), stride=2, groups=dim)
        # 将四个子带在 channel 维拼接，输出通道 = 4*C
        x = torch.cat([x_ll, x_lh, x_hl, x_hh], dim=1)
        return x

    @staticmethod
    def backward(ctx, dx):
        if ctx.needs_input_grad[0]:
            # print(">>> DWT_Function backward called")
            w_ll, w_lh, w_hl, w_hh = ctx.saved_tensors
            _, C, _, _ = ctx.shape
            dx_ll, dx_lh, dx_hl, dx_hh = dx[:, :C], dx[:, C:C * 2], dx[:, C * 2:C * 3], dx[:, C * 3:]

            dx_x_ll = torch.nn.functional.conv_transpose2d(dx_ll, w_ll.expand(C, -1, -1, -1) * 4, stride=2, groups=C)
            dx_x_lh = torch.nn.functional.conv_transpose2d(dx_lh, w_lh.expand(C, -1, -1, -1) * 4, stride=2, groups=C)
            dx_x_hl = torch.nn.functional.conv_transpose2d(dx_hl, w_hl.expand(C, -1, -1, -1) * 4, stride=2, groups=C)
            dx_x_hh = torch.nn.functional.conv_transpose2d(dx_hh, w_hh.expand(C, -1, -1, -1) * 4, stride=2, groups=C)
            return dx_x_ll + dx_x_lh + dx_x_hl + dx_x_hh, None, None, None, None
        else:
            return dx, None, None, None, None


class DWT_2D(nn.Module):
    def __init__(self):
        super(DWT_2D, self).__init__()
        # -----------------------------
        # 定义 2×2 Haar 小波核（固定权重，不参与训练）
        # -----------------------------
        # LL：低频（平均）分量
        w_ll = torch.tensor([[[[0.25, 0.25], [0.25, 0.25]]]], dtype=torch.float32, requires_grad=False)
        # LH：水平高频（竖直方向变化）
        w_lh = torch.tensor([[[[0.25, 0.25], [-0.25, -0.25]]]], dtype=torch.float32, requires_grad=False)
        # HL：垂直高频（水平方向变化）
        w_hl = torch.tensor([[[[0.25, -0.25], [0.25, -0.25]]]], dtype=torch.float32, requires_grad=False)
        # HH：对角高频（斜向变化）
        w_hh = torch.tensor([[[[0.25, -0.25], [-0.25, 0.25]]]], dtype=torch.float32, requires_grad=False)
        # -----------------------------
        # register_buffer：保存为模型 buffer
        # 不会作为参数更新，但会随模型保存和加载
        # -----------------------------
        self.register_buffer('w_ll', w_ll)
        self.register_buffer('w_lh', w_lh)
        self.register_buffer('w_hl', w_hl)
        self.register_buffer('w_hh', w_hh)
        # 将核强制转为 float32（确保 forward 时精度一致）
        self.w_ll = w_ll.to(dtype=torch.float32)
        self.w_lh = w_lh.to(dtype=torch.float32)
        self.w_hl = w_hl.to(dtype=torch.float32)
        self.w_hh = w_hh.to(dtype=torch.float32)

    def forward(self, x):
        # -----------------------------
        # 使用自定义 DWT_Function 实现前向 + 反向传播
        # apply：自定义 autograd 函数
        # -----------------------------
        return DWT_Function.apply(x, self.w_ll, self.w_lh, self.w_hl, self.w_hh)


class IDWT_Function(Function):
    @staticmethod
    def forward(ctx, x, filters):
        ctx.save_for_backward(filters)
        ctx.shape = x.shape

        _, C, _, _ = x.shape
        w_ll, w_lh, w_hl, w_hh = torch.unbind(filters, dim=0)
        x_ll, x_lh, x_hl, x_hh = x[:, :C // 4], x[:, C // 4:C * 2 // 4], x[:, C * 2 // 4:C * 3 // 4], x[:, C * 3 // 4:]
        x_1_ll = torch.nn.functional.conv_transpose2d(x_ll, w_ll.expand(C // 4, -1, -1, -1), stride=2, groups=C // 4)
        x_1_lh = torch.nn.functional.conv_transpose2d(x_lh, w_lh.expand(C // 4, -1, -1, -1), stride=2, groups=C // 4)
        x_1_hl = torch.nn.functional.conv_transpose2d(x_hl, w_hl.expand(C // 4, -1, -1, -1), stride=2, groups=C // 4)
        x_1_hh = torch.nn.functional.conv_transpose2d(x_hh, w_hh.expand(C // 4, -1, -1, -1), stride=2, groups=C // 4)
        return x_1_ll + x_1_lh + x_1_hl + x_1_hh

    @staticmethod
    def backward(ctx, dx):
        if ctx.needs_input_grad[0]:
            # print(">>> IDWT_Function backward called")
            filters = ctx.saved_tensors
            filters = filters[0]
            _, C, _, _ = ctx.shape
            C //= 4

            w_ll, w_lh, w_hl, w_hh = torch.unbind(filters, dim=0)
            x_ll = torch.nn.functional.conv2d(dx, w_ll.unsqueeze(1).expand(C, -1, -1, -1) / 4, stride=2, groups=C)
            x_lh = torch.nn.functional.conv2d(dx, w_lh.unsqueeze(1).expand(C, -1, -1, -1) / 4, stride=2, groups=C)
            x_hl = torch.nn.functional.conv2d(dx, w_hl.unsqueeze(1).expand(C, -1, -1, -1) / 4, stride=2, groups=C)
            x_hh = torch.nn.functional.conv2d(dx, w_hh.unsqueeze(1).expand(C, -1, -1, -1) / 4, stride=2, groups=C)
            dx = torch.cat([x_ll, x_lh, x_hl, x_hh], dim=1)
        return dx, None


class IDWT_2D(nn.Module):
    def __init__(self):
        super(IDWT_2D, self).__init__()
        w_ll = torch.tensor([[[[1, 1], [1, 1]]]], dtype=torch.float32, requires_grad=False)
        w_lh = torch.tensor([[[[1, 1], [-1, -1]]]], dtype=torch.float32, requires_grad=False)
        w_hl = torch.tensor([[[[1, -1], [1, -1]]]], dtype=torch.float32, requires_grad=False)
        w_hh = torch.tensor([[[[1, -1], [-1, 1]]]], dtype=torch.float32, requires_grad=False)

        filters = torch.cat([w_ll, w_lh, w_hl, w_hh], dim=0)
        self.register_buffer('filters', filters)
        self.filters = filters

    def forward(self, x):
        return IDWT_Function.apply(x, self.filters)


class Attention(nn.Module):
    def __init__(self, channel, num_heads, dropout):
        super(Attention, self).__init__()
        self.num_heads, self.channel = num_heads, channel
        self.head_channel = channel // num_heads
        self.scale = self.head_channel ** 0.5

        self.q_proj = nn.Sequential(
            nn.LayerNorm(channel),
            nn.Linear(channel, channel, bias=True),
            nn.GELU(),
            nn.Linear(channel, channel, bias=True),
            nn.Dropout(dropout),
        )
        self.k_proj = nn.Sequential(
            nn.LayerNorm(channel),
            nn.Linear(channel, channel, bias=True),
            nn.GELU(),
            nn.Linear(channel, channel, bias=True),
            nn.Dropout(dropout),
        )
        self.v_proj = nn.Sequential(
            nn.LayerNorm(channel),
            nn.Linear(channel, channel, bias=True),
            nn.GELU(),
            nn.Linear(channel, channel, bias=True),
            nn.Dropout(dropout),
        )

        self.mlp = nn.Sequential(
            nn.Linear(channel, channel, bias=True),
            nn.GELU(),
            nn.Linear(channel, channel, bias=True),
            nn.Dropout(dropout),
        )
        self.norm = nn.LayerNorm(channel)
        self.proj = nn.Linear(channel, channel)

    def forward(self, q, k, v):
        B, q_C, H, W = q.shape
        _, v_C, _, _ = v.shape
        shortcut = v
        q_attn = self.q_proj(q.permute(0, 2, 3, 1).reshape(B, -1, q_C)).reshape(B, -1, self.num_heads,
                                                                                self.head_channel).permute(0, 2, 1, 3)
        k_attn = self.k_proj(k.permute(0, 2, 3, 1).reshape(B, -1, q_C)).reshape(B, -1, self.num_heads,
                                                                                self.head_channel).permute(0, 2, 3, 1)
        v_attn_1 = self.v_proj(v.permute(0, 2, 3, 1).reshape(B, -1, v_C))
        v_attn = v_attn_1.reshape(B, -1, self.num_heads, self.head_channel).permute(0, 2, 1, 3)
        attn = ((q_attn @ k_attn) / self.scale).softmax(dim=-1)
        x = (attn @ v_attn).permute(0, 2, 1, 3).reshape(B, -1, v_C)
        x = self.proj(x)
        rs1 = self.norm(shortcut.permute(0, 2, 3, 1).reshape(B, -1, v_C) + x).permute(0, 2, 1).reshape(B, v_C, H, W)
        rs2 = rs1 + self.mlp(rs1.permute(0, 2, 3, 1).reshape(B, -1, v_C)).permute(0, 2, 1).reshape(B, v_C, H, W)
        return rs2


class Wavelet_Attention(nn.Module):
    def __init__(self, embedding_dim, num_heads, dropout):
        super(Wavelet_Attention, self).__init__()
        self.embedding_dim = embedding_dim
        self.wavelet_transform = DWT_2D()

        self.v_ll_attn = Attention(channel=embedding_dim, num_heads=num_heads, dropout=dropout)
        self.v_lh_attn = Attention(channel=embedding_dim, num_heads=num_heads, dropout=dropout)
        self.v_hl_attn = Attention(channel=embedding_dim, num_heads=num_heads, dropout=dropout)
        self.v_hh_attn = Attention(channel=embedding_dim, num_heads=num_heads, dropout=dropout)

        self.attention_idwt = IDWT_2D()
        self.norm = LayerNorm2d(embedding_dim)

    def forward(self, vit_early_embedding):
        wt_ll, wt_lh, wt_hl, wt_hh = torch.split(self.wavelet_transform(vit_early_embedding),
                                                 [self.embedding_dim, self.embedding_dim,
                                                  self.embedding_dim, self.embedding_dim], dim=1)

        v_ll = self.v_ll_attn(q=wt_ll, k=wt_ll, v=wt_ll)
        v_lh = self.v_lh_attn(q=wt_lh, k=wt_ll, v=wt_ll)
        v_hl = self.v_hl_attn(q=wt_hl, k=wt_ll, v=wt_ll)
        v_hh = self.v_hh_attn(q=wt_hh, k=wt_ll, v=wt_ll)

        attention_idwt_x = self.attention_idwt(torch.cat([v_ll, v_lh, v_hl, v_hh], dim=1))
        attention_idwt_x = self.norm(attention_idwt_x)
        x = attention_idwt_x + vit_early_embedding

        return x, v_ll, v_lh, v_hl, v_hh


class WaveletOffsetToken(nn.Module):
    def __init__(self, in_channels, token_dim=256, down=2):
        super().__init__()

        # --- 四个子带对应的 conv ---
        self.conv_ll = nn.Conv2d(in_channels, 1, kernel_size=3, padding=1)
        self.conv_lh = nn.Conv2d(in_channels, 1, kernel_size=3, padding=1)
        self.conv_hl = nn.Conv2d(in_channels, 1, kernel_size=3, padding=1)
        self.conv_hh = nn.Conv2d(in_channels, 1, kernel_size=3, padding=1)

        # --- 下采样 ---
        self.pool = nn.MaxPool2d(kernel_size=down, stride=down)

        self.token_dim = token_dim

        # --- 为每个子带加一个 proj ---
        self.proj_ll = nn.Linear(token_dim, token_dim)
        self.proj_lh = nn.Linear(token_dim, token_dim)
        self.proj_hl = nn.Linear(token_dim, token_dim)
        self.proj_hh = nn.Linear(token_dim, token_dim)

    def _encode_single(self, x, conv, proj):
        t = conv(x)
        t = self.pool(t)
        t = t.flatten(1)

        # 尺寸检查
        assert t.size(1) == self.token_dim, \
            f"Flattened token dim {t.size(1)} != token_dim {self.token_dim}"

        t = proj(t)

        return t

    def forward(self, wt_ll, wt_lh, wt_hl, wt_hh):
        # --- 四个子带分别生成 token ---
        t_ll = self._encode_single(wt_ll, self.conv_ll, self.proj_ll)
        t_lh = self._encode_single(wt_lh, self.conv_lh, self.proj_lh)
        t_hl = self._encode_single(wt_hl, self.conv_hl, self.proj_hl)
        t_hh = self._encode_single(wt_hh, self.conv_hh, self.proj_hh)

        # 堆叠为 [B, 4, token_dim]
        return torch.stack([t_ll, t_lh, t_hl, t_hh], dim=1)


@BACKBONES.register_module()
class Wavelet_Block(nn.Module):
    def __init__(self, in_chans, embed_dim, num_heads, token_dim, dropout):
        super(Wavelet_Block, self).__init__()
        self.neck = nn.Sequential(
            nn.Conv2d(in_chans, embed_dim, kernel_size=1, bias=False),
            LayerNorm2d(embed_dim),
            nn.Conv2d(embed_dim, embed_dim,kernel_size=3, padding=1, bias=False),
            LayerNorm2d(embed_dim),
        )
        self.wavelet_block = Wavelet_Attention(embedding_dim=embed_dim, num_heads=num_heads, dropout=dropout)
        self.wavelet_token = WaveletOffsetToken(in_channels=embed_dim, token_dim=token_dim)
        self.early_enhance = nn.Sequential(
            nn.ConvTranspose2d(embed_dim, embed_dim // 4, kernel_size=2, stride=2),
            LayerNorm2d(embed_dim // 4),
            nn.GELU(),
            nn.ConvTranspose2d(embed_dim // 4, embed_dim // 8, kernel_size=2, stride=2))

        self.final_enhance = nn.Sequential(
            nn.ConvTranspose2d(embed_dim, embed_dim // 4, kernel_size=2, stride=2),
            LayerNorm2d(embed_dim // 4),
            nn.GELU(),
            nn.ConvTranspose2d(embed_dim // 4, embed_dim // 8, kernel_size=2, stride=2),
        )

    def forward(self, vit_early_embedding, vit_final_embedding):
        vit_early_embedding = self.neck(vit_early_embedding)
        enhanced_early, wt_ll, wt_lh, wt_hl, wt_hh = self.wavelet_block(vit_early_embedding)
        wavelet_token_feature = self.wavelet_token(wt_ll, wt_lh, wt_hl, wt_hh)
        enhanced_feature = self.final_enhance(vit_final_embedding) + self.early_enhance(enhanced_early)
        return enhanced_feature, wavelet_token_feature
