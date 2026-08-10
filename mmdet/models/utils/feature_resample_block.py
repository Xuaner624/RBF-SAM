import torch
from torch import Tensor, nn
from ..builder import BACKBONES
from .common import MLPBlock, LayerNorm2d
import torch.nn.functional as F
import einops


@BACKBONES.register_module()
class Feature_Resample_Block(nn.Module):
    def __init__(
            self,
            embedding_dim: int,
            num_groups: int,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.n_groups = num_groups
        self.n_group_channels = self.embedding_dim // self.n_groups
        self.dilations = [1, 2, 3, 4]
        self.offset_range_factors = [0.5, 1, 2, 4]
        self.conv_vit_features = nn.ModuleList([
            nn.Conv2d(
                embedding_dim, embedding_dim,
                kernel_size=3, stride=1,
                padding=d, dilation=d,
                groups=1,
                bias=False
            )
            for d in self.dilations
        ])

        self.conv_offsets = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(
                    self.n_group_channels, self.n_group_channels,
                    kernel_size=5, stride=1, padding=2,
                    groups=self.n_group_channels
                ),
                LayerNorm2d(self.n_group_channels),
                nn.GELU(),
                nn.Conv2d(self.n_group_channels, 2, 1, 1, 0, bias=False)
            )
            for _ in self.dilations
        ])

        self.offset_fuse_conv = nn.Conv2d(
            in_channels=len(self.dilations) * 2,
            out_channels=2,
            kernel_size=1,
            stride=1,
            bias=False
        )

        self.vit_fuse_conv = nn.Conv2d(
            in_channels=len(self.dilations) * embedding_dim,
            out_channels=embedding_dim,
            kernel_size=1, stride=1
        )

        # self.vis_monitor_layer = nn.Identity()

    @torch.no_grad()
    def _get_ref_points(self, H_key, W_key, B, dtype, device):
        ref_y, ref_x = torch.meshgrid(
            torch.linspace(0.5, H_key - 0.5, H_key, dtype=dtype, device=device),
            torch.linspace(0.5, W_key - 0.5, W_key, dtype=dtype, device=device),
            indexing='ij'
        )
        ref = torch.stack((ref_y, ref_x), -1)
        ref[..., 1].div_(W_key - 1.0).mul_(2.0).sub_(1.0)
        ref[..., 0].div_(H_key - 1.0).mul_(2.0).sub_(1.0)
        ref = ref[None, ...].expand(B * self.n_groups, -1, -1, -1)  # B * g H W 2
        return ref

    def forward(self, v: Tensor):  # -> Tensor
        # learn offset
        B, C, H, W = v.size()
        dtype, device = v.dtype, v.device
        vit_feature_list = []
        offset_list = []
        for conv_vit, conv_off, off_factor in zip(self.conv_vit_features, self.conv_offsets, self.offset_range_factors):
            vit_feature = conv_vit(v)
            vit_feature_list.append(vit_feature)
            vit_off = einops.rearrange(vit_feature, 'b (g c) h w -> (b g) c h w', g=self.n_groups, c=self.n_group_channels)
            offset = conv_off(vit_off)
            offset_range = torch.tensor([1.0 / (H - 1.0), 1.0 / (W - 1.0)],
                                        device=device).reshape(1, 2, 1, 1)
            offset = offset.tanh().mul(offset_range).mul(off_factor)
            offset_list.append(offset)

        offset_multi = torch.cat(offset_list, dim=1)
        offset_fused = self.offset_fuse_conv(offset_multi)
        offset_fused = einops.rearrange(offset_fused, 'b p h w -> b h w p')
        reference = self._get_ref_points(H, W, B, dtype, device)
        pos = (offset_fused + reference).clamp(-1., +1.)

        vit_feature_multi = torch.cat(vit_feature_list, dim=1)
        vit_feature_fused = self.vit_fuse_conv(vit_feature_multi)

        vit_sampled = F.grid_sample(
            input=vit_feature_fused.reshape(B * self.n_groups, self.n_group_channels, H, W),
            grid=pos[..., (1, 0)],  # y, x -> x, y
            mode='bilinear', align_corners=True)  # B * g, Cg, Hg, Wg
        vit_sampled = vit_sampled.reshape(B, H*W, C)  # 1,4096,256
        # vit_sampled = self.vis_monitor_layer(vit_sampled)

        return vit_sampled
