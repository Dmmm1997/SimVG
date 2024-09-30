from functools import partial
from collections import OrderedDict

import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange
from einops.layers.torch import Rearrange
from timm.models.layers import DropPath, trunc_normal_

from .utils import FrozenBatchNorm2d, to_2tuple
from simvg.models import VIS_ENCODERS
from simvg.utils import get_root_logger, is_main
from mmcv.runner import force_fp32


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.0):
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


class Attention(nn.Module):
    def __init__(
        self,
        dim_in,
        dim_out,
        num_heads,
        qkv_bias=False,
        attn_drop=0.0,
        proj_drop=0.0,
        method="dw_bn",
        kernel_size=3,
        stride_kv=1,
        stride_q=1,
        padding_kv=1,
        padding_q=1,
        with_cls_token=True,
        freeze_bn=False,
        **kwargs,
    ):
        super().__init__()
        self.stride_kv = stride_kv
        self.stride_q = stride_q
        self.dim = dim_out
        self.num_heads = num_heads
        # head_dim = self.qkv_dim // num_heads
        self.scale = dim_out**-0.5
        self.with_cls_token = with_cls_token
        if freeze_bn:
            conv_proj_post_norm = FrozenBatchNorm2d
        else:
            conv_proj_post_norm = nn.BatchNorm2d

        self.conv_proj_q = self._build_projection(
            dim_in, dim_out, kernel_size, padding_q, stride_q, "linear" if method == "avg" else method, conv_proj_post_norm
        )
        self.conv_proj_k = self._build_projection(dim_in, dim_out, kernel_size, padding_kv, stride_kv, method, conv_proj_post_norm)
        self.conv_proj_v = self._build_projection(dim_in, dim_out, kernel_size, padding_kv, stride_kv, method, conv_proj_post_norm)

        self.proj_q = nn.Linear(dim_in, dim_out, bias=qkv_bias)
        self.proj_k = nn.Linear(dim_in, dim_out, bias=qkv_bias)
        self.proj_v = nn.Linear(dim_in, dim_out, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim_out, dim_out)
        self.proj_drop = nn.Dropout(proj_drop)

    def _build_projection(self, dim_in, dim_out, kernel_size, padding, stride, method, norm):
        if method == "dw_bn":
            proj = nn.Sequential(
                OrderedDict(
                    [
                        ("conv", nn.Conv2d(dim_in, dim_in, kernel_size=kernel_size, padding=padding, stride=stride, bias=False, groups=dim_in)),
                        ("bn", norm(dim_in)),
                        ("rearrage", Rearrange("b c h w -> b (h w) c")),
                    ]
                )
            )
        elif method == "avg":
            proj = nn.Sequential(
                OrderedDict(
                    [
                        ("avg", nn.AvgPool2d(kernel_size=kernel_size, padding=padding, stride=stride, ceil_mode=True)),
                        ("rearrage", Rearrange("b c h w -> b (h w) c")),
                    ]
                )
            )
        elif method == "linear":
            proj = None
        else:
            raise ValueError("Unknown method ({})".format(method))

        return proj

    def forward_conv(self, x, t_h, t_w, s_h, s_w):
        template, search = torch.split(x, [t_h * t_w, s_h * s_w], dim=1)
        template = rearrange(template, "b (h w) c -> b c h w", h=t_h, w=t_w).contiguous()
        search = rearrange(search, "b (h w) c -> b c h w", h=s_h, w=s_w).contiguous()

        if self.conv_proj_q is not None:
            t_q = self.conv_proj_q(template)
            s_q = self.conv_proj_q(search)
            q = torch.cat([t_q, s_q], dim=1)
        else:
            t_q = rearrange(template, "b c h w -> b (h w) c").contiguous()
            s_q = rearrange(search, "b c h w -> b (h w) c").contiguous()
            q = torch.cat([t_q, s_q], dim=1)

        if self.conv_proj_k is not None:
            t_k = self.conv_proj_k(template)
            s_k = self.conv_proj_k(search)
            k = torch.cat([t_k, s_k], dim=1)
        else:
            t_k = rearrange(template, "b c h w -> b (h w) c").contiguous()
            s_k = rearrange(search, "b c h w -> b (h w) c").contiguous()
            k = torch.cat([t_k, s_k], dim=1)

        if self.conv_proj_v is not None:
            t_v = self.conv_proj_v(template)
            s_v = self.conv_proj_v(search)
            v = torch.cat([t_v, s_v], dim=1)
        else:
            t_v = rearrange(template, "b c h w -> b (h w) c").contiguous()
            s_v = rearrange(search, "b c h w -> b (h w) c").contiguous()
            v = torch.cat([t_v, s_v], dim=1)

        return q, k, v

    def forward_conv_test(self, x, s_h, s_w):
        search = x
        search = rearrange(search, "b (h w) c -> b c h w", h=s_h, w=s_w).contiguous()

        if self.conv_proj_q is not None:
            q = self.conv_proj_q(search)
        else:
            q = rearrange(search, "b c h w -> b (h w) c").contiguous()

        if self.conv_proj_k is not None:
            k = self.conv_proj_k(search)
        else:
            k = rearrange(search, "b c h w -> b (h w) c").contiguous()
        k = torch.cat([self.t_k, self.ot_k, k], dim=1)

        if self.conv_proj_v is not None:
            v = self.conv_proj_v(search)
        else:
            v = rearrange(search, "b c h w -> b (h w) c").contiguous()
        v = torch.cat([self.t_v, self.ot_v, v], dim=1)

        return q, k, v

    def forward(self, x, t_h, t_w, s_h, s_w):
        """
        Asymmetric mixed attention.
        """
        if self.conv_proj_q is not None or self.conv_proj_k is not None or self.conv_proj_v is not None:
            q, k, v = self.forward_conv(x, t_h, t_w, s_h, s_w)

        q = rearrange(self.proj_q(q), "b t (h d) -> b h t d", h=self.num_heads).contiguous()
        k = rearrange(self.proj_k(k), "b t (h d) -> b h t d", h=self.num_heads).contiguous()
        v = rearrange(self.proj_v(v), "b t (h d) -> b h t d", h=self.num_heads).contiguous()

        # Attention!: k/v compression，1/4 of q_size（conv_stride=2）
        q_mt, q_s = torch.split(q, [t_h * t_w, s_h * s_w], dim=2)
        # k_t, k_ot, k_s = torch.split(k, [t_h*t_w//4, t_h*t_w//4, s_h*s_w//4], dim=2)
        # v_t, v_ot, v_s = torch.split(v, [t_h * t_w // 4, t_h * t_w // 4, s_h * s_w // 4], dim=2)
        k_mt, k_s = torch.split(k, [((t_h + 1) // 2) ** 2, s_h * s_w // 4], dim=2)
        v_mt, v_s = torch.split(v, [((t_h + 1) // 2) ** 2, s_h * s_w // 4], dim=2)

        # template attention
        attn_score = torch.einsum("bhlk,bhtk->bhlt", [q_mt, k_mt]) * self.scale
        attn = F.softmax(attn_score, dim=-1)
        attn = self.attn_drop(attn)
        x_mt = torch.einsum("bhlt,bhtv->bhlv", [attn, v_mt])
        x_mt = rearrange(x_mt, "b h t d -> b t (h d)")

        # search region attention
        attn_score = torch.einsum("bhlk,bhtk->bhlt", [q_s, k]) * self.scale
        attn = F.softmax(attn_score, dim=-1)
        attn = self.attn_drop(attn)
        x_s = torch.einsum("bhlt,bhtv->bhlv", [attn, v])
        x_s = rearrange(x_s, "b h t d -> b t (h d)")

        x = torch.cat([x_mt, x_s], dim=1)

        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class Block(nn.Module):
    def __init__(
        self,
        dim_in,
        dim_out,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        freeze_bn=False,
        **kwargs,
    ):
        super().__init__()

        self.with_cls_token = kwargs["with_cls_token"]

        self.norm1 = norm_layer(dim_in)
        self.attn = Attention(dim_in, dim_out, num_heads, qkv_bias, attn_drop, drop, freeze_bn=freeze_bn, **kwargs)

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim_out)

        dim_mlp_hidden = int(dim_out * mlp_ratio)
        self.mlp = Mlp(in_features=dim_out, hidden_features=dim_mlp_hidden, act_layer=act_layer, drop=drop)

    def forward(self, x, t_h, t_w, s_h, s_w):
        res = x

        x = self.norm1(x)
        attn = self.attn(x, t_h, t_w, s_h, s_w)
        x = res + self.drop_path(attn)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


class ConvEmbed(nn.Module):
    """Image to Conv Embedding"""

    def __init__(self, patch_size=7, in_chans=3, embed_dim=64, stride=4, padding=2, norm_layer=None):
        super().__init__()
        patch_size = to_2tuple(patch_size)
        self.patch_size = patch_size

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride, padding=padding)
        self.norm = norm_layer(embed_dim) if norm_layer else None

    def forward(self, x):
        x = self.proj(x)

        B, C, H, W = x.shape
        x = rearrange(x, "b c h w -> b (h w) c").contiguous()
        if self.norm:
            x = self.norm(x)
        x = rearrange(x, "b (h w) c -> b c h w", h=H, w=W).contiguous()

        return x


class VisionTransformer(nn.Module):
    """Vision Transformer with support for patch or hybrid CNN input stage"""

    def __init__(
        self,
        patch_size=16,
        patch_stride=16,
        patch_padding=0,
        in_chans=3,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=False,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        init="trunc_norm",
        freeze_bn=False,
        **kwargs,
    ):
        super().__init__()
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models

        self.rearrage = None

        self.patch_embed = ConvEmbed(
            # img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            stride=patch_stride,
            padding=patch_padding,
            embed_dim=embed_dim,
            norm_layer=norm_layer,
        )
        
        self.mlp = nn.Linear()

        with_cls_token = kwargs["with_cls_token"]
        if with_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        else:
            self.cls_token = None

        self.pos_drop = nn.Dropout(p=drop_rate)
        # stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        blocks = []
        for j in range(depth):
            blocks.append(
                Block(
                    dim_in=embed_dim,
                    dim_out=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[j],
                    act_layer=act_layer,
                    norm_layer=norm_layer,
                    freeze_bn=freeze_bn,
                    **kwargs,
                )
            )
        self.blocks = nn.ModuleList(blocks)

        if self.cls_token is not None:
            trunc_normal_(self.cls_token, std=0.02)

        if init == "xavier":
            self.apply(self._init_weights_xavier)
        else:
            self.apply(self._init_weights_trunc_normal)

    def _init_weights_trunc_normal(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def _init_weights_xavier(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, template, search):
        """
        :param template: (batch, c, 128, 128)
        :param search: (batch, c, 320, 320)
        :return:
        """
        # x = self.patch_embed(x)
        # B, C, H, W = x.size()
        template = self.patch_embed(template)
        t_B, t_C, t_H, t_W = template.size()
        # search = self.patch_embed(search)
        # s_B, s_C, s_H, s_W = search.size()

        template = rearrange(template, "b c h w -> b (h w) c").contiguous()
        # search = rearrange(search, "b c h w -> b (h w) c").contiguous()
        x = torch.cat([template, search], dim=1)

        x = self.pos_drop(x)

        for i, blk in enumerate(self.blocks):
            x = blk(x, t_H, t_W, s_H, s_W)

        # if self.cls_token is not None:
        #     cls_tokens, x = torch.split(x, [1, H*W], 1)
        template, search = torch.split(x, [t_H * t_W, s_H * s_W], dim=1)
        template = rearrange(template, "b (h w) c -> b c h w", h=t_H, w=t_W).contiguous()
        search = rearrange(search, "b (h w) c -> b c h w", h=s_H, w=s_W).contiguous()

        return template, search


spec_lib = {
    "cvt13": {
        "NUM_STAGES": 3,
        "PATCH_SIZE": [7, 3, 3],
        "PATCH_STRIDE": [4, 2, 2],
        "PATCH_PADDING": [2, 1, 1],
        "DIM_EMBED": [64, 192, 384],
        "NUM_HEADS": [1, 3, 6],
        "DEPTH": [1, 2, 10],
        "MLP_RATIO": [4.0, 4.0, 4.0],
        "ATTN_DROP_RATE": [0.0, 0.0, 0.0],
        "DROP_RATE": [0.0, 0.0, 0.0],
        "DROP_PATH_RATE": [0.0, 0.0, 0.1],
        "QKV_BIAS": [True, True, True],
        "CLS_TOKEN": [False, False, True],
        "POS_EMBED": [False, False, False],
        "QKV_PROJ_METHOD": ["dw_bn", "dw_bn", "dw_bn"],
        "KERNEL_QKV": [3, 3, 3],
        "PADDING_KV": [1, 1, 1],
        "STRIDE_KV": [2, 2, 2],
        "PADDING_Q": [1, 1, 1],
        "STRIDE_Q": [1, 1, 1],
        "FREEZE_BN": True,
    },
    "cvt21": {
        "NUM_STAGES": 3,
        "PATCH_SIZE": [7, 3, 3],
        "PATCH_STRIDE": [4, 2, 2],
        "PATCH_PADDING": [2, 1, 1],
        "DIM_EMBED": [64, 192, 384],
        "NUM_HEADS": [1, 3, 6],
        "DEPTH": [1, 4, 16],
        "MLP_RATIO": [4.0, 4.0, 4.0],
        "ATTN_DROP_RATE": [0.0, 0.0, 0.0],
        "DROP_RATE": [0.0, 0.0, 0.0],
        "DROP_PATH_RATE": [0.0, 0.0, 0.1],
        "QKV_BIAS": [True, True, True],
        "CLS_TOKEN": [False, False, False],
        "POS_EMBED": [False, False, False],
        "QKV_PROJ_METHOD": ["dw_bn", "dw_bn", "dw_bn"],
        "KERNEL_QKV": [3, 3, 3],
        "PADDING_KV": [1, 1, 1],
        "STRIDE_KV": [2, 2, 2],
        "PADDING_Q": [1, 1, 1],
        "STRIDE_Q": [1, 1, 1],
        "FREEZE_BN": True,
    },
}


@VIS_ENCODERS.register_module()
class ConvolutionalVisionTransformerMix(nn.Module):
    def __init__(
        self,
        type_name='cvt13',
        in_chans=3,
        act_layer=QuickGELU,
        norm_layer=partial(LayerNorm, eps=1e-5),
        init="trunc_norm",
        pretrained="",
        frozen_stages=-1,
    ):
        spec = spec_lib[type_name]
        super().__init__()
        # self.num_classes = num_classes

        self.num_stages = spec["NUM_STAGES"]
        self.pretrained = pretrained
        self.frozen_stages = frozen_stages
        for i in range(self.num_stages):
            kwargs = {
                "patch_size": spec["PATCH_SIZE"][i],
                "patch_stride": spec["PATCH_STRIDE"][i],
                "patch_padding": spec["PATCH_PADDING"][i],
                "embed_dim": spec["DIM_EMBED"][i],
                "depth": spec["DEPTH"][i],
                "num_heads": spec["NUM_HEADS"][i],
                "mlp_ratio": spec["MLP_RATIO"][i],
                "qkv_bias": spec["QKV_BIAS"][i],
                "drop_rate": spec["DROP_RATE"][i],
                "attn_drop_rate": spec["ATTN_DROP_RATE"][i],
                "drop_path_rate": spec["DROP_PATH_RATE"][i],
                "with_cls_token": spec["CLS_TOKEN"][i],
                "method": spec["QKV_PROJ_METHOD"][i],
                "kernel_size": spec["KERNEL_QKV"][i],
                "padding_q": spec["PADDING_Q"][i],
                "padding_kv": spec["PADDING_KV"][i],
                "stride_kv": spec["STRIDE_KV"][i],
                "stride_q": spec["STRIDE_Q"][i],
                "freeze_bn": spec["FREEZE_BN"],
            }

            stage = VisionTransformer(in_chans=in_chans, init=init, act_layer=act_layer, norm_layer=norm_layer, **kwargs)
            setattr(self, f"stage{i}", stage)

            in_chans = spec["DIM_EMBED"][i]

        dim_embed = spec["DIM_EMBED"][-1]
        self.norm = norm_layer(dim_embed)
        self.cls_token = spec["CLS_TOKEN"][-1]
        
        
        self.init_weights()

        # # Classifier head
        # self.head = nn.Linear(dim_embed, 1000)
        # trunc_normal_(self.head.weight, std=0.02)

    @force_fp32(apply_to=("x", "y"))
    def forward(self, x, y):
        """
        :param x: (b, 3, 480, 640)
        :param y: (b, 1024, 1, 1)
        :return:
        """

        res_x_list = []
        res_y_list = []
        for i in range(self.num_stages):
            x, y = getattr(self, f"stage{i}")(x, y)
            res_x_list.append(x)
            res_y_list.append(y)

        return res_x_list, res_y_list
    
    def init_weights(self):
        if len(self.pretrained)>0:
            ckpt_path = self.pretrained
            ckpt = torch.load(ckpt_path, map_location='cpu')
            missing_keys, unexpected_keys = self.load_state_dict(ckpt, strict=False)
            print("Load pretrained backbone checkpoint from:", ckpt_path)
            print("missing keys:", missing_keys)
            print("unexpected keys:", unexpected_keys)
            print("Loading pretrained CVT done.")


def get_mixformer_cvt(type_name, **kwargs):
    spec = spec_lib[type_name]
    msvit = ConvolutionalVisionTransformer(
        in_chans=3,
        act_layer=QuickGELU,
        norm_layer=partial(LayerNorm, eps=1e-5),
        init="trunc_norm",
        spec=spec,
    )

    # if pretrain:
    #     try:
    #         ckpt_path = pretrain_path
    #         ckpt = torch.load(ckpt_path, map_location='cpu')
    #         missing_keys, unexpected_keys = msvit.load_state_dict(ckpt, strict=False)
    #         print("Load pretrained backbone checkpoint from:", ckpt_path)
    #         print("missing keys:", missing_keys)
    #         print("unexpected keys:", unexpected_keys)
    #         print("Loading pretrained CVT done.")
    #     except:
    #         print("Warning: Pretrained CVT weights are not loaded")

    return msvit
