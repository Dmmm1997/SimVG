from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm.models.vision_transformer
from einops import rearrange
from timm.models.layers import DropPath, Mlp

from .utils import get_2d_sincos_pos_embed

from .utils import FrozenBatchNorm2d, to_2tuple
from simvg.models import VIS_ENCODERS
from simvg.utils import get_root_logger, is_main
from mmcv.runner import force_fp32
from functools import partial
import numpy as np
from timm.layers import resample_abs_pos_embed
from timm.models.vision_transformer import _load_weights

class PatchEmbed(nn.Module):
    """ 2D Image to Patch Embedding
    """
    def __init__(self, patch_size=16, in_chans=3, embed_dim=768, norm_layer=None, flatten=True):
        super().__init__()
        patch_size = to_2tuple(patch_size)
        self.flatten = flatten

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2).contiguous()  # BCHW -> BNC
        x = self.norm(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.qkv_mem = None

    def forward(self, x, image_tokens, text_tokens):
        """
        x is a concatenated vector of template and search region features.
        """
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)   # make torchscript happy (cannot use tensor as tuple)

        q_mt, q_s = torch.split(q, [text_tokens, image_tokens], dim=2)
        k_mt, k_s = torch.split(k, [text_tokens, image_tokens], dim=2)
        v_mt, v_s = torch.split(v, [text_tokens, image_tokens], dim=2)

        # asymmetric mixed attention
        attn = (q_mt @ k_mt.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x_mt = (attn @ v_mt).transpose(1, 2).reshape(B, text_tokens, C)

        attn = (q_s @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x_s = (attn @ v).transpose(1, 2).reshape(B, image_tokens, C)

        x = torch.cat([x_mt, x_s], dim=1)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    def __init__(
            self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
            drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x, image_tokens, text_tokens):
        x = x + self.drop_path1(self.attn(self.norm1(x), image_tokens, text_tokens))
        x = x + self.drop_path2(self.mlp(self.norm2(x)))
        return x
    

@VIS_ENCODERS.register_module()
class VisionTransformerMix(timm.models.vision_transformer.VisionTransformer):
    """ Vision Transformer with support for global average pooling
    """
    def __init__(self, pretrain=None, frozen_stages=None,
                 img_size_s=(384,384), patch_size=16, in_chans=3, num_classes=1000, embed_dim=768,
                 depth=12, num_heads=12, mlp_ratio=4., qkv_bias=True, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0.1, weight_init='jax', embed_layer=PatchEmbed, norm_layer=partial(nn.LayerNorm, eps=1e-6), act_layer=None):
        super(VisionTransformerMix, self).__init__(img_size=img_size_s[0], patch_size=patch_size, in_chans=in_chans,
                                                num_classes=num_classes, embed_dim=embed_dim, depth=depth,
                                                num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                                                drop_rate=drop_rate, attn_drop_rate=attn_drop_rate,
                                                drop_path_rate=drop_path_rate, weight_init=weight_init,
                                                norm_layer=norm_layer, act_layer=act_layer)

        # self.patch_embed = embed_layer(
        #     patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.Sequential(*[
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i],
                norm_layer=norm_layer) for i in range(depth)])
        self.pretrain = pretrain
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.frozen_stages = frozen_stages

        self.grid_size = [img_size_s[0] // patch_size, img_size_s[1] // patch_size]
        self.num_patches = self.grid_size[0]* self.grid_size[1]
        self.pos_embed_img = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))
        self.pos_embed_text = nn.Parameter(torch.zeros(1, 1, embed_dim))

        self.init_pos_embed()

        if weight_init != 'skip':
            self.init_weights(weight_init)
        self.init_pretrain_checkpoints()
        
    def train(self, mode=True):
        """Convert the model into training mode while keep layers freezed."""
        super(VisionTransformerMix, self).train(mode)
        self._freeze_stages()
        
    def _freeze_stages(self):
        if self.frozen_stages>=0:
            for i in range(1, self.frozen_stages + 1):
                m = self.blocks[i - 1]
                m.eval()
                for param in m.parameters():
                    param.requires_grad = False
        

    def init_pos_embed(self):
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed_img = get_2d_sincos_pos_embed(self.embed_dim, self.grid_size,
                                            cls_token=False)
        
        self.pos_embed_img.data.copy_(torch.from_numpy(pos_embed_img).float().unsqueeze(0))
        
        # pos_embed_text = torch.zeros(text_feat)
        
        # self.pos_embed_text.data.copy_(torch.from_numpy(pos_embed_text).float().unsqueeze(0))

        # pos_embed_text = get_2d_sincos_pos_embed(text_feat.shape[1], (text_feat.shape[1], 1),
        #                                       cls_token=False)
        # pos_embed_text = torch.from_numpy(pos_embed_text).float().unsqueeze(0)
        
    def get_pos_embed(self, H, W):
        pos_embed = resample_abs_pos_embed(
            self.pos_embed_img,
            (H, W),
            num_prefix_tokens=0,
        )
        return pos_embed
        
    @force_fp32(apply_to=("img","text_feat"))
    def forward(self, img, text_feat):
        """
        :param text_feat: (batch, 1, c)
        :param img: (batch, 3, 288, 288)
        :return:
        """
        # import time
        # start = time.time()
        img_feat = self.patch_embed(img)  # BCHW-->BNC
        img_shape = (img.shape[-2]//self.patch_size, img.shape[-1]//self.patch_size)
        # text_feat = text_feat
        B, C = img_feat.size(0), img_feat.size(-1)
        H_img, W_img = img_shape
        
        pos_embed_img = self.get_pos_embed(H_img, W_img)

        img_feat = img_feat + pos_embed_img
        text_feat = text_feat + self.pos_embed_text
        img_tokens = img_feat.shape[1]
        text_tokens = text_feat.shape[1]
        x = torch.cat([text_feat, img_feat], dim=1)
        x = self.pos_drop(x)

        # print("pre_time:{}".format(time.time()-start))
        # start = time.time()
        for blk in self.blocks:
            x = blk(x, text_tokens, img_tokens)
            
        # print("block time:{}".format(time.time()-start))

        text_feat, img_feat = torch.split(x, [text_tokens, img_tokens], dim=1)

        img_feat = img_feat.transpose(1, 2).reshape(B, C, H_img, W_img)
        text_feat = text_feat.transpose(1, 2).reshape(B, text_tokens, C)

        return img_feat
    
    def init_pretrain_checkpoints(self):
        logger = get_root_logger()
        ckpt_path = self.pretrain
        if ckpt_path is not None and len(ckpt_path)>0:
            if ckpt_path.endswith(".pth") or ckpt_path.endswith(".pth.tar"):
                ckpt = torch.load(ckpt_path, map_location='cpu')
                new_dict = {}
                for k, v in ckpt.items():
                    if 'pos_embed' not in k and 'mask_token' not in k:    # use fixed pos embed
                        new_dict[k] = v
                missing_keys, unexpected_keys = self.load_state_dict(new_dict, strict=False)
                if is_main():
                    logger.info("Load pretrained backbone checkpoint from:{}".format(ckpt_path))
                    logger.info("missing keys:{}".format(missing_keys))
                    logger.info("unexpected keys:{}".format(unexpected_keys))
                    logger.info("Loading pretrained ViT done.")
            elif ckpt_path.endswith(".npz"):
                self.load_pretrained(ckpt_path)
            else:
                logger.info("checkpoint format is not correct!!!")
                
                
                

