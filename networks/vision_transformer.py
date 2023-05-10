# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import logging
import math
from einops import rearrange
from os.path import join as pjoin
import torch
import torch.nn as nn
import numpy as np
from utils import get_mask,get_patch_mask
from torch.nn import CrossEntropyLoss, Dropout, Softmax, Linear, Conv2d, LayerNorm
from torch.nn.modules.utils import _pair
from scipy import ndimage
from .swin_transformer_unet_skip_expand_decoder_sys import SwinTransformerSys

logger = logging.getLogger(__name__)

class FinalPatchExpand_X4(nn.Module):
    def __init__(self, input_resolution, dim, dim_scale=4, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.dim_scale = dim_scale
        self.expand = nn.Linear(dim, 16*dim, bias=False)
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
        x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=self.dim_scale, p2=self.dim_scale, c=C//(self.dim_scale**2))
        x = x.view(B,-1,self.output_dim)
        x= self.norm(x)

        return x

class SwinUnet(nn.Module):
    def __init__(self, config, img_size=224, num_classes=21843, mask_rate=0.25,mask_size=4,zero_head=False, vis=False):
        super(SwinUnet, self).__init__()
        self.num_classes = num_classes
        self.zero_head = zero_head
        self.config = config
        self.mask_rate=mask_rate
        self.mask_size=mask_size
        self.swin_unet = SwinTransformerSys(img_size=config.DATA.IMG_SIZE,
                                patch_size=config.MODEL.SWIN.PATCH_SIZE,
                                in_chans=config.MODEL.SWIN.IN_CHANS,
                                num_classes=self.num_classes,
                                embed_dim=config.MODEL.SWIN.EMBED_DIM,
                                depths=config.MODEL.SWIN.DEPTHS,
                                num_heads=config.MODEL.SWIN.NUM_HEADS,
                                window_size=config.MODEL.SWIN.WINDOW_SIZE,
                                mlp_ratio=config.MODEL.SWIN.MLP_RATIO,
                                qkv_bias=config.MODEL.SWIN.QKV_BIAS,
                                qk_scale=config.MODEL.SWIN.QK_SCALE,
                                drop_rate=config.MODEL.DROP_RATE,
                                drop_path_rate=config.MODEL.DROP_PATH_RATE,
                                ape=config.MODEL.SWIN.APE,
                                patch_norm=config.MODEL.SWIN.PATCH_NORM,
                                use_checkpoint=config.TRAIN.USE_CHECKPOINT)

        if self.swin_unet.final_upsample == "expand_first":
            print("---final upsample expand_first---")
            self.up = FinalPatchExpand_X4(input_resolution=(img_size//self.swin_unet.patch_size,img_size//self.swin_unet.patch_size),dim_scale=4,dim=self.swin_unet.embed_dim)
            "分割任务层"
            self.output = nn.Conv2d(in_channels=self.swin_unet.embed_dim,out_channels=self.num_classes,kernel_size=1,bias=False)

            "重构任务层"
            self.output1 = nn.Conv2d(in_channels=self.swin_unet.embed_dim,out_channels=self.swin_unet.in_channel,kernel_size=1,bias=False)
            #self.output1 = nn.Linear(in_features=self.swin_unet.embed_dim*img_size*img_size,out_features=self.swin_unet.in_channel*self.swin_unet.patch_size*self.swin_unet.patch_size)

    def up_x4_seg(self, x):
        H, W = self.swin_unet.patches_resolution
        B, L, C = x.shape
        assert L == H * W, "input features has wrong size"

        if self.swin_unet.final_upsample == "expand_first":
            x = self.up(x)
            x = x.view(B, 4 * H, 4 * W, -1)
            x = x.permute(0, 3, 1, 2)  # B,C,H,W
            x = self.output(x)
        return x

    def up_x4_mask(self, x):
        H, W = self.swin_unet.patches_resolution
        B, L, C = x.shape
        assert L == H * W, "input features has wrong size"

        if self.swin_unet.final_upsample == "expand_first":
            x = self.up(x)
            x = x.view(B, 4 * H, 4 * W, -1)
            x = x.permute(0, 3, 1, 2)  # 调整张量为B,C,H,W
            #x = self.flatten(x)  # B,embed_dim*H*W
            x = self.output1(x)

            #x = x.view(B, self.in_channel, H, W)
        return x

    def forward(self, x):
        if x.size()[1] == 1:
            x = x.repeat(1,3,1,1)

        "掩码矩阵"
        maskid = get_patch_mask(x,self.mask_rate,self.mask_size)
        "掩码图像"
        x_mask = x * maskid

        "分割任务"
        x = self.swin_unet(x)
        logits0 = self.up_x4_seg(x)

        "掩码重构标签任务"
        x_masks = self.swin_unet(x_mask)
        logits1 = self.up_x4_seg(x_masks)

        return logits0,logits1,x_mask

    def load_from(self, config):
        pretrained_path = config.MODEL.PRETRAIN_CKPT
        if pretrained_path is not None:
            print("pretrained_path:{}".format(pretrained_path))
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            pretrained_dict = torch.load(pretrained_path, map_location=device)
            if "model"  not in pretrained_dict:
                print("---start load pretrained modle by splitting---")
                pretrained_dict = {k[17:]:v for k,v in pretrained_dict.items()}
                for k in list(pretrained_dict.keys()):
                    if "output" in k:
                        print("delete key:{}".format(k))
                        del pretrained_dict[k]
                msg = self.swin_unet.load_state_dict(pretrained_dict,strict=False)
                # print(msg)
                return
            pretrained_dict = pretrained_dict['model']
            print("---start load pretrained modle of swin encoder---")

            model_dict = self.swin_unet.state_dict()
            full_dict = copy.deepcopy(pretrained_dict)
            for k, v in pretrained_dict.items():
                if "layers." in k:
                    current_layer_num = 3-int(k[7:8])
                    current_k = "layers_up." + str(current_layer_num) + k[8:]
                    full_dict.update({current_k:v})
            for k in list(full_dict.keys()):
                if k in model_dict:
                    if full_dict[k].shape != model_dict[k].shape:
                        print("delete:{};shape pretrain:{};shape model:{}".format(k,v.shape,model_dict[k].shape))
                        del full_dict[k]

            msg = self.swin_unet.load_state_dict(full_dict, strict=False)
            # print(msg)
        else:
            print("none pretrain")
 