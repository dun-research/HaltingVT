import numpy as np
from functools import partial
from einops import rearrange

import torch
import torch.nn as nn
import torch.nn.functional as F

from mmaction.registry import MODELS
from mmengine.logging import MessageHub

from .vit_utils import (IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, 
                        DropPath, trunc_normal_, load_pretrained,
                        )
from .timesformer_utils import Mlp, Attention, Block, PatchEmbed
from .glimpser_helpers import GlimpserBlock

def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 
        'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'patch_embed.proj', 'classifier': 'head',
        **kwargs
    }

default_cfgs = {
    'vit_base_patch16_224': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_p16_224-80ecf9dd.pth',
        mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    }


class JointHaltBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, qk_scale=None,
                 halt_scale=5, halt_center=-10, attention_type='divided_space_time'):
        super().__init__()
        assert(attention_type in ['joint_space_time'])
        self.attention_type = attention_type
        self.dim = dim
        
        self.halt_scale = halt_scale
        self.halt_center = halt_center
        
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias,
                              attn_drop=attn_drop, proj_drop=drop)
    
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.mlp_hidden_dim = mlp_hidden_dim

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x))) 
        halting_score = x[:, :, 0]  
        halting_score = torch.sigmoid(halting_score * self.halt_scale - self.halt_center)
        return x, halting_score

# sin-cos position encoding
# https://github.com/jadore801120/attention-is-all-you-need-pytorch/blob/master/transformer/Models.py#L31
def get_sinusoid_encoding_table(n_position, d_hid): 
    ''' Sinusoid position encoding table ''' 
    def get_position_angle_vec(position): 
        return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)] 

    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)]) 
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2]) # dim 2i 
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2]) # dim 2i+1 

    return  torch.tensor(sinusoid_table, dtype=torch.float, requires_grad=False).unsqueeze(0) 


class HaltingVTModel(nn.Module):
    """ Vision Transformere
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0.1, hybrid_backbone=None, norm_layer=nn.LayerNorm, num_frames=8, act_layer=None,
                 divid_depth=2, dropout=0., keep_rate=None, fuse_token=False, get_idx=False,
                 use_halting=False, halt_scale=5.0, halt_center=10., eps=0.01, use_distr_prior=0.01,
                 use_motion_token=False, use_learnable_pos_emb=True, get_mask=False,
                 ):
        super().__init__()
        self.divid_depth = divid_depth
        self.depth = depth
        self.dropout = nn.Dropout(dropout)
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        self.num_patches = self.patch_embed.num_patches
        
        act_layer = act_layer or nn.GELU
        self.use_learnable_pos_emb = use_learnable_pos_emb

        ## glimpser Settings
        self.keep_rate = keep_rate
        self.fuse_token = fuse_token
        self.get_idx = get_idx
        
        ## haltingvt settings
        self.use_halting = use_halting
        self.eps = eps
        self.use_distr_prior = use_distr_prior
        self.get_mask = get_mask

        ## motion loss setting
        self.use_motion_token = use_motion_token
        self.motion_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if use_motion_token else None
        
        ## Positional Embeddings. only for divided/joint-ST
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        if self.use_motion_token:
            pos_embed_token_num = self.num_patches+2
        else:
            pos_embed_token_num = self.num_patches+1
        
        if use_learnable_pos_emb:
            self.pos_embed = nn.Parameter(torch.zeros(1, pos_embed_token_num, embed_dim))
        else:
            self.pos_embed = get_sinusoid_encoding_table(pos_embed_token_num, embed_dim)

        self.pos_drop = nn.Dropout(p=drop_rate)
        self.time_embed = nn.Parameter(torch.zeros(1, num_frames, embed_dim))
        self.time_drop = nn.Dropout(p=drop_rate)

        ## Attention Blocks
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, self.depth)]  # stochastic depth decay rule
        self.divid_blocks = [
            GlimpserBlock(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate,
                attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer,
                fuse_token=fuse_token, attention_type="divided_space_time",
                qk_scale=qk_scale
                )
            for i in range(self.divid_depth)]
        
        if self.use_halting:
            self.joint_blocks = [
                JointHaltBlock(
                    dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                    drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, attention_type="joint_space_time",
                    halt_scale=halt_scale, halt_center=halt_center
                    )
                for i in range(self.divid_depth, self.depth)]
        else:
            self.joint_blocks = [
                Block(
                    dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                    drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, attention_type="joint_space_time")
                for i in range(self.divid_depth, self.depth)]
        
        self.blocks = nn.ModuleList(self.divid_blocks + self.joint_blocks)
        
        self.norm = norm_layer(embed_dim)

        # Classifier head
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        if self.use_motion_token:
            self.motion_head = nn.Linear(embed_dim, 2)

        if use_learnable_pos_emb:
            trunc_normal_(self.pos_embed, std=.02)        
        trunc_normal_(self.cls_token, std=.02)
        if self.use_motion_token:
            trunc_normal_(self.motion_token, std=.02)
        self.apply(self._init_weights)

        ## initialization of temporal attention weights, mode as divided_space_time
        i = 0
        for m in self.blocks.modules():
            m_str = str(m)
            if 'Block' in m_str and "temporal_fc" in m_str:
                if i > 0:
                    nn.init.constant_(m.temporal_fc.weight, 0)
                    nn.init.constant_(m.temporal_fc.bias, 0)
                i += 1

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


    @torch.jit.ignore
    def no_weight_decay(self):
        if self.use_motion_token:
            return {'pos_embed', 'cls_token', 'time_embed', 'motion_token'}
        else:
            return {'pos_embed', 'cls_token', 'time_embed'}

    def get_classifier(self):
        if self.use_motion_token:
            return self.head, self.motion_head
        else:
            return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        B = x.shape[0]
        C = x.shape[-1]
        device= x.device
        x, T, W = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(x.size(0), -1, -1)
        if self.use_motion_token:
            motion_tokens = self.motion_token.expand(x.size(0), -1, -1)
            x = torch.cat((cls_tokens, motion_tokens, x), dim=1)
        else:
            x = torch.cat((cls_tokens, x), dim=1)

        if not self.use_learnable_pos_emb:
            self.pos_embed = self.pos_embed.to(x.device)

        ## resizing the positional embeddings in case they don't match the input at inference
        if x.size(1) != self.pos_embed.size(1):
            pos_embed = self.pos_embed
            cls_pos_embed = pos_embed[0,0,:].unsqueeze(0).unsqueeze(1)
            other_pos_embed = pos_embed[0,1:,:].unsqueeze(0).transpose(1, 2)
            P = int(other_pos_embed.size(2) ** 0.5)
            H = x.size(1) // W
            other_pos_embed = other_pos_embed.reshape(1, x.size(2), P, P)
            new_pos_embed = F.interpolate(other_pos_embed, size=(H, W), mode='nearest')
            new_pos_embed = new_pos_embed.flatten(2)
            new_pos_embed = new_pos_embed.transpose(1, 2)
            new_pos_embed = torch.cat((cls_pos_embed, new_pos_embed), 1)
            x = x + new_pos_embed
        else:
            x = x + self.pos_embed
        x = self.pos_drop(x)


        ## Time Embeddings
        cls_tokens = x[:B, 0, :].unsqueeze(1)
        x = x[:,1:]
        x = rearrange(x, '(b t) n m -> (b n) t m',b=B,t=T)
        ## Resizing time embeddings in case they don't match
        if T != self.time_embed.size(1):
            time_embed = self.time_embed.transpose(1, 2)
            new_time_embed = F.interpolate(time_embed, size=(T), mode='nearest')
            new_time_embed = new_time_embed.transpose(1, 2)
            x = x + new_time_embed
        else:
            x = x + self.time_embed
        x = self.time_drop(x)
        x = rearrange(x, '(b n) t m -> b (n t) m',b=B,t=T)
        x = torch.cat((cls_tokens, x), dim=1)    #  b (n t)+1 m

        if self.use_halting:
            output = None
            out = None
            halting_score_layer = []
            not_reached_token_layer = []

        ## Attention blocks      
        idxs = []
        left_tokens = []
        left_token = x.size(1) - 1
        token_nums = []
        mask_record = []
        for i, blk in enumerate(self.blocks):
            if i < self.divid_depth:
                x, left_token, idx = blk(x, self.keep_rate[i], None, self.get_idx,
                                        T=T, W=W)   
                left_tokens.append(left_token)
                if idx is not None:
                    idxs.append(idx)
            elif self.use_halting: 
                if out == None:
                    c_token = torch.zeros(B, left_token+1).to(device)
                    R_token = torch.ones(B, left_token+1).to(device)
                    mask_token = torch.ones(B, left_token+1).to(device)
                    rho_token = torch.zeros(B, left_token+1).to(device)
                    counter_token = torch.ones(B, left_token+1).to(device)
                    out = x
                
                out = out * mask_token.float().view(B, left_token+1, 1) 
                token_nums.append(mask_token.sum(dim=-1, keepdim=True))
                block_output, h_token = blk(out)  # b (n t) m
                
                halting_score_layer.append(torch.mean(h_token, dim=-1))
                
                # for act
                out = block_output.clone()              # Deep copy needed for the next layer
                
                block_output = block_output * mask_token.float().view(B, left_token+1, 1)

                if i==len(self.blocks)-1:
                    h_token = torch.ones(B, left_token+1).to(device)

                # for token part
                c_token = c_token + h_token   # cumulative halting scores
                rho_token = rho_token + mask_token.float() 

                # Case 1: threshold reached in this iteration
                # token part
                reached_token = c_token > 1 - self.eps   
                reached_token = reached_token.float() * mask_token.float() 
                delta1 = block_output * R_token.view(B, left_token+1, 1) * reached_token.view(B, left_token+1, 1)  
                rho_token = rho_token + R_token * reached_token 

                # Case 2: threshold not reached
                # token part
                not_reached_token = c_token < 1 - self.eps
                not_reached_token = not_reached_token.float()
                R_token = R_token - (not_reached_token.float() * h_token)   
                delta2 = block_output * h_token.view(B, left_token+1, 1) * not_reached_token.view(B, left_token+1, 1) 

                counter_token = counter_token + not_reached_token # These data points will need at least one more layer
                not_reached_token_layer.append(not_reached_token.sum(dim=1).mean())

                # Update the mask
                mask_token = c_token < 1 - self.eps
                if self.get_mask:
                    mask_record.append(mask_token)

                if output is None:
                    output = delta1 + delta2
                else:
                    output = output + (delta1 + delta2)
            else: 
                x = blk(x, B, T, W)
            
        if self.use_halting:
            x = output
                
        x = self.norm(x)
        res_kwargs = {'left_tokens': left_tokens,
                      'idxs': idxs,
                      'mask_record': mask_record,
                      }
        if self.use_halting:
            token_nums = torch.cat(token_nums, dim=-1)  
            res_kwargs.update(
                {'rho_token': rho_token,
                 'counter_token': counter_token,
                 'halting_score_layer': halting_score_layer,
                 'not_reached_token_layer': not_reached_token_layer,
                 'token_nums': token_nums,
                }
            )
        if self.use_motion_token:
            res_kwargs.update({'motion_token': x[:, 1]})

        return x[:, 0], res_kwargs

    def forward(self, x): 
        
        x, res_kwargs = self.forward_features(x)
        
        # for print in training log
        if 'not_reached_token_layer' in res_kwargs.keys():
            not_reached_token_layer = res_kwargs['not_reached_token_layer']
            message_hub = MessageHub.get_current_instance()
            rho_token_dict = {}
            for i in range(len(not_reached_token_layer)):
                rho_token_dict[f'train/token_cnts/token_cnt_{i+1}'] = not_reached_token_layer[i]
            message_hub.update_scalars(rho_token_dict)
            
        x = self.head(x)
        if self.use_motion_token:
            motion_cls_scores = self.motion_head(res_kwargs['motion_token'])
            res_kwargs.update({'motion_cls_scores': motion_cls_scores})
            res_kwargs.pop('motion_token')
        return x, res_kwargs

def _conv_filter(state_dict, patch_size=16):
    """ convert patch embedding weight from manual patchify + linear proj to conv"""
    out_dict = {}
    for k, v in state_dict.items():
        if 'patch_embed.proj.weight' in k:
            if v.shape[-1] != patch_size:
                patch_size = v.shape[-1]
            v = v.reshape((v.shape[0], 3, patch_size, patch_size))
        out_dict[k] = v
    return out_dict


@MODELS.register_module()
class HaltingVT(nn.Module):
    
    def __init__(self, img_size=224, patch_size=16, num_classes=400, num_frames=8, 
                 num_heads=12, embed_dim=768,
                 divid_depth=2, 
                 pretrained_model="",  
                 pretrained=True,
                 keep_rate=[0.89, 0.89],   
                 get_idx=False,
                 use_halting=False,  
                 use_motion_token=False,
                 use_learnable_pos_emb=True,
                 get_mask=False,
                 **kwargs):
        super(HaltingVT, self).__init__()
        
        if keep_rate:
            self.keep_rate = keep_rate
        else:
            self.keep_rate = [1.0 for _ in range(divid_depth)]  
               
        self.model = HaltingVTModel(img_size=img_size, num_classes=num_classes, patch_size=patch_size, 
                                         embed_dim=embed_dim, depth=12, num_heads=num_heads, mlp_ratio=4, qkv_bias=True, 
                                         norm_layer=partial(nn.LayerNorm, eps=1e-6), drop_rate=0.,
                                         attn_drop_rate=0., drop_path_rate=0.1, num_frames=num_frames, 
                                         divid_depth=divid_depth, 
                                         keep_rate=self.keep_rate, get_idx=get_idx,
                                         use_halting=use_halting,
                                         use_motion_token=use_motion_token,
                                         use_learnable_pos_emb=use_learnable_pos_emb,
                                         get_mask=get_mask,
                                         **kwargs)
        
        self.model.default_cfg = default_cfgs['vit_base_patch'+str(patch_size)+'_224']
        self.num_patches = self.model.patch_embed.num_patches
        
        if pretrained:
            load_pretrained(self.model, num_classes=self.model.num_classes, 
                            in_chans=kwargs.get('in_chans', 3), filter_fn=_conv_filter, 
                            num_frames=num_frames, num_patches=self.num_patches,
                            attention_type="divided_space_time", 
                            pretrained_model=pretrained_model)
    def forward(self, x):
        x = self.model(x)
        return x
