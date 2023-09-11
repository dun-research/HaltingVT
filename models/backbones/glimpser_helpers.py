import math
from einops import rearrange

import torch
import torch.nn as nn
import torch.nn.functional as F

from .timesformer_utils import DropPath, Mlp


class GlimpserAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        # self.keep_rate = keep_rate
        # assert 0 < keep_rate <= 1, "keep_rate must > 0 and <= 1, got {0}".format(keep_rate)

    def forward(self, x, keep_rate=None, tokens=None):
        # if keep_rate is None:
        #     keep_rate = self.keep_rate
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)  # [3, B, num_heads, N, C//num_heads]
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple) # [B, num_heads, N, C//num_heads]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)   # [B, num_heads, N, C//num_heads]

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        left_tokens = N - 1
        # if self.keep_rate < 1 and keep_rate < 1 or tokens is not None:  # double check the keep rate
        if not keep_rate and not tokens:
            return  x
        elif keep_rate == 1.0 or tokens == N - 1:  # some layer skip tokens selction
            return x, None, None, None, left_tokens, None, None
        else:
            if keep_rate < 1.0:
                left_tokens = math.ceil(keep_rate * (N - 1))
            elif tokens is not None:
                left_tokens = tokens
            assert left_tokens >= 1
            cls_attn = attn[:, :, 0, 1:]  # [B, num_heads, C//num_heads-1] cls_token 与其他所有image_tokens的attentive
            cls_attn = cls_attn.mean(dim=1)  # [B, C//num_heads-1] 取所有head的平均值 TODO: 可以考虑用其他的融合方式，感觉取均值可能反而磨平了某些特征
            _, idx = torch.topk(cls_attn, left_tokens, dim=1, largest=True, sorted=True)  # [B, left_tokens]
            _, idx_compl = torch.topk(cls_attn, (N-1-left_tokens), dim=1, largest=False, sorted=True)  # [B, N-1-left_tokens]
            # cls_idx = torch.zeros(B, 1, dtype=idx.dtype, device=idx.device)
            # index = torch.cat([cls_idx, idx + 1], dim=1)
            index = idx.unsqueeze(-1).expand(-1, -1, C)  # [B, left_tokens, C]
            index_compl = idx_compl.unsqueeze(-1).expand(-1, -1, C)  # [B, N-1-left_tokens, C]

            return x, index, idx, cls_attn, left_tokens, index_compl, idx_compl

        # return  x, None, None, None, left_tokens
        

class GlimpserBlock(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 fuse_token=False,
                 attention_type='divided_space_time', qk_scale=None):
        super().__init__()
        assert(attention_type in ['divided_space_time', 'space_only','joint_space_time'])
        self.attention_type = attention_type
        self.dim = dim
        
        self.norm1 = norm_layer(dim)
        self.attn = GlimpserAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias,
                              attn_drop=attn_drop, proj_drop=drop)
        
        ## Temporal Attention Parameters
        if self.attention_type == 'divided_space_time':
            self.temporal_norm1 = norm_layer(dim)
            self.temporal_attn = GlimpserAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias, 
                                            attn_drop=attn_drop, proj_drop=drop)
            self.temporal_fc = nn.Linear(dim, dim)

        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.mlp_hidden_dim = mlp_hidden_dim
        self.fuse_token = fuse_token
        

    # def forward(self, x, keep_rate=None, tokens=None, get_idx=False):
    def forward(self, x, keep_rate=None, tokens=None, get_idx=False,
                T=8, W=14):
        """
            x: [B, THW, C]
            keep_rate: tuple, 指定每层保留的token比例
            tokens: tuple, 指定每层保留的token数量, 和keep_rate二选一即可
        """

        # if keep_rate is None:
        #     keep_rate = self.keep_rate  # this is for inference, use the default keep rate
        
        B, N, C = x.shape
        num_spatial_tokens = (N - 1) // T  #  x.shape: torch.Size([2, 1569, 768])
        H = num_spatial_tokens // W
        
        if self.attention_type in ['space_only', 'joint_space_time']:
            x = x + self.drop_path(self.attn(self.norm1(x)))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
            return x
        elif self.attention_type == 'divided_space_time':
            ## Temporal
            xt = x[:,1:,:]
            xt = rearrange(xt, 'b (st t) m -> (b st) t m',b=B,st=num_spatial_tokens,t=T)   # [B*196, 8, 768]
            res_temporal = self.drop_path(self.temporal_attn(self.temporal_norm1(xt), ))
            res_temporal = rearrange(res_temporal, '(b st) t m -> b (st t) m',b=B,st=num_spatial_tokens,t=T)  # [B, 196*8, 768]
            res_temporal = self.temporal_fc(res_temporal)
            xt = x[:,1:,:] + res_temporal   # [B, 196*8, 768]
            
            ## Spatial
            init_cls_token = x[:,0,:].unsqueeze(1)   # [B, 1, 768]
            cls_token = init_cls_token.repeat(1, T, 1)   # [B, 8, 768]
            cls_token = rearrange(cls_token, 'b t m -> (b t) m',b=B,t=T).unsqueeze(1)
            xs = xt
            xs = rearrange(xs, 'b (st t) m -> (b t) st m',b=B,st=num_spatial_tokens,t=T)
            xs = torch.cat((cls_token, xs), 1)  # [B*8, 197, 768]
            # res_spatial = self.drop_path(self.attn(self.norm1(xs)))
            # evit attention
            xs = self.norm1(xs)
            tmp, index, idx, cls_attn, left_tokens, index_compl, idx_compl = self.attn(xs, keep_rate, tokens)
            res_spatial = self.drop_path(tmp)  # [B*8, 197, 768]
            
            ### Taking care of CLS token
            cls_token = res_spatial[:,0,:]    # [B*8, 768]
            cls_token = rearrange(cls_token, '(b t) m -> b t m',b=B,t=T)  # [B, 8, 768]
            #TODO: 每一帧做完attention时都有各自的cls—token，可以考虑据此做frame selection
            cls_token = torch.mean(cls_token,1,True) ## averaging for every frame    [B, 1, 768]
            
            res_spatial = res_spatial[:,1:,:]  # [B*8, 196, 768]
            res_spatial = rearrange(res_spatial, '(b t) (st) m -> b (st t) m',b=B,st=num_spatial_tokens, t=T)  # [B, 196*8, 768]
            
            x = xt  # [B, 196*8, 768]
            x = torch.cat((init_cls_token, x), 1) + torch.cat((cls_token, res_spatial), 1)   # [B, 196*8+1, 768]
            
            # evit token selection
            if index is not None:
                # B, N, C = x.shape
                non_cls = x[:,1:,:]  # [B, 196*8, 768]
                # num_spatial_tokens_ori = non_cls.shape[1] // T
                non_cls = rearrange(non_cls, 'b (st t) m -> (b t) st m',b=B,st=num_spatial_tokens, t=T)   # [B*8, 196, 768]
                x_others = torch.gather(non_cls, dim=1, index=index)  # [B*8, left_tokens, C]
                num_spatial_tokens = x_others.shape[1]
                # x_others = rearrange(x_others, '(b t) (st) m -> b (st t) m',b=B,st=num_spatial_tokens, t=T)   # [B, 196*8, 768]

                if self.fuse_token:
                    # compl = complement_idx(idx, num_spatial_tokens_ori - 1)  # [B, N-1-left_tokens]
                    # non_topk = torch.gather(non_cls, dim=1, index=compl.unsqueeze(-1).expand(-1, -1, C))  # [B*8, N-1-left_tokens, C]
                    non_topk = torch.gather(non_cls, dim=1, index=index_compl)  # [B*8, N-1-left_tokens, C]
                    non_topk_attn = torch.gather(cls_attn, dim=1, index=index_compl[:,:,0])  # [B, N-1-left_tokens]
                    extra_token = torch.sum(non_topk * non_topk_attn.unsqueeze(-1), dim=1, keepdim=True)  # [B*8, 1, C]
                    num_spatial_tokens_nontopk = extra_token.shape[1]
                    # extra_token = rearrange(extra_token, '(b t) (st) m -> b (st t) m',b=B,st=num_spatial_tokens_nontopk, t=T)   # [B, num_spatial_tokens_nontopk*8, 768]
                    
                    res = torch.cat([x_others, extra_token], dim=1)  # torch.Size([B*8, num_spatial_tokens_total, 384])
                    num_spatial_tokens_total = num_spatial_tokens + num_spatial_tokens_nontopk
                    res = rearrange(res, '(b t) (st) m -> b (st t) m',b=B,st=num_spatial_tokens_total, t=T)
                    # x = torch.cat([x[:, 0:1], x_others, extra_token], dim=1)
                    x = torch.cat([cls_token, res], dim=1)  # [B, new_tokens_num*8, 768]
                else:
                    x_others = rearrange(x_others, '(b t) (st) m -> b (st t) m',b=B,st=num_spatial_tokens, t=T)
                    x = torch.cat([cls_token, x_others], dim=1)
                    
        # x = torch.cat((init_cls_token, x), 1) + res    # [B, 196*8+1, 768]
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        n_tokens = x.shape[1] - 1   # tokens = T*(196*keep_rate)
        if get_idx and index is not None:
            return x, n_tokens, idx
        return x, n_tokens, None