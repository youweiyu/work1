import torch
import torch.nn as nn
from math import ceil
from torch import nn, einsum
from einops import rearrange, reduce

class AlignFusion(nn.Module):
    def __init__(self, embedding_dim: int, num_heads: int, lowemb, mlp_dim=512):
    
        super().__init__()
    
        self.cross_attn_pathway_to_histology = Attention(embedding_dim, num_heads)
        self.norm1 = nn.LayerNorm(embedding_dim)

        self.mlp = MLPBlock(embedding_dim, mlp_dim)
        self.norm2 = nn.LayerNorm(embedding_dim)

        self.cross_attn_histology_to_pathway = Attention(embedding_dim, num_heads)
        self.norm3 = nn.LayerNorm(embedding_dim)
        
        ## for reconstruction
        self.fusion_linear = nn.Linear(embedding_dim*2, lowemb)

    def forward(self, gene_feat, img_feat):
        """
        1.self-attention for gene
        2.cross-attention Q:gene K,V:wsi
        3.MLP for gene
        4.cross-attention Q:wsi  K,V:gene
        """
        # Align Block
        
        # Self attention block
        keys_p = img_feat
        queries_p = gene_feat
        
        keys=keys_p
        queries=queries_p

        attn_out = self.cross_attn_pathway_to_histology(q=queries, k=keys, v=keys)
        queries = queries + attn_out
        queries = self.norm1(queries)
        
        # MLP block(more one layer)
        mlp_out = self.mlp(queries)
        queries = queries + mlp_out
        queries = self.norm2(queries)

        # Cross attention block, image embedding attending to tokens
        attn_out = self.cross_attn_histology_to_pathway(q=keys, k=queries, v=queries)
        keys = keys + attn_out
        keys = self.norm3(keys)

        k = keys  + keys_p
        q = queries + queries_p

        fusion_feat = torch.cat((q, k), dim=1)

        fusion_feat = self.fusion_linear(fusion_feat)

        return fusion_feat

class MLPBlock(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        mlp_dim: int,
        act: type[nn.Module] = nn.ReLU,
    ) -> None:
        super().__init__()
        self.lin1 = nn.Linear(embedding_dim, mlp_dim)
        self.lin2 = nn.Linear(mlp_dim, embedding_dim)
        self.act = nn.ReLU6()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin2(self.act(self.lin1(x)))

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_proj = nn.Linear(dim, dim, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, q, k, v):

        N, C = q.shape
        assert k.shape == v.shape
        M, C = k.shape
        q = self.q_proj(q).reshape(N, self.num_heads, C // self.num_heads)
        k = self.k_proj(k).reshape(M, self.num_heads, C // self.num_heads)
        v = self.v_proj(v).reshape(M, self.num_heads, C // self.num_heads)

        attn = torch.einsum('nkc,mkc->knm', q, k) * self.scale

        attn = attn.softmax(dim=-1)

        x = torch.einsum('knm,mkc->nkc', attn, v).reshape(N, C)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x
