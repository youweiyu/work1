import sys
import torch
import numpy as np
import scanpy as sc
import pandas as pd
import torch.nn as nn
from random import shuffle
import torch.optim as optim
import torch.nn.functional as F
from sklearn.cluster import KMeans
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module

from fusion import AlignFusion
from layers import MGCN,decoder
from cluster import multiclusterloss
from dynamic import dynamic_weighting

class stMMR(nn.Module):
    def __init__(self,nfeatX,nfeatI,hidden_dims,k_clusters,num_gene,num_img, dim_sub, heads, contrasive, eta, beta1, beta2, b, gamma, tao):
        super(stMMR, self).__init__()
        self.mgcn = MGCN(nfeatX,nfeatI,hidden_dims)
        
        self.dynamic_weight = dynamic_weighting(hidden_dims[1],hidden_dims[1],num_gene,num_img)
        self.align_fusion = AlignFusion(embedding_dim=hidden_dims[1], num_heads=heads, mlp_dim=hidden_dims[1]//2, lowemb=hidden_dims[2])
        self.cluster_loss = multiclusterloss(input_dim=hidden_dims[2], num_clusters=k_clusters, dim_sub=dim_sub,
                                             contrasive = contrasive, eta=eta, beta1=beta1, beta2=beta2, b=b, gamma=gamma, tao=tao)
        
        self.ZINB = decoder(hidden_dims[2],nfeatX)

    def forward(self,x,i,a):
        emb_x,emb_i = self.mgcn(x,i,a)

        emb_x,emb_i = self.dynamic_weight(emb_x,emb_i)
        fusion_feat = self.align_fusion(emb_x,emb_i)
        z_I = fusion_feat
        [pi, disp, mean]  = self.ZINB(z_I)
        loss_cluster = self.cluster_loss(z_I)

        return z_I, loss_cluster, pi, disp, mean
    
    def get_embedding(self, x, i, a):
        with torch.no_grad():
            emb_x,emb_i = self.mgcn(x,i,a)
            emb_x,emb_i = self.dynamic_weight(emb_x,emb_i)
            fusion_feat = self.align_fusion(emb_x,emb_i)
        
        return fusion_feat
