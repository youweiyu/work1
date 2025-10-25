import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from sklearn.cluster import KMeans
import torch.optim as optim
from random import shuffle
import pandas as pd
import numpy as np
import scanpy as sc
from stMMR.layers import GraphConvolution,SelfAttention,MLP,MGCN,decoder
import sys

from dynamic import dynamic_weighting
from fusion import AlignFusion
from cluster import multiclusterloss

heads = 5

class stMMR(nn.Module):
    def __init__(self,nfeatX,nfeatI,hidden_dims,k_clusters,num_gene,num_img):
        super(stMMR, self).__init__()
        self.mgcn = MGCN(nfeatX,nfeatI,hidden_dims)
        
        self.dynamic_weight = dynamic_weighting(hidden_dims[1],hidden_dims[1],num_gene,num_img)
        self.align_fusion = AlignFusion(embedding_dim=hidden_dims[1], num_heads=heads, mlp_dim=hidden_dims[1]//2, lowemb=hidden_dims[2])
        self.cluster_loss = multiclusterloss(input_dim=hidden_dims[2], num_clusters=k_clusters)
        # self.attlayer1 = SelfAttention(dropout=0.1)
        # self.attlayer2 = SelfAttention(dropout=0.1)
        # self.fc = nn.Linear(hidden_dims[1]*2, hidden_dims[1])
        # self.mlp = MLP(hidden_dims[1], dropout_rate=0.1)
        
        self.ZINB = decoder(hidden_dims[2],nfeatX)

    def forward(self,x,i,a):
        emb_x,emb_i = self.mgcn(x,i,a)
        ## attention for omics specific information of scRNA-seq
        # att_weights_x, att_emb_x = self.attlayer1(emb_x, emb_x, emb_x)

        ## attention for omics specific information of scATAC
        # att_weights_i, att_emb_i = self.attlayer2(emb_i, emb_i, emb_i)

        # q_x, q_i = self.mlp(emb_x, emb_i)

        # cl_loss = crossview_contrastive_Loss(q_x, q_i)

        # capture the consistency information
        # emb_con = torch.cat([q_x, q_i], dim=1)
        # z_xi = self.fc(emb_con)


        # z_I = 20*att_emb_x + 1*att_emb_i + 10*z_xi

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
        
        return fusion_feat, self.cluster_loss.d


