import torch
import torch.nn as nn

class dynamic_weighting(nn.Module):
    def __init__(self, emb_gene, emb_img, num_gene, num_img):
        super().__init__()
        self.num_gene = num_gene
        self.num_img = num_img

        self.confidence_gene_pre = nn.Linear(emb_gene, 1)
        self.confidence_img_pre = nn.Linear(emb_img, 1)

        self.confiNet_gene = nn.Sequential(
            nn.Linear(num_gene, num_gene*2),
            nn.Linear(num_gene*2, num_gene),
            nn.Linear(num_gene, 1),
            nn.Sigmoid()
        )

        self.confiNet_img = nn.Sequential(
            nn.Linear(num_img, num_img*2),
            nn.Linear(num_img*2, num_img),
            nn.Linear(num_img, 1),
            nn.Sigmoid()
        )

    def forward(self, gene_feat, img_feat):
        """
        gene_feat: (N_gene, D_gene)
        img_feat: (N_img, D_img)
        """

        gene_confidence = self.confidence_gene_pre(gene_feat).squeeze(-1)  # (N_gene,)
        img_confidence = self.confidence_img_pre(img_feat).squeeze(-1)      # (N_img,)

        gene_confidence = self.confiNet_gene(gene_confidence.unsqueeze(0)).squeeze(-1)  # (1,)
        img_confidence = self.confiNet_img(img_confidence.unsqueeze(0)).squeeze(-1)     # (1,)

        gene_holo = torch.log(gene_confidence)/torch.log(gene_confidence*img_confidence+1e-8)
        img_holo = torch.log(img_confidence)/torch.log(gene_confidence*img_confidence+1e-8)

        w_gene = gene_confidence+gene_holo
        w_img = img_confidence+img_holo
        w = torch.softmax(torch.cat((w_gene, w_img), dim=0), dim=0)
        w_gene = w[0]
        w_img = w[1]

        gene_feat = gene_feat * w_gene
        img_feat = img_feat * w_img

        return gene_feat, img_feat
