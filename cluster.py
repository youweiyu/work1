import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F

d = 400

class multiclusterloss(nn.Module):
    def __init__(self,input_dim, num_clusters,d=d,eta=2,alpha=1e-3,beta=1.0,gamma=1):
        super().__init__()
        self.input_dim = input_dim
        self.num_clusters = num_clusters
        self.d = d         # dimension of each subspace
        self.eta = eta     # parameter for subspace affinity
        self.alpha = alpha # weight for constraint loss
        self.beta = beta   # weight for kl loss
        self.gamma = gamma # weight for contrastive loss

        cluster_dims = self.d * num_clusters
        self.cluster_emb = Cluster(input_dim, cluster_dims)
        self.cluster_layers = Parameter(torch.Tensor(num_clusters, cluster_dims))
        self.D = Parameter(torch.Tensor(cluster_dims, cluster_dims))

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def loss_cal(self, x=None, x_aug=None):

        if x is not None and x_aug is not None:
            T = 0.2 # temperature for contrastive loss
            batch_size, _ = x.size()
            x_abs = x.norm(dim=1)
            x_aug_abs = x_aug.norm(dim=1)

            sim_matrix = torch.einsum('ik,jk->ij', x, x_aug) / torch.einsum('i,j->ij', x_abs, x_aug_abs)
            sim_matrix = torch.exp(sim_matrix / T)
            pos_sim = sim_matrix[range(batch_size), range(batch_size)]
            loss = pos_sim / (sim_matrix.sum(dim=1) - pos_sim)
            loss = - torch.log(loss).mean()
        else:
            loss = 0

        # Constraints
        d_cons1 = D_constraint1(self.device)
        d_cons2 = D_constraint2(self.device)
        loss_d1 = d_cons1(self.D)
        loss_d2 = d_cons2(self.D, self.d, self.num_clusters)
        loss = self.gamma*loss + self.alpha*(loss_d1 + loss_d2)

        return loss

    def forward(self, x):

        z = self.cluster_emb(x)
        s : torch.Tensor = None

        #euclidean distance
        q = 1.0 / (1.0 + torch.sum(
        torch.pow(z.unsqueeze(1) - self.cluster_layers, 2), 2) / self.alpha)
        q = q.pow((self.alpha + 1.0) / 2.0)
        q = (q.t() / torch.sum(q, 1)).t()

        # Calculate subspace affinity
        for i in range(self.num_clusters):
            si = torch.sum(torch.pow(torch.mm(z, self.D[:, i * self.d:(i + 1) * self.d]), 2), 1, keepdim=True)
            if s is None:
                s = si
            else:
                s = torch.cat((s, si), 1)
        s = (s + self.eta * self.d) / ((self.eta + 1) * self.d)
        s = (s.t() / torch.sum(s, 1)).t()

        p_eu = target_distribution(q)
        p_sub = target_distribution(s)

        loss1 = self.loss_cal() # contrastive loss + D constraints
        loss2 = F.kl_div(q.log(), p_eu, reduction='sum') + F.kl_div(s.log(), p_sub, reduction='sum') # kl loss

        loss = loss1 + self.beta * loss2

        return loss.to('cuda' if torch.cuda.is_available() else 'cpu')


def target_distribution(q):
    weight = q ** 2 / q.sum(0)
    return (weight.t() / weight.sum(1)).t()

# to get cluster embeddings
class Cluster(nn.Module):
    def __init__(self, input_dim, cluster_dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(input_dim, int(input_dim/2)),
            nn.LeakyReLU(),
            nn.Linear(int(input_dim/2), cluster_dim),
            nn.LeakyReLU(),
        )
        self.linear_shortcut = nn.Linear(input_dim, cluster_dim)
        # self.ff = FF(input_dim) full connection layer


        self.init_emb()

    def forward(self, x):
        return self.block(x) + self.linear_shortcut(x)


    def init_emb(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)

EPS = 1e-15
def loss_balance_entropy(prob, *kwargs):
    prob = prob.clamp(EPS)
    entropy = prob * prob.log()

    # return negative entropy to maximize it
    if entropy.ndim == 1:
        return entropy.sum()
    elif entropy.ndim == 2:
        return entropy.sum(dim=1).mean()
    else:
        raise ValueError(f'Probability is {entropy.ndim}-d')


class D_constraint1(torch.nn.Module):

    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        super(D_constraint1, self).__init__()
        self.device = device

    def forward(self, d):
        I = torch.eye(d.shape[0]).to(self.device)
        loss_d1_constraint = torch.norm(torch.mm(d,d.t()) * I - I)
        return loss_d1_constraint

   
class D_constraint2(torch.nn.Module):

    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        super(D_constraint2, self).__init__()
        self.device = device

    def forward(self, d, dim, n_clusters):
        S = torch.ones(d.shape[0],d.shape[0]).to(self.device)
        zero = torch.zeros(dim, dim)
        for i in range(n_clusters):
            S[i*dim:(i+1)*dim, i*dim:(i+1)*dim] = zero
        loss_d2_constraint = torch.norm(torch.mm(d,d.t()) * S)
        return loss_d2_constraint

class FF(nn.Module):
    def __init__(self, input_dim):
        super().__init__()

        self.block = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, input_dim),

            nn.ReLU(),
            nn.Linear(input_dim, input_dim),

            nn.ReLU()
        )
        self.linear_shortcut = nn.Linear(input_dim, input_dim)


    def forward(self, x):
        return self.block(x) + self.linear_shortcut(x)
