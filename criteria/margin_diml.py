import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
import batchminer

"""================================================================================================="""
ALLOWED_MINING_OPS  = list(batchminer.BATCHMINING_METHODS.keys())
REQUIRES_BATCHMINER = True
REQUIRES_OPTIM      = True

### MarginLoss with trainable class separation margin beta. Runs on Mini-batches as well.
class Criterion(torch.nn.Module):
    def __init__(self, opt, batchminer):
        super(Criterion, self).__init__()
        self.n_classes          = opt.n_classes

        self.margin             = opt.loss_margin_margin
        self.nu                 = opt.loss_margin_nu
        self.beta_constant      = opt.loss_margin_beta_constant
        self.beta_val           = opt.loss_margin_beta

        if opt.loss_margin_beta_constant:
            self.beta = opt.loss_margin_beta
        else:
            self.beta = torch.nn.Parameter(torch.ones(opt.n_classes)*opt.loss_margin_beta)

        self.batchminer = batchminer

        self.name  = 'margin'

        self.lr    = opt.loss_margin_beta_lr

        ####
        self.ALLOWED_MINING_OPS  = ALLOWED_MINING_OPS
        self.REQUIRES_BATCHMINER = REQUIRES_BATCHMINER
        self.REQUIRES_OPTIM      = REQUIRES_OPTIM

        self.eps = 0.05
        self.max_iter = 100
        self.use_uniform = opt.use_uniform

    def normalize_all(self, x, y, x_mean, y_mean):
        x = F.normalize(x, dim=1)
        y = F.normalize(y, dim=1)
        x_mean = F.normalize(x_mean, dim=1)
        y_mean = F.normalize(y_mean, dim=1)
        return x, y, x_mean, y_mean

    def cross_attention(self, x, y, x_mean, y_mean):
        N, C = x.shape[:2]
        x = x.view(N, C, -1)
        y = y.view(N, C, -1)

        att = F.relu(torch.einsum("nc,ncr->nr", x_mean, y)).view(N, -1)
        u = att / (att.sum(dim=1, keepdims=True) + 1e-5)
        att = F.relu(torch.einsum("nc,ncr->nr", y_mean, x)).view(N, -1)
        v = att / (att.sum(dim=1, keepdims=True) + 1e-5)
        return u, v

    def pair_wise_wdist(self, x, y):
        B, C, H, W = x.size()
        x = x.view(B, C, -1, 1)
        y = y.view(B, C, 1, -1)
        x_mean = x.mean([2, 3])
        y_mean = y.mean([2, 3])

        x, y, x_mean, y_mean = self.normalize_all(x, y, x_mean, y_mean)
        dist1 = torch.sqrt(torch.sum(torch.pow(x - y, 2), dim=1) + 1e-6).view(B, H*W, H*W)
        dist2 = torch.sqrt(torch.sum(torch.pow(x_mean - y_mean, 2), dim=1) + 1e-6).view(B)

        x = x.view(B, C, -1)
        y = y.view(B, C, -1)

        sim = torch.einsum('bcs, bcm->bsm', x, y).contiguous()
        if self.use_uniform:
            u = torch.zeros(B, H*W, dtype=sim.dtype, device=sim.device).fill_(1. / (H * W))
            v = torch.zeros(B, H*W, dtype=sim.dtype, device=sim.device).fill_(1. / (H * W))
        else:
            u, v = self.cross_attention(x, y, x_mean, y_mean)

        wdist = 1.0 - sim.view(B, H*W, H*W)

        with torch.no_grad():
            K = torch.exp(-wdist / self.eps)
            T = self.Sinkhorn(K, u, v)

        if torch.isnan(T).any():
            return None

        dist1 = torch.sum(T * dist1, dim=(1, 2))
        dist = dist1 + dist2
        dist = dist / 2

        return dist

    def Sinkhorn(self, K, u, v):
        r = torch.ones_like(u)
        c = torch.ones_like(v)
        thresh = 1e-1
        for i in range(self.max_iter):
            r0 = r
            r = u / torch.matmul(K, c.unsqueeze(-1)).squeeze(-1)
            c = v / torch.matmul(K.permute(0, 2, 1).contiguous(), r.unsqueeze(-1)).squeeze(-1)
            err = (r - r0).abs().mean()
            if err.item() < thresh:
                break

        T = torch.matmul(r.unsqueeze(-1), c.unsqueeze(-2)) * K

        return T

    def forward(self, batch, labels, **kwargs):
        # sampled_triplets = self.batchminer(batch, labels)

        pooled_feature = batch.mean([2, 3])
        pooled_feature = F.normalize(pooled_feature, dim=-1)

        sampled_triplets = self.batchminer(pooled_feature, labels)

        if len(sampled_triplets):
            d_ap, d_an = [],[]
            for triplet in sampled_triplets:
                train_triplet = {'Anchor': batch[triplet[0]], 'Positive':batch[triplet[1]], 'Negative':batch[triplet[2]]}

                pos_dist = self.pair_wise_wdist(train_triplet['Anchor'].unsqueeze(0), train_triplet['Positive'].unsqueeze(0))
                neg_dist = self.pair_wise_wdist(train_triplet['Anchor'].unsqueeze(0), train_triplet['Negative'].unsqueeze(0))

                if pos_dist is None or neg_dist is None:
                    continue

                d_ap.append(pos_dist)
                d_an.append(neg_dist)
            d_ap, d_an = torch.stack(d_ap), torch.stack(d_an)

            if self.beta_constant:
                beta = self.beta
            else:
                beta = torch.stack([self.beta[labels[triplet[0]]] for triplet in sampled_triplets]).to(torch.float).to(d_ap.device)

            pos_loss = torch.nn.functional.relu(d_ap-beta+self.margin)
            neg_loss = torch.nn.functional.relu(beta-d_an+self.margin)

            pair_count = torch.sum((pos_loss>0.)+(neg_loss>0.)).to(torch.float).to(d_ap.device)

            if pair_count == 0.:
                loss = torch.sum(pos_loss+neg_loss)
            else:
                loss = torch.sum(pos_loss+neg_loss)/pair_count

            if self.nu: loss = loss + beta_regularisation_loss.to(torch.float).to(d_ap.device)

        else:
            loss = torch.tensor(0.).to(torch.float).to(batch.device)

        return loss
