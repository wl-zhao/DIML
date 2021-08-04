import torch, torch.nn as nn
import torch.nn.functional as F



"""================================================================================================="""
ALLOWED_MINING_OPS  = None
REQUIRES_BATCHMINER = False
REQUIRES_OPTIM      = False

class Criterion(torch.nn.Module):
    def __init__(self, opt):
        super(Criterion, self).__init__()
        self.n_classes          = opt.n_classes

        self.pos_weight = opt.loss_multisimilarity_pos_weight
        self.neg_weight = opt.loss_multisimilarity_neg_weight
        self.margin     = opt.loss_multisimilarity_margin
        self.thresh     = opt.loss_multisimilarity_thresh

        self.name           = 'multisimilarity_w'

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
        x_mean = x.mean([2, 3])
        y_mean = y.mean([2, 3])
        x = x.view(B, C, -1)
        y = y.view(B, C, -1)
        x, y, x_mean, y_mean = self.normalize_all(x, y, x_mean, y_mean)

        if self.use_uniform:
            u = torch.zeros(B, H * W, dtype=x.dtype, device=x.device).fill_(1. / (H * W))
            v = torch.zeros(B, H * W, dtype=x.dtype, device=x.device).fill_(1. / (H * W))
        else:
            u, v = self.cross_attention(x, y, x_mean, y_mean)

        sim1 = torch.einsum('bcs, bcm->bsm', x, y).contiguous()
        sim2 = torch.einsum('bc, bc->b', x_mean, y_mean).contiguous().reshape(B, 1, 1)

        wdist = 1.0 - sim1.view(B, H * W, H * W)

        with torch.no_grad():
            K = torch.exp(-wdist / self.eps)
            T = self.Sinkhorn(K, u, v).detach()

        sim = (sim1 + sim2) / 2
        sim = torch.sum(T * sim, dim=(1, 2))

        return sim

    def Sinkhorn(self, K, u, v):
        r = torch.ones_like(u)
        c = torch.ones_like(v)
        thresh = 1e-1
        for i in range(self.max_iter):
            r0 = r
            r = u / torch.matmul(K, c.unsqueeze(-1)).squeeze(-1)
            c = v / torch.matmul(K.permute(0, 2, 1).contiguous(), r.unsqueeze(-1)).squeeze(-1)
            err = (r - r0).abs().mean(dim=1)
            err = err[~torch.isnan(err)]

            if len(err) == 0 or torch.max(err).item() < thresh:
                break

        T = torch.matmul(r.unsqueeze(-1), c.unsqueeze(-2)) * K
        return T

    def forward(self, batch, labels, **kwargs):
        b, _, _, _ = batch.size()
        batch_repeat = torch.repeat_interleave(batch, b, dim=0)
        batch_cat = torch.cat([batch for _ in range(b)], dim=0)
        similarity = self.pair_wise_wdist(batch_repeat, batch_cat).view(b, b)
        if torch.isnan(similarity).any():
            similarity.sum().backward()
            return None

        loss = []
        for i in range(len(batch)):
            pos_idxs       = labels==labels[i]
            pos_idxs[i]    = 0
            neg_idxs       = labels!=labels[i]

            anchor_pos_sim = similarity[i][pos_idxs]
            anchor_neg_sim = similarity[i][neg_idxs]

            # filter nan
            anchor_pos_sim = anchor_pos_sim[~torch.isnan(anchor_pos_sim)] 
            anchor_neg_sim = anchor_neg_sim[~torch.isnan(anchor_neg_sim)]

            ### This part doesn't really work, especially when you dont have a lot of positives in the batch...
            if len(anchor_pos_sim) == 0 or len(anchor_neg_sim) == 0:
                print("all nan")
                continue

            neg_idxs = (anchor_neg_sim + self.margin) > torch.min(anchor_pos_sim)
            pos_idxs = (anchor_pos_sim - self.margin) < torch.max(anchor_neg_sim)
            if not torch.sum(neg_idxs) or not torch.sum(pos_idxs):
                continue
            anchor_neg_sim = anchor_neg_sim[neg_idxs]
            anchor_pos_sim = anchor_pos_sim[pos_idxs]

            pos_term = 1./self.pos_weight * torch.log(1+torch.sum(torch.exp(-self.pos_weight* (anchor_pos_sim - self.thresh))))
            neg_term = 1./self.neg_weight * torch.log(1+torch.sum(torch.exp(self.neg_weight * (anchor_neg_sim - self.thresh))))

            if torch.isnan(pos_term) or torch.isnan(neg_term):
                print("pos, neg:", pos_term, neg_term)
                print(anchor_pos_sim, anchor_neg_sim)

            loss.append(pos_term + neg_term)

        if len(loss) < 1:
            print("no loss")
            loss = None
        else:
            loss = torch.mean(torch.stack(loss))
        return loss
