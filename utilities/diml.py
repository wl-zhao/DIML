import torch
import torch.nn.functional as F

def Sinkhorn(K, u, v):
    r = torch.ones_like(u)
    c = torch.ones_like(v)
    thresh = 1e-1
    for _ in range(100):
        r0 = r
        r = u / torch.matmul(K, c.unsqueeze(-1)).squeeze(-1)
        c = v / torch.matmul(K.permute(0, 2, 1).contiguous(), r.unsqueeze(-1)).squeeze(-1)
        err = (r - r0).abs().mean()
        if err.item() < thresh:
            break
    T = torch.matmul(r.unsqueeze(-1), c.unsqueeze(-2)) * K
    return T

def calc_similarity(anchor, anchor_center, fb, fb_center, stage, use_uniform=False):
    if stage == 0:
        sim = torch.einsum('c,nc->n', anchor_center, fb_center)
    else:
        N, _, R = fb.size()

        sim = torch.einsum('cm,ncs->nsm', anchor, fb).contiguous().view(N, R, R)
        dis = 1.0 - sim
        K = torch.exp(-dis / 0.05)

        if use_uniform:
            u = torch.zeros(N, R, dtype=sim.dtype, device=sim.device).fill_(1. / R)
            v = torch.zeros(N, R, dtype=sim.dtype, device=sim.device).fill_(1. / R)
        else:
            att = F.relu(torch.einsum("c,ncr->nr", anchor_center, fb)).view(N, R)
            u = att / (att.sum(dim=1, keepdims=True) + 1e-5)

            att = F.relu(torch.einsum("cr,nc->nr", anchor, fb_center)).view(N, R)
            v = att / (att.sum(dim=1, keepdims=True) + 1e-5)

        T = Sinkhorn(K, u, v)
        sim = torch.sum(T * sim, dim=(1, 2))
    return sim

