import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm, trange

from utilities.diml import Sinkhorn
from evaluation.metrics import get_metrics_rank, get_metrics

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

def evaluate(model, dataloader, no_training=True, trunc_nums=None, use_uniform=False, grid_size=4):
    model.eval()
    with torch.no_grad():
        if no_training:
            if 7 % grid_size == 0:
                resize = nn.AdaptiveAvgPool2d(grid_size)
            else:
                resize = nn.Sequential(
                    nn.Upsample(grid_size * 4, mode='bilinear', align_corners=True),
                    nn.AdaptiveAvgPool2d(grid_size),
                )
            feature_bank_center = []

        target_labels = []
        feature_bank = []

        labels = []
        final_iter = tqdm(dataloader, desc='Embedding Data...')
        for idx, inp in enumerate(final_iter):
            input_img, target = inp[1], inp[0]
            target_labels.extend(target.numpy().tolist())
            out = model(input_img.cuda())
            if isinstance(out, tuple): out, aux_f = out

            if no_training:
                enc_out, no_avg_feat = aux_f
                no_avg_feat = no_avg_feat.transpose(1, 3)
                no_avg_feat = model.model.last_linear(no_avg_feat)
                no_avg_feat = no_avg_feat.transpose(1, 3)
                no_avg_feat = resize(no_avg_feat)
                feature_bank.append(no_avg_feat.data)
                feature_bank_center.append(out.data)
            else:
                feature_bank.append(out.data)

            labels.append(target)

        feature_bank = torch.cat(feature_bank, dim=0)
        labels = torch.cat(labels, dim=0)
        N, C, H, W = feature_bank.size()
        feature_bank = feature_bank.view(N, C, -1)

        if no_training:
            feature_bank_center = torch.cat(feature_bank_center, dim=0)
        else:
            feature_bank_center = feature_bank.mean(2)

        feature_bank = torch.nn.functional.normalize(feature_bank, p=2, dim=1)
        feature_bank_center = torch.nn.functional.normalize(feature_bank_center, p=2, dim=1)


        trunc_nums = trunc_nums or [0, 5, 10, 50, 100, 500, 1000]

        overall_r1 = {k: 0.0 for k in trunc_nums}
        overall_rp = {k: 0.0 for k in trunc_nums}
        overall_mapr = {k: 0.0 for k in trunc_nums}

        for idx in trange(len(feature_bank)):
            anchor_center = feature_bank_center[idx]
            approx_sim = calc_similarity(None, anchor_center, None, feature_bank_center, 0)
            approx_sim[idx] = -100

            approx_tops = torch.argsort(approx_sim, descending=True)

            if max(trunc_nums) > 0:
                top_inds = approx_tops[:max(trunc_nums)]

                anchor = feature_bank[idx]
                sim = calc_similarity(anchor, anchor_center, feature_bank[top_inds], feature_bank_center[top_inds], 1, use_uniform)
                rank_in_tops = torch.argsort(sim + approx_sim[top_inds], descending=True)

            for trunc_num in trunc_nums:
                if trunc_num == 0:
                    final_tops = approx_tops
                else:
                    rank_in_tops_real = top_inds[rank_in_tops][:trunc_num]

                    final_tops = torch.cat([rank_in_tops_real, approx_tops[trunc_num:]], dim=0)

            # sim[idx] = -100
                r1, rp, mapr = get_metrics_rank(final_tops.data.cpu(), labels[idx], labels)

                overall_r1[trunc_num] += r1
                overall_rp[trunc_num] += rp
                overall_mapr[trunc_num] += mapr


        for trunc_num in trunc_nums:
            overall_r1[trunc_num] /= float(N / 100)
            overall_rp[trunc_num] /= float(N / 100)
            overall_mapr[trunc_num] /= float(N / 100)

            print("trunc_num: ", trunc_num)
            print('###########')
            print('Now rank-1 acc=%f, RP=%f, MAP@R=%f' % (overall_r1[trunc_num], overall_rp[trunc_num], overall_mapr[trunc_num]))

    data = {
        'r1': [overall_r1[k] for k in trunc_nums],
        'rp': [overall_rp[k] for k in trunc_nums],
        'mapr': [overall_mapr[k] for k in trunc_nums],
    }
    return data