import torch

def get_metrics(sim, query_label, gallery_label):
    tops = torch.argsort(sim, descending=True)

    top1 = tops[0]
    r1 = 0.0
    if query_label == gallery_label[top1]:
        r1 = 1.0

    num_pos = torch.sum(gallery_label == query_label).item()

    rp = torch.sum(gallery_label[tops[0:num_pos]] == query_label).float() / float(num_pos)

    equality = gallery_label[tops[0:num_pos]] == query_label
    equality = equality.float()
    cumulative_correct = torch.cumsum(equality, dim=0)
    k_idx = torch.arange(num_pos) + 1
    precision_at_ks = (cumulative_correct * equality) / k_idx

    rp = rp.item()
    mapr = torch.mean(precision_at_ks).item()

    return r1, rp, mapr

def get_metrics_rank(tops, query_label, gallery_label):

    # tops = torch.argsort(sim, descending=True)
    top1 = tops[0]
    r1 = 0.0
    if query_label == gallery_label[top1]:
        r1 = 1.0

    num_pos = torch.sum(gallery_label == query_label).item()

    rp = torch.sum(gallery_label[tops[0:num_pos]] == query_label).float() / float(num_pos)

    equality = gallery_label[tops[0:num_pos]] == query_label
    equality = equality.float()
    cumulative_correct = torch.cumsum(equality, dim=0)
    k_idx = torch.arange(num_pos) + 1
    precision_at_ks = (cumulative_correct * equality) / k_idx

    rp = rp.item()
    mapr = torch.mean(precision_at_ks).item()

    return r1, rp, mapr



