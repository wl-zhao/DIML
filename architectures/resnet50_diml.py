"""
The network architectures and weights are adapted and used from the great https://github.com/Cadene/pretrained-models.pytorch.
"""
import torch, torch.nn as nn
import pretrainedmodels as ptm

"""============================================================="""
class Network(torch.nn.Module):
    def __init__(self, opt):
        super(Network, self).__init__()

        self.pars  = opt
        self.r1 = opt.r1
        self.r2 = opt.r2
        self.model = ptm.__dict__['resnet50'](num_classes=1000, pretrained='imagenet' if not opt.not_pretrained else None)

        self.name = opt.arch

        if 'frozen' in opt.arch:
            for module in filter(lambda m: type(m) == nn.BatchNorm2d, self.model.modules()):
                module.eval()
                module.train = lambda _: None

        self.model.last_linear = torch.nn.Conv2d(self.model.last_linear.in_features, opt.embed_dim, 1)

        self.layer_blocks = nn.ModuleList([self.model.layer1, self.model.layer2, self.model.layer3, self.model.layer4])

        self.out_adjust = None
        if opt.dropout is not None:
            self.dropout = nn.Dropout(opt.dropout)
        else:
            self.dropout = nn.Identity()


    def forward(self, x, **kwargs):
        x = self.model.maxpool(self.model.relu(self.model.bn1(self.model.conv1(x))))
        for layerblock in self.layer_blocks:
            x = layerblock(x)
        no_avg_feat = x

        x = torch.nn.functional.upsample(x, size=(self.r1, self.r1), mode='bilinear', align_corners=True)
        x = self.dropout(x)
        x = torch.nn.functional.adaptive_avg_pool2d(x, output_size=(self.r2, self.r2))

        per_point_pred = self.model.last_linear(x)

        x = self.model.avgpool(no_avg_feat)
        enc_out = x.view(x.size(0), -1)

        return per_point_pred, (enc_out, no_avg_feat)
