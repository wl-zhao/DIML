import architectures.resnet50
import architectures.googlenet
import architectures.bninception
import architectures.resnet50_diml

def select(arch, opt):
    if 'resnet50_diml' in arch:
        return resnet50_diml.Network(opt)
    if 'resnet50' in arch:
        return resnet50.Network(opt)
    if 'googlenet' in arch:
        return googlenet.Network(opt)
    if 'bninception' in arch:
        return bninception.Network(opt)
