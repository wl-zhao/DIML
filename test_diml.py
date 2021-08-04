"""==================================================================================================="""
################### LIBRARIES ###################
### Basic Libraries
import comet_ml
import warnings
warnings.filterwarnings("ignore")

import os, sys, numpy as np, argparse, imp, datetime, pandas as pd, copy
import time, pickle as pkl, random, json, collections
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import torch

from tqdm import tqdm, trange
from utilities.misc import load_checkpoint
import torch.nn.functional as F
import shutil

import parameters    as par


"""==================================================================================================="""
################### INPUT ARGUMENTS ###################
parser = argparse.ArgumentParser()

parser = par.basic_training_parameters(parser)
parser = par.batch_creation_parameters(parser)
parser = par.batchmining_specific_parameters(parser)
parser = par.loss_specific_parameters(parser)

##### Read in parameters
opt = parser.parse_args()


### Load Remaining Libraries that neeed to be loaded after comet_ml
import torch, torch.nn as nn
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
import architectures as archs
import datasampler   as dsamplers
import datasets      as datasets
import criteria      as criteria
import batchminer    as bmine
import evaluation    as eval
from utilities import misc
from utilities import logger


"""==================================================================================================="""
opt.source_path += '/'+opt.dataset
opt.save_path   += '/'+opt.dataset
print(opt.save_path)

#Assert that the construction of the batch makes sense, i.e. the division into class-subclusters.
assert not opt.bs%opt.samples_per_class, 'Batchsize needs to fit number of samples per class for distance sampling and margin/triplet loss!'

opt.pretrained = not opt.not_pretrained




"""==================================================================================================="""
################### GPU SETTINGS ###########################
os.environ["CUDA_DEVICE_ORDER"]   ="PCI_BUS_ID"



"""==================================================================================================="""
#################### SEEDS FOR REPROD. #####################
torch.backends.cudnn.deterministic=True; np.random.seed(opt.seed); random.seed(opt.seed)
torch.manual_seed(opt.seed); torch.cuda.manual_seed(opt.seed); torch.cuda.manual_seed_all(opt.seed)



"""==================================================================================================="""
##################### NETWORK SETUP ##################
opt.device = torch.device('cuda')


"""============================================================================"""
#################### DATALOADER SETUPS ##################
dataloaders = {}
datasets    = datasets.select(opt.dataset, opt, opt.source_path)

dataloaders['testing']    = torch.utils.data.DataLoader(datasets['testing'],    num_workers=opt.kernels, batch_size=opt.bs, shuffle=False)
opt.n_classes  = len(dataloaders['testing'].dataset.avail_classes)
model      = archs.select(opt.arch, opt)
_  = model.to(opt.device)


"""============================================================================"""
################### Summary #########################3
data_text  = 'Dataset:\t {}'.format(opt.dataset.upper())
setup_text = 'Objective:\t {}'.format(opt.loss.upper())
arch_text  = 'Backbone:\t {} (#weights: {})'.format(opt.arch.upper(), misc.gimme_params(model))
summary    = data_text+'\n'+setup_text+'\n'+arch_text
print(summary)

"""============================================================================"""
################### SCRIPT MAIN ##########################
print('\n-----\n')

iter_count = 0
loss_args  = {'batch': None, 'labels':None, 'batch_features':None, 'f_embed':None}


# prepare path
CUB_LOGS = {
    'Angular': ['CUB_Angular_s0'],
    'Arcface': ['CUB_ArcFace_s0'],
    'Contrasitive': ['CUB_Contrastive_s0'],
    'NPair': ['CUB_Npair_s0'],
    'GenLifted': ['CUB_GenLifted_s0'],
    'ProxyNCA': ['CUB_ProxyNCA_s0'],
    'Histogram': ['CUB_Histogram_s0'],
    'Quadruplet': ['CUB_Quadruplet_Distance_s0'],
    'SNR': ['CUB_SNR_Distance_s0'],
    'Softmax': ['CUB_Softmax_s0'],
    'Triplet_Random': ['CUB_Triplet_Random_s0'],
    'Triplet_Semihard': ['CUB_Triplet_Semihard_s0'],
    'Triplet_Softhard': ['CUB_Triplet_Softhard_s0'],
    'Triplet_Distance': ['CUB_Triplet_Distance_s0'],
    'Margin_b12_64': ['CUB_Margin_b12_Distance_64_s0'],
    'Margin_b12': ['CUB_Margin_b12_Distance_s0_3'],
    'Margin_b12_512': ['CUB_Margin_b12_Distance_512_s0'],
    'Margin_b06': ['CUB_Margin_b06_Distance_s0'],
    'Multisimilarity_64': ['CUB_MS_64_s0'],
    'Multisimilarity': ['CUB_MS_s0'],
    'Multisimilarity_512': ['CUB_MS_512_s0'],
}

CARS_LOGS = {
    'Angular': ['CARS_Angular_s0'],
    'Arcface': ['CARS_ArcFace_s0'],
    'Contrasitive': ['CARS_Contrastive_s0'],
    'NPair': ['CARS_Npair_s0'],
    'GenLifted': ['CARS_GenLifted_s0'],
    'ProxyNCA': ['CARS_ProxyNCA_s0'],
    'Histogram': ['CARS_Histogram_s0'],
    'Quadruplet': ['CARS_Quadruplet_Distance_s0'],
    'SNR': ['CARS_SNR_Distance_s0'],
    'Softmax': ['CARS_Softmax_s0'],
    'Triplet_Random': ['CARS_Triplet_Random_s0'],
    'Triplet_Semihard': ['CARS_Triplet_Semihard_s0'],
    'Triplet_Softhard': ['CARS_Triplet_Softhard_s0'],
    'Triplet_Distance': ['CARS_Triplet_Distance_s0'],
    'Margin_b12': ['CARS_Margin_b12_Distance_s0'],
    'Margin_b06': ['CARS_Margin_b06_Distance_s0'],
    'Multisimilarity': ['CARS_MS_s0'],
    'Margin_b12_64': ['CARS_Margin_b12_Distance_64_s0_1'],
    'Multisimilarity_64': ['CARS_MS_64_s2_1'],
    'Margin_b12_512': ['CARS_Margin_b12_Distance_512_s0_1'],
    'Multisimilarity_512': ['CARS_MS_512_s2_1'],
}

SOP_LOGS = {
    'Angular': ['SOP_Angular_s0'],
    'Arcface': ['SOP_ArcFace_s0'],
    'Contrasitive': ['SOP_Contrastive_s0'],
    'NPair': ['SOP_Npair_s0'],
    'GenLifted': ['SOP_GenLifted_s0'],
    'Histogram': ['SOP_Histogram_s0'],
    'Quadruplet': ['SOP_Quadruplet_Distance_s0'],
    'SNR': ['SOP_SNR_Distance_s0'],
    'Softmax': ['SOP_Softmax_s0'],
    'Triplet_Random': ['SOP_Triplet_Random_s0'],
    'Triplet_Semihard': ['SOP_Triplet_Semihard_s0'],
    'Triplet_Softhard': ['SOP_Triplet_Softhard_s0'],
    'Triplet_Distance': ['SOP_Triplet_Distance_s0'],
    'ProxyNCA': ['SOP_ProxyNCA_s0'],
    'Margin_b12': ['SOP_Margin_b12_Distance_s0'],
    'Margin_b06': ['SOP_Margin_b06_Distance_s0'],
    'Multisimilarity': ['SOP_MS_s0'],
}

if opt.dataset == 'cub200':
    LOGS = CUB_LOGS
elif opt.dataset == 'cars196':
    LOGS = CARS_LOGS
else:
    LOGS = SOP_LOGS


from evaluation.eval_diml import evaluate

results = []
methods = []
data = {
    k: []
    for k in ['method', 'r1', 'rp', 'mapr']
}

trunc_nums = [0, 100]

for method, info in LOGS.items():
    path = f'Training_Results/{opt.dataset}/{info[0]}/best.pth' 
    best_metrics = load_checkpoint(model, None, path)
    print(best_metrics)

    result = evaluate(model, dataloaders['testing'], True, trunc_nums, use_uniform=False, grid_size=4)

    print(result)
    result['method'] = [f'{method} + ours ({trunc})' for trunc in trunc_nums]
    for k, v in data.items():
        v.extend(result[k])

df = pd.DataFrame(data)
os.makedirs('test_results', exist_ok=True)
df.to_csv(f'test_results/test_diml_{opt.dataset}.csv')
