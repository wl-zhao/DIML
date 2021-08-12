"""==================================================================================================="""
################### LIBRARIES ###################
### Basic Libraries
import comet_ml
import warnings
import logging
warnings.filterwarnings("ignore")

import os, sys, numpy as np, argparse, imp, datetime, pandas as pd, copy
import time, pickle as pkl, random, json, collections
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import torch

from evaluation.metrics import get_metrics
from tqdm import tqdm, trange
import shutil
import torch.nn as nn

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


"""==================================================================================================="""
opt.savename = opt.group + '_s{}'.format(opt.seed)

"""==================================================================================================="""
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
full_training_start_time = time.time()



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
model      = archs.select(opt.arch, opt)

if opt.fc_lr<0:
    to_optim   = [{'params':model.parameters(),'lr':opt.lr,'weight_decay':opt.decay}]
else:
    all_but_fc_params = [x[-1] for x in list(filter(lambda x: 'last_linear' not in x[0], model.named_parameters()))]
    fc_params         = model.model.last_linear.parameters()
    to_optim          = [{'params':all_but_fc_params,'lr':opt.lr,'weight_decay':opt.decay},
                         {'params':fc_params,'lr':opt.fc_lr,'weight_decay':opt.decay}]

_  = model.to(opt.device)


model = nn.DataParallel(model)

"""============================================================================"""

dataloaders, train_data_sampler = datasets.build_dataset(opt, model)
opt.n_classes  = len(dataloaders['training'].dataset.avail_classes)


"""============================================================================"""
#################### CREATE LOGGING FILES ###############
sub_loggers = ['Train', 'Test', 'Model Grad']
if opt.use_tv_split: sub_loggers.append('Val')
LOG = logger.LOGGER(opt, sub_loggers=sub_loggers, start_new=True, log_online=False)

"""============================================================================"""
#################### LOSS SETUP ####################
batchminer   = bmine.select(opt.batch_mining, opt)
criterion, to_optim = criteria.select(opt.loss, opt, to_optim, batchminer)
_ = criterion.to(opt.device)

if 'criterion' in train_data_sampler.name:
    train_data_sampler.internal_criterion = criterion

"""============================================================================"""
#################### OPTIM SETUP ####################
if opt.optim == 'adam':
    optimizer    = torch.optim.Adam(to_optim)
elif opt.optim == 'sgd':
    optimizer    = torch.optim.SGD(to_optim, momentum=0.9)
else:
    raise Exception('Optimizer <{}> not available!'.format(opt.optim))
scheduler    = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=opt.tau, gamma=opt.gamma)


"""============================================================================"""
#################### METRIC COMPUTER ####################
opt.rho_spectrum_embed_dim = opt.embed_dim

"""============================================================================"""
################### Summary #########################3
data_text  = 'Dataset:\t {}'.format(opt.dataset.upper())
setup_text = 'Objective:\t {}'.format(opt.loss.upper())
miner_text = 'Batchminer:\t {}'.format(opt.batch_mining if criterion.REQUIRES_BATCHMINER else 'N/A')
arch_text  = 'Backbone:\t {} (#weights: {})'.format(opt.arch.upper(), misc.gimme_params(model))
summary    = data_text+'\n'+setup_text+'\n'+miner_text+'\n'+arch_text
print(summary)


"""============================================================================"""
################### SCRIPT MAIN ##########################
print('\n-----\n')

iter_count = 0
loss_args  = {'batch':None, 'labels':None, 'batch_features':None, 'f_embed':None}


best_r1 = 0
best_rp = 0
best_mapr = 0

file_handler = logging.FileHandler(filename=os.path.join(opt.save_path, 'log.txt'))
stdout_handler = logging.StreamHandler(sys.stdout)
handlers = [file_handler, stdout_handler]

logging.basicConfig(
    level=logging.INFO, 
    format='[%(asctime)s] %(levelname)s - %(message)s',
    handlers=handlers
)

logger = logging.getLogger('root')


for epoch in range(opt.n_epochs):
    epoch_start_time = time.time()

    if epoch>0 and opt.data_idx_full_prec and train_data_sampler.requires_storage:
        train_data_sampler.full_storage_update(dataloaders['evaluation'], model, opt.device)

    opt.epoch = epoch
    ### Scheduling Changes specifically for cosine scheduling
    if opt.scheduler!='none': print('Running with learning rates {}...'.format(' | '.join('{}'.format(x) for x in scheduler.get_lr())))

    """======================================="""
    if train_data_sampler.requires_storage:
        train_data_sampler.precompute_indices()


    """======================================="""
    ### Train one epoch
    start = time.time()
    _ = model.train()


    loss_collect = []
    data_iterator = tqdm(dataloaders['training'], desc='Epoch {} Training...'.format(epoch))
    logger.info(f"Epoch {epoch} start")

    print(opt.save_path)

    for i,out in enumerate(data_iterator):
        class_labels, input, input_indices = out

        ### Compute Embedding
        input      = input.to(opt.device)
        model_args = {'x':input.to(opt.device)}
        # Needed for MixManifold settings.
        if 'mix' in opt.arch: model_args['labels'] = class_labels
        embeds  = model(**model_args)
        if isinstance(embeds, tuple): embeds, (avg_features, features) = embeds

        ### Compute Loss
        loss_args['batch']          = embeds
        loss_args['labels']         = class_labels
        # loss_args['f_embed']        = model.module.model.last_linear
        loss_args['batch_features'] = features
        loss      = criterion(**loss_args)

        ###
        optimizer.zero_grad()
        loss.backward()

        ### Compute Model Gradients and log them!
        grads              = np.concatenate([p.grad.detach().cpu().numpy().flatten() for p in model.parameters() if p.grad is not None])
        grad_l2, grad_max  = np.mean(np.sqrt(np.mean(np.square(grads)))), np.mean(np.max(np.abs(grads)))
        LOG.progress_saver['Model Grad'].log('Grad L2',  grad_l2,  group='L2')
        LOG.progress_saver['Model Grad'].log('Grad Max', grad_max, group='Max')

        ### Update network weights!
        optimizer.step()

        ###
        loss_collect.append(loss.item())

        ###
        iter_count += 1

        if i==len(dataloaders['training'])-1: data_iterator.set_description('Epoch (Train) {0}: Mean Loss [{1:.4f}]'.format(epoch, np.mean(loss_collect)))

        """======================================="""
        if train_data_sampler.requires_storage and train_data_sampler.update_storage:
            train_data_sampler.replace_storage_entries(embeds.detach().cpu(), input_indices)

    result_metrics = {'loss': np.mean(loss_collect)}

    ####
    LOG.progress_saver['Train'].log('epochs', epoch)
    for metricname, metricval in result_metrics.items():
        LOG.progress_saver['Train'].log(metricname, metricval)
    LOG.progress_saver['Train'].log('time', np.round(time.time()-start, 4))



    """======================================="""
    ### Evaluate Metric for Training & Test (& Validation)
    _ = model.eval()
    dataloader = dataloaders['testing']

    with torch.no_grad():
        target_labels = []
        feature_bank = []
        labels = []
        final_iter = tqdm(dataloader, desc='Embedding Data...')
        image_paths = [x[0] for x in dataloader.dataset.image_list]
        for idx, inp in enumerate(final_iter):
            input_img, target = inp[1], inp[0]
            target_labels.extend(target.numpy().tolist())
            out = model(input_img.to(opt.device))
            if isinstance(out, tuple): out, aux_f = out
            feature_bank.append(out.data)
            labels.append(target)
        feature_bank = torch.cat(feature_bank, dim=0)
        labels = torch.cat(labels, dim=0)

        N, C = feature_bank.size()
        feature_bank = torch.nn.functional.normalize(feature_bank, p=2, dim=1)

        overall_r1 = 0.0
        overall_rp = 0.0
        overall_mapr = 0.0

        for idx in range(len(feature_bank)):
            feature = feature_bank[idx]
            sim = torch.sum(feature_bank * feature.unsqueeze(0), dim=1)
            sim[idx] = -100
            r1, rp, mapr = get_metrics(sim.data.cpu(), labels[idx], labels)
            overall_r1 += r1
            overall_rp += rp
            overall_mapr += mapr

        overall_r1 = overall_r1 / float(N)
        overall_rp = overall_rp / float(N)
        overall_mapr = overall_mapr / float(N)

        is_best = False
        if overall_r1 > best_r1:
            best_r1 = overall_r1
            is_best = True
        if overall_rp > best_rp:
            best_rp = overall_rp
        if overall_mapr > best_mapr:
            best_mapr = overall_mapr

        all_metrics = {
            'r1': overall_r1,
            'rp': overall_rp,
            'mapr': overall_mapr,
        }
        best_metrics = {
            'r1': best_r1,
            'rp': best_rp,
            'mapr': best_mapr,
        }

        for k, v in all_metrics.items():
            LOG.progress_saver['Test'].log(k, v)

        logger.info('saving checkpoint...')
        misc.save_checkpoint(model, optimizer, os.path.join(opt.save_path, 'latest.pth'), all_metrics, best_metrics, epoch)
        if is_best:
            logger.info('saving best checkpoint...')
            shutil.copy2(os.path.join(opt.save_path, 'latest.pth'), os.path.join(opt.save_path, 'best.pth'))


        print('###########')
        logger.info('Now rank-1 acc=%f, RP=%f, MAP@R=%f' % (overall_r1, overall_rp, overall_mapr))
        logger.info('Best rank-1 acc=%f, RP=%f, MAP@R=%f' % (best_r1,  best_rp, best_mapr))

    LOG.update(all=True)


    """======================================="""
    ### Learning Rate Scheduling Step
    if opt.scheduler != 'none':
        scheduler.step()

    print('Total Epoch Runtime: {0:4.2f}s'.format(time.time()-epoch_start_time))
    print('\n-----\n')




"""======================================================="""
### CREATE A SUMMARY TEXT FILE
summary_text = ''
full_training_time = time.time()-full_training_start_time
summary_text += 'Training Time: {} min.\n'.format(np.round(full_training_time/60,2))

summary_text += '---------------\n'
for sub_logger in LOG.sub_loggers:
    metrics       = LOG.graph_writer[sub_logger].ov_title
    summary_text += '{} metrics: {}\n'.format(sub_logger.upper(), metrics)

with open(opt.save_path+'/training_summary.txt','w') as summary_file:
    summary_file.write(summary_text)
