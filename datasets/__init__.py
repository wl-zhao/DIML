import torch
import datasets.cub200
import datasets.cars196
import datasets.stanford_online_products
import datasampler as dsamplers


def select(dataset, opt, data_path):
    if 'cub200' in dataset:
        return cub200.Give(opt, data_path)

    if 'cars196' in dataset:
        return cars196.Give(opt, data_path)

    if 'online_products' in dataset:
        return stanford_online_products.Give(opt, data_path)

    raise NotImplementedError('A dataset for {} is currently not implemented.\n\
                               Currently available are : cub200, cars196 & online_products!'.format(dataset))

def build_dataset(opt, model):
    #################### DATALOADER SETUPS ##################
    dataloaders = {}
    datasets    = select(opt.dataset, opt, opt.source_path)

    dataloaders['evaluation'] = torch.utils.data.DataLoader(datasets['evaluation'], num_workers=opt.kernels, batch_size=opt.bs, shuffle=False)
    dataloaders['testing']    = torch.utils.data.DataLoader(datasets['testing'],    num_workers=opt.kernels, batch_size=opt.bs, shuffle=False)
    if opt.use_tv_split:
        dataloaders['validation'] = torch.utils.data.DataLoader(datasets['validation'], num_workers=opt.kernels, batch_size=opt.bs,shuffle=False)

    train_data_sampler      = dsamplers.select(opt.data_sampler, opt, datasets['training'].image_dict, datasets['training'].image_list)
    if train_data_sampler.requires_storage:
        train_data_sampler.create_storage(dataloaders['evaluation'], model, opt.device)

    dataloaders['training'] = torch.utils.data.DataLoader(datasets['training'], num_workers=opt.kernels, batch_sampler=train_data_sampler)
    return dataloaders, train_data_sampler
