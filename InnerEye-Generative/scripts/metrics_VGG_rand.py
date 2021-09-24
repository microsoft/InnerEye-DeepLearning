#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------

from pathlib import Path
import sys
import os
import torchvision
from prdc import compute_prdc
import numpy as np
import torch
from datetime import datetime
from argparse import ArgumentParser
from scipy.ndimage.filters import gaussian_filter 
from skimage.transform import swirl
import matplotlib.pyplot as plt
from pylab import figure, show, legend
current_file = Path(__file__)
root_dir = current_file.parent.parent.parent
print(current_file, root_dir)
sys.path.append(str(root_dir))
sys.path.append(str(root_dir / 'InnerEye-Generative'))
from health.azure.himl import submit_to_azure_if_needed
from loaders.prostate_loader import Prostate2DSimpleDataset
# Users should import calculate_frechet_distance from:
# https://github.com/mseitzer/pytorch-fid/blob/master/src/pytorch_fid/fid_score.py
# from metrics.FID import calculate_frechet_distance
from locations import DATASETPATH, ENVIRONMNENT, WORKSPACECONFIG, CLUSTER, DATASTORE
DATASETPATH = Path(DATASETPATH)
ENVIRONMNENT = Path(ENVIRONMNENT)
WORKSPACECONFIG = Path(WORKSPACECONFIG)

class SmallOutputVGG(torch.nn.Module):
    def __init__(self, model, dim_out=2048):
        super().__init__()
        model.features[0] = torch.nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)
        model.classifier = torch.nn.Linear(512 * 7 * 7, dim_out)
        self.model = model

    def forward(self, x):
        return self.model.forward(x)

def noise(img):
    if len(img.shape) == 2:
        return np.random.rand(img.shape[0], img.shape[1])
    elif len(img.shape) == 3:
        return np.random.rand(img.shape[0], img.shape[1], img.shape[2])
    else:
        raise ValueError

def swirl_img(img, rotation=0, strength=1, radius=120):
    return np.moveaxis(swirl(np.moveaxis(img, 0, -1), rotation=rotation, strength=strength, radius=radius), -1, 0)

def multiplicative_noise(img, sigma=1):
    return np.clip(img * (1 + np.random.randn(img.shape[0], img.shape[1], img.shape[2]) * sigma), 0, 1)

def plot_metrics(metrics, base_metrics, Title='', xaxis='', figsize=(10,6), savefig=False, title='fig'):
    # create the general figure
    fig1 = figure(figsize=figsize)
    lines = []
    # and the first axes using subplot populated with data 
    ax1 = fig1.add_subplot(111)
    for el in ['precision', 'recall', 'density', 'coverage']:
        m =  np.array([base_metrics[el][0]] + [metrics[key][el][0] for key in metrics])
        std = np.array([base_metrics[el][1]] + [metrics[key][el][1] for key in metrics])
        ax1.fill_between([0] + list(metrics), m-std, m+std, alpha=.5)
    
    # now, the second axes that shares the x-axis with the ax1
    el = 'FID'
    m =  np.array([base_metrics[el][0]] + [metrics[key][el][0] for key in metrics])
    std = np.array([base_metrics[el][1]] + [metrics[key][el][1] for key in metrics])
    ax2 = fig1.add_subplot(111, sharex=ax1, frameon=False)
    ax2.fill_between([0] + list(metrics), m-std, m+std, alpha=.5, color='C4')
    ax2.yaxis.tick_right()
    ax2.yaxis.set_label_position("right")
    
    for el in ['precision', 'recall', 'density', 'coverage']:
        m =  np.array([base_metrics[el][0]] + [metrics[key][el][0] for key in metrics])
        std = np.array([base_metrics[el][1]] + [metrics[key][el][1] for key in metrics])
        lines = lines + ax1.plot([0] + list(metrics), m)

    el = 'FID'
    m =  np.array([base_metrics[el][0]] + [metrics[key][el][0] for key in metrics])
    std = np.array([base_metrics[el][1]] + [metrics[key][el][1] for key in metrics])
    lines = lines + ax2.plot([0] + list(metrics), m, c='C4')

    # for the legend, remember that we used two different axes so, we need 
    # to build the legend manually
    legend(tuple(lines),  ['precision', 'recall', 'density', 'coverage', 'FID (RHS)'])
    plt.xlabel(xaxis)
    plt.title(Title)
    if savefig:
        plt.tight_layout()
        plt.savefig(title)
    show()

def main():
    print('{} -- {}:{}'.format(datetime.now().date(), datetime.now().hour+1, datetime.now().minute), end=' ')
    print('-- Starting up')
    # args
    parser = ArgumentParser()
    parser.add_argument("--local_dataset_path", default= str(DATASETPATH / '2D'), type=str)
    parser.add_argument("--csv_base_name", default='dataset.csv', type=str)   
    parser.add_argument("--gpu", default=None, type=int)
    parser.add_argument("--azureml", '-aml', action='store_true', default=False, help="submit to azureml")
    parser.add_argument("--test", action='store_true', default=False)

    args = parser.parse_args()
    if args.test:
        test()
        return 0
    if args.azureml:  
        run_info = submit_to_azure_if_needed(entry_script=current_file,
                                            snapshot_root_directory=root_dir,
                                            workspace_config_file=WORKSPACECONFIG,
                                            compute_cluster_name=CLUSTER,
                                            default_datastore=DATASTORE,
                                            conda_environment_file=ENVIRONMNENT,
                                            input_datasets=[""],
                                            submit_to_azureml=args.azureml
                                            )
        print(args.local_dataset_path, flush=True)
        args.local_dataset_path = run_info.input_datasets[0] or args.local_dataset_path
        print(args.local_dataset_path, flush=True)
        
    # run on indicated GPU:
    if args.gpu is not None and isinstance(args.gpu, int):
        # Make sure that it only uses a single GPU..
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
        args.gpus = 1

    model = torchvision.models.vgg11(pretrained=False, progress=True)
    model = SmallOutputVGG(model)
    model.eval()
    model.to('cuda')    
    Metrics = {}
    k_iterations = 5
    k_nearest_neigh=5
    path = args.local_dataset_path
    file = args.csv_base_name

    metrics = {'precision':[], 'recall':[],'density':[], 'coverage':[], 'FID':[]}
    for _ in range(k_iterations):
        ds1 = Prostate2DSimpleDataset(path, file, None, input_channels=1)
        ds2 = Prostate2DSimpleDataset(path, file, None, input_channels=1)
        idx = np.arange(len(ds1))
        np.random.shuffle(idx)
        ds1.df = ds1.df.loc[ds1.df.index[idx[:int(len(ds1)/2)]]]
        ds2.df = ds2.df.loc[ds2.df.index[idx[int(len(ds2)/2):]]]
        dls = [torch.utils.data.DataLoader(ds,
                                        batch_size=16,
                                        shuffle=False,
                                        drop_last=False,
                                        num_workers=8) for ds in [ds1, ds2]]
        preds = []
        for dl in dls:
            _preds = []
            for item in dl:
                with torch.no_grad():
                    # x = model(item)
                    x = model(item.to('cuda'))
                _preds.append(x.cpu())
            _preds = torch.cat(_preds, 0)
            preds.append(_preds.numpy())
        _metrics = compute_prdc(preds[0], preds[1], k_nearest_neigh)

        mu = [np.mean(pred, axis=0) for pred in preds]
        sigma = [np.cov(pred, rowvar=False) for pred in preds]
        FID = calculate_frechet_distance(mu[0], sigma[0], mu[1], sigma[1])
        _metrics['FID'] = FID
        for key in _metrics:
            metrics[key].append(_metrics[key])
    Metrics['data_vs_data'] = {}
    for key in metrics:
        print(key, np.mean(metrics[key]), np.std(metrics[key]))
        Metrics['data_vs_data'][key] = [np.mean(metrics[key]), np.std(metrics[key])]

    #
    #

    Metrics['data_vs_noise'] = {}
    metrics = {'precision':[], 'recall':[],'density':[], 'coverage':[], 'FID':[]}
    for _ in range(k_iterations):

        ds1 = Prostate2DSimpleDataset(path, file, None, input_channels=1)
        ds2 = Prostate2DSimpleDataset(path, file,  None, input_channels=1, transforms=noise)
        idx = np.arange(len(ds1))
        np.random.shuffle(idx)
        ds1.df = ds1.df.loc[ds1.df.index[idx[:int(len(ds1)/2)]]]
        ds2.df = ds2.df.loc[ds2.df.index[idx[int(len(ds2)/2):]]]

        dls = [torch.utils.data.DataLoader(ds,
                                        batch_size=16,
                                        shuffle=False,
                                        drop_last=False,
                                        num_workers=8) for ds in [ds1, ds2]]
        preds = []
        for dl in dls:
            _preds = []
            for item in dl:
                with torch.no_grad():
                    #x = model(item)
                    x = model(item.to('cuda'))
                _preds.append(x.cpu())
            _preds = torch.cat(_preds, 0)
            preds.append(_preds.numpy())
        _metrics = compute_prdc(preds[0], preds[1], k_nearest_neigh)

        mu = [np.mean(pred, axis=0) for pred in preds]
        sigma = [np.cov(pred, rowvar=False) for pred in preds]
        FID = calculate_frechet_distance(mu[0], sigma[0], mu[1], sigma[1])
        _metrics['FID'] = FID
        for key in _metrics:
            metrics[key].append(_metrics[key])
    for key in metrics:
        print(key, np.mean(metrics[key]), np.std(metrics[key]))
        Metrics['data_vs_noise'][key] = [np.mean(metrics[key]), np.std(metrics[key])]

    #
    #

    Gaussian_blur_metrics = {}
    for Sigma in [.25,.5,.75,1,2]:
        metrics = {'precision':[], 'recall':[],'density':[], 'coverage':[], 'FID':[]}

        for _ in range(k_iterations):

            ds1 = Prostate2DSimpleDataset(path, file,  None, input_channels=1)
            ds2 = Prostate2DSimpleDataset(path, file,  None, input_channels=1, transforms=gaussian_filter, transforms_args={'sigma':Sigma})
            idx = np.arange(len(ds1))
            np.random.shuffle(idx)
            ds1.df = ds1.df.loc[ds1.df.index[idx[:int(len(ds1)/2)]]]
            ds2.df = ds2.df.loc[ds2.df.index[idx[int(len(ds2)/2):]]]

            dls = [torch.utils.data.DataLoader(ds,
                                            batch_size=16,
                                            shuffle=False,
                                            drop_last=False,
                                            num_workers=8) for ds in [ds1, ds2]]
            preds = []
            for dl in dls:
                _preds = []
                for item in dl:
                    with torch.no_grad():
                        #x = model(item)
                        x = model(item.to('cuda'))
                    _preds.append(x.cpu())
                _preds = torch.cat(_preds, 0)
                preds.append(_preds.numpy())
            _metrics = compute_prdc(preds[0], preds[1], k_nearest_neigh)

            mu = [np.mean(pred, axis=0) for pred in preds]
            sigma = [np.cov(pred, rowvar=False) for pred in preds]
            FID = calculate_frechet_distance(mu[0], sigma[0], mu[1], sigma[1])
            _metrics['FID'] = FID
            for key in _metrics:
                metrics[key].append(_metrics[key])
        Gaussian_blur_metrics[Sigma] = {}
        for key in metrics:
            print(key, np.mean(metrics[key]), np.std(metrics[key]))
            Gaussian_blur_metrics[Sigma][key] = [np.mean(metrics[key]), np.std(metrics[key])]
    Metrics['gaussian_blur'] = Gaussian_blur_metrics

    #
    #

    Mult_noise_metrics = {}
    for Sigma in [.1,.25,.5,.75,1]:
        metrics = {'precision':[], 'recall':[],'density':[], 'coverage':[], 'FID':[]}

        for _ in range(k_iterations):

            ds1 = Prostate2DSimpleDataset(path, file,  None, input_channels=1)
            ds2 = Prostate2DSimpleDataset(path, file,  None, input_channels=1, transforms=multiplicative_noise, transforms_args={'sigma':Sigma})
            idx = np.arange(len(ds1))
            np.random.shuffle(idx)
            ds1.df = ds1.df.loc[ds1.df.index[idx[:int(len(ds1)/2)]]]
            ds2.df = ds2.df.loc[ds2.df.index[idx[int(len(ds2)/2):]]]

            dls = [torch.utils.data.DataLoader(ds,
                                            batch_size=16,
                                            shuffle=False,
                                            drop_last=False,
                                            num_workers=8) for ds in [ds1, ds2]]
            preds = []
            for dl in dls:
                _preds = []
                for item in dl:
                    with torch.no_grad():
                        # = model(item)
                        x = model(item.to('cuda'))
                    _preds.append(x.cpu())
                _preds = torch.cat(_preds, 0)
                preds.append(_preds.numpy())
            _metrics = compute_prdc(preds[0], preds[1], k_nearest_neigh)

            mu = [np.mean(pred, axis=0) for pred in preds]
            sigma = [np.cov(pred, rowvar=False) for pred in preds]
            FID = calculate_frechet_distance(mu[0], sigma[0], mu[1], sigma[1])
            _metrics['FID'] = FID
            for key in _metrics:
                metrics[key].append(_metrics[key])
        Mult_noise_metrics[Sigma] ={}
        for key in metrics:
            print(key, np.mean(metrics[key]), np.std(metrics[key]))
            Mult_noise_metrics[Sigma][key] = [ np.mean(metrics[key]), np.std(metrics[key])]
    Metrics['mult_noise'] = Mult_noise_metrics

    #
    #

    swirl_metrics = {}
    for Sigma in [.0001,.1,.25,.5,.75,1, 2.5, 5,7.5,10]:
        metrics = {'precision':[], 'recall':[],'density':[], 'coverage':[], 'FID':[]}

        for _ in range(k_iterations):

            ds1 = Prostate2DSimpleDataset(path, file,  None, input_channels=1)
            ds2 = Prostate2DSimpleDataset(path, file,  None, input_channels=1, transforms=swirl_img, transforms_args={'strength':Sigma})
            idx = np.arange(len(ds1))
            np.random.shuffle(idx)
            ds1.df = ds1.df.loc[ds1.df.index[idx[:int(len(ds1)/2)]]]
            ds2.df = ds2.df.loc[ds2.df.index[idx[int(len(ds2)/2):]]]

            dls = [torch.utils.data.DataLoader(ds,
                                            batch_size=16,
                                            shuffle=False,
                                            drop_last=False,
                                            num_workers=8) for ds in [ds1, ds2]]
            preds = []
            for dl in dls:
                _preds = []
                for item in dl:
                    with torch.no_grad():
                        #x = model(item)
                        x = model(item.to('cuda'))
                    _preds.append(x.cpu())
                _preds = torch.cat(_preds, 0)
                preds.append(_preds.numpy())
            _metrics = compute_prdc(preds[0], preds[1], k_nearest_neigh)

            mu = [np.mean(pred, axis=0) for pred in preds]
            sigma = [np.cov(pred, rowvar=False) for pred in preds]
            FID = calculate_frechet_distance(mu[0], sigma[0], mu[1], sigma[1])
            _metrics['FID'] = FID
            for key in _metrics:
                metrics[key].append(_metrics[key])
        swirl_metrics[Sigma] ={}
        for key in metrics:
            print(key, np.mean(metrics[key]), np.std(metrics[key]))
            swirl_metrics[Sigma][key] = [ np.mean(metrics[key]), np.std(metrics[key])]
    Metrics['swirl'] = swirl_metrics

    os.makedirs('./outputs', exist_ok=True)
    plot_metrics(Metrics['gaussian_blur'], Metrics['data_vs_data'], """Randomly initialised VGG metrics
        with increasing Gaussian blur""", 'sigma: data = Gaussian_blur(img, mu=0, sigma=sigma)', figsize=(8,5), savefig=True, title='./outputs/gaussian_blur.png')
    plot_metrics(Metrics['mult_noise'], Metrics['data_vs_data'], """Randomly initialised VGG metrics
        with increasing multiplicative noise""", 'w: data = img * (1 + w * N(0,1))', figsize=(8,5), savefig=True, title='./outputs/mult_noise.png')

    _metrics = {}
    for el in  [0.1, 0.25, 0.5, 0.75, 1, 2.5]:
        _metrics[el] = Metrics['swirl'][el]
    plot_metrics(_metrics, Metrics['data_vs_data'], """Randomly initialised VGG metrics
        with increasing swirl""", 's: data = swirl(img, strength=s, radius=120)', figsize=(8,5), savefig=True, title='./outputs/swirl_short.png')

    plot_metrics(Metrics['swirl'], Metrics['data_vs_data'], """Randomly initialised VGG metrics
        with increasing swirl""", 's: data = swirl(img, strength=s, radius=120)', figsize=(8,5), savefig=True, title='./outputs/swirl.png')

def test():
    print('datasetpath: {} -- {}'.format(os.path.isdir(DATASETPATH), DATASETPATH))
    print('environment: {} -- {}'.format(os.path.isfile(ENVIRONMNENT), ENVIRONMNENT))
    print('workspace: {} -- {}'.format(os.path.isfile(WORKSPACECONFIG), WORKSPACECONFIG))


if __name__ == '__main__':
    main()
