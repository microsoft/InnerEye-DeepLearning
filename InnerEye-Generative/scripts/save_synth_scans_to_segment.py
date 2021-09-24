#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------

from pathlib import Path
import sys
import os
import requests
from datetime import datetime
from pytorch_lightning import seed_everything, Trainer
from argparse import ArgumentParser
import torch
import numpy as np
import nibabel as nib
import pickle
current_file = Path(__file__)
root_dir = current_file.parent.parent.parent
sys.path.append(str(root_dir / 'stylegan2-ada-pytorch'))
sys.path.append(str(root_dir))
sys.path.append(str(root_dir / 'InnerEye-Generative'))
from health.azure.himl import submit_to_azure_if_needed
from metrics.get_VGG_model import load_model
from loaders.prostate_loader import Prostate2DSimpleDataLoader
from models.StyleGAN2_ADA_frozen import TransferLearningStyleGAN2ADA
from helpers.loggers import AzureMLLogger, TensorboardWithImgs, LoggerCollectionWithImgs
from locations import DATASETPATH, ENVIRONMNENT, WORKSPACECONFIG, CLUSTER, DATASTORE, TRAINEDSTYLEGAN2WEIGHTS, GENSCANSPATH 
DATASETPATH = Path(DATASETPATH)
ENVIRONMNENT = Path(ENVIRONMNENT)
WORKSPACECONFIG = Path(WORKSPACECONFIG)
GENSCANSPATH = Path(GENSCANSPATH)
TRAINEDSTYLEGAN2WEIGHTS = Path(TRAINEDSTYLEGAN2WEIGHTS)


def test():
    print('datasetpath: {} -- {}'.format(os.path.isdir(DATASETPATH), DATASETPATH))
    print('environment: {} -- {}'.format(os.path.isfile(ENVIRONMNENT), ENVIRONMNENT))
    print('workspace: {} -- {}'.format(os.path.isfile(WORKSPACECONFIG), WORKSPACECONFIG))
    print('gen scans: {} -- {}'.format(os.path.isdir(GENSCANSPATH), GENSCANSPATH))
    print('StyleGAN2 weights: {} -- {}'.format(os.path.isfile(TRAINEDSTYLEGAN2WEIGHTS), TRAINEDSTYLEGAN2WEIGHTS))    

def upsample_latents(latents, end_res):
    with torch.no_grad():
        output = torch.empty(latents[-1].shape[0], 0, latents[-1].shape[-2], latents[-1].shape[-1], device=latents[-1].device)
        for l in latents:
            us = torch.nn.Upsample(scale_factor=int(end_res / l.shape[-1]))
            l_out = us(l)
            output = torch.cat((output, l_out), 1)
    return output

def save_scans(model, n_steps, dir, f):
    os.makedirs(dir / 'scans', exist_ok=True)
    for step in range(n_steps):
        model.activations = {}
        with torch.no_grad():
            gen_imgs, _ = model.generator_forward(torch.randn(args.batch_size, 1, 1), None)
            gen_imgs = gen_imgs[:, 0].detach().cpu().numpy()
            latents = model.activations['0']
            latents = upsample_latents(latents, 128)
            latents = torch.FloatTensor(latents).detach().cpu().numpy()
            for j in range(args.batch_size):
                gen_img = gen_imgs[j]
                latent = latents[j]
                # convert img to be read correctly by InnerEye's Radiomics app
                gen_img = gen_img[:, :, np.newaxis]
                gen_img = np.moveaxis(gen_img, 0, 1)
                gen_img = np.flip(gen_img, 1)
                affine = np.eye(4)
                nifti_file = nib.Nifti1Image(gen_img, affine)
                nifti_file.set_header = f.header
                nifti_file.header.set_xyzt_units('mm') 
                nib.save(nifti_file, str(dir / 'scans' / '{}.nii.gz'.format(step * args.batch_size + j)))
                #
                with open(dir / '{}.pkl'.format(step * args.batch_size + j), 'wb') as _f:
                    pickle.dump(latent, _f)

def main(args):
    seed_everything(args.seed)
    if args.azureml:   
        raise NotImplementedError('Not implemented - requires some changing of input and output dataset, and paths logic downstream')
        run_info = submit_to_azure_if_needed(entry_script=current_file,
                                            snapshot_root_directory=root_dir,
                                            workspace_config_file=WORKSPACECONFIG,
                                            compute_cluster_name=CLUSTER,
                                            default_datastore=DATASTORE,
                                            conda_environment_file=ENVIRONMNENT,
                                            input_datasets=[""],
                                            submit_to_azureml=args.azureml
                                        )
    weights_file = root_dir / 'InnerEye-Generative/assets/weights.pkl'
    if not os.path.isfile(weights_file):
        print('{} -- {}:{}'.format(datetime.now().date(), datetime.now().hour+1, datetime.now().minute), end=' ')
        print('-- Downloading StyleGAN2 weights ...')
        url = 'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/transfer-learning-source-nets/ffhq-res256-mirror-paper256-noaug.pkl'
        myfile = requests.get(url)
        os.makedirs(root_dir / 'InnerEye-Generative/assets', exist_ok=True)
        open(weights_file, 'wb').write(myfile.content)
        args.model_weights_path = weights_file
        print('{} -- {}:{}'.format(datetime.now().date(), datetime.now().hour+1, datetime.now().minute), end=' ')
        print('-- ... done')

    # run on indicated GPU:
    if args.gpu is not None and isinstance(args.gpu, int):
        # Make sure that it only uses a single GPU..
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
        args.gpus = 1
    elif args.gpu is None:
        raise NotImplementedError('The code to extract the latent representations from the GAN only works with a single GPU at the moment')

    # load metrics embedding
    embedder = load_model(dim_in=3)

    # initialise model
    model = TransferLearningStyleGAN2ADA(embedder, **vars(args))

    # add hooks 
    model.activations = {}
    for res in [4, 8, 16, 32, 64, 128]:
        block = getattr(model.generator.synthesis, f'b{res}')
        for conv in ['conv0', 'conv1']:
            if conv in list(block._modules):
                print(res, conv)
                target_layer = block._modules[conv]
                model.hooks.append(target_layer.register_forward_hook(model.forward_hook_fn))

    # load weights
    if not os.path.isfile(TRAINEDSTYLEGAN2WEIGHTS):
        raise NameError('Expecting {} path to exist and contain weights for StyleGAN'. format(TRAINEDSTYLEGAN2WEIGHTS))
    weights = torch.load(TRAINEDSTYLEGAN2WEIGHTS)['state_dict']
    model.load_state_dict(weights)

    a = 2. / (155. - (-100.))
    b = 1 - a * 155
    # load an example scan to take header info
    f = nib.load(DATASETPATH / '0/ct.nii.gz')
    model.to('cuda')
    model.eval()
    # train
    os.makedirs(GENSCANSPATH / 'train', exist_ok=True)
    n_steps = np.ceil(args.n_scans_train / args.batch_size)
    save_scans(model, n_steps, dir=GENSCANSPATH / 'train', f=f)
    # val
    os.makedirs(GENSCANSPATH / 'val', exist_ok=True)
    n_steps = np.ceil(args.n_scans_val / args.batch_size)
    save_scans(model, n_steps, dir=GENSCANSPATH / 'val', f=f)


if __name__ == '__main__':
    print('{} -- {}:{}'.format(datetime.now().date(), datetime.now().hour+1, datetime.now().minute), end=' ')
    print('-- Starting up')
    seed_everything(1234)
    # args
    parser = ArgumentParser()
    parser.add_argument("--batch_size", default=8, type=int)
    parser.add_argument("--test", action='store_true', default=False)
    parser.add_argument("--debug", default=False, type=bool)
    parser.add_argument("--gpu", default=None, type=int)
    parser.add_argument("--azureml", '-aml', action='store_true', default=False, help="submit to azureml")
    parser.add_argument("--n_scans_train", default=32, type=int)
    parser.add_argument("--n_scans_val", default=16, type=int)
    parser.add_argument("--seed", default=4321, type=int)
    parser = Trainer.add_argparse_args(parser)
    parser = TransferLearningStyleGAN2ADA.add_model_specific_args(parser)

    args = parser.parse_args()
    if args.test:
        test()
    else:
        main(args)
