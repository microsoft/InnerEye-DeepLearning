#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------

from pathlib import Path
import sys
import os
import torch
import numpy as np
from datetime import datetime
import pickle
from argparse import ArgumentParser
from tqdm import tqdm
current_file = Path(__file__)
root_dir = current_file.parent.parent.parent
print(root_dir, current_file, flush=True)
sys.path.append(str(root_dir / 'stylegan2-ada-pytorch'))
sys.path.append(str(root_dir))
sys.path.append(str(root_dir / 'InnerEye-Generative'))
from health.azure.himl import submit_to_azure_if_needed
from metrics.get_VGG_model import load_model
from models.StyleGAN2_ADA_frozen import TransferLearningStyleGAN2ADA
from models.StyleGAN2_segment_addon import Model
from locations import DATASETPATH, ENVIRONMNENT, WORKSPACECONFIG, CLUSTER, DATASTORE, TRAINEDSTYLEGAN2WEIGHTS, GENSCANSPATH 
DATASETPATH = Path(DATASETPATH)
ENVIRONMNENT = Path(ENVIRONMNENT)
WORKSPACECONFIG = Path(WORKSPACECONFIG)
GENSCANSPATH = Path(GENSCANSPATH)
TRAINEDSTYLEGAN2WEIGHTS = Path(TRAINEDSTYLEGAN2WEIGHTS)


def upsample_latents(latents, end_res):
    with torch.no_grad():
        output = torch.empty(latents[-1].shape[0], 0, latents[-1].shape[-2], latents[-1].shape[-1], device=latents[-1].device)
        for l in latents:
            us = torch.nn.Upsample(scale_factor=int(end_res / l.shape[-1]))
            l_out = us(l)
            output = torch.cat((output, l_out), 1)
    return output


if __name__ == '__main__':
    print('{} -- {}:{}'.format(datetime.now().date(), datetime.now().hour+1, datetime.now().minute), end=' ')
    print('-- Starting up')
    seed_everything(4321)
    parser = ArgumentParser()
    parser.add_argument("--batch_size", default=16, type=int)
    parser.add_argument("--gpu", default=0, type=int)
    parser.add_argument("--StyleGAN_weights", default=TRAINEDSTYLEGAN2WEIGHTS, type=str)
    parser.add_argument("--segStyleGAN_folder", '-r', default='n_16', type=str)
    parser.add_argument('--classes_weights', nargs='+', type=float, default=[1, 2, 5, 5])
    parser.add_argument('--dim', default=4352, type=int, help='input dim')
    parser.add_argument("--add_scan_to_latent", '-a', action='store_true', default=False)
    parser.add_argument('--labels', nargs='+', type=str, default=['femurs', 'bladder', 'prostate'], 
    help="for now only [femurs, bladder, prostate'] labels are supported") 
    parser.add_argument('--n_samples', default=10000, type=int)
    parser.add_argument("--azureml", '-aml', action='store_true', default=False, help="submit to azureml")
    parser = TransferLearningStyleGAN2ADA.add_model_specific_args(parser)
    args = parser.parse_args()
    args.n_classes = len(args.labels) + 1
    # make dir to save data
    path = './outputs/'
    os.makedirs(path, exist_ok=True)
    if '_scan' in args.segStyleGAN_folder:
        args.add_scan_to_latent = True
    if args.add_scan_to_latent and '_scan' not in args.segStyleGAN_folder:
        args.segStyleGAN_folder += '_scan'
    if args.azureml:  
        run_info = submit_to_azure_if_needed(entry_script=current_file,
                                            snapshot_root_directory=root_dir,
                                            workspace_config_file=WORKSPACECONFIG,
                                            compute_cluster_name=CLUSTER,
                                            default_datastore=DATASTORE,
                                            conda_environment_file=ENVIRONMNENT,
                                            input_datasets=[""],
                                            output_datasets=[""],
                                            submit_to_azureml=args.azureml
                                        )

        args.local_dataset_path = run_info.input_datasets[0] or args.local_dataset_path

        args.StyleGAN_weights = Path(run_info.input_datasets[0] / 'epoch=999-step=168999.ckpt')
        args.segStyleGAN_folder = os.path.join(str(run_info.input_datasets[0]), args.segStyleGAN_folder)
        args.segStyleGAN_weights = Path(args.segStyleGAN_folder) / os.listdir(args.segStyleGAN_folder)[0]

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

        path = run_info.output_datasets[0]
    else:
        raise NotImplementedError('This method has only been implemented to work on AML. To work locally, one needs to make sure the path dependencies are accurate')

    # run on indicated GPU:
    device = 'cpu'
    if args.gpu is not None and isinstance(args.gpu, int):
        # Make sure that it only uses a single GPU..
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
        device = 'cuda'

    # initialise StylGAN2
    # load metrics embedding
    embedder = load_model(dim_in=3)
    model = TransferLearningStyleGAN2ADA(embedder, **vars(args))
    # load weights
    stuff = torch.load(args.StyleGAN_weights)
    model.load_state_dict(stuff['state_dict']) 
    # add hooks
    model.activations = {}
    for res in [4, 8, 16, 32, 64, 128]:
        block = getattr(model.generator.synthesis, f'b{res}')
        for conv in ['conv0', 'conv1']:
            if conv in list(block._modules):
                target_layer = block._modules[conv]
                model.hooks.append(target_layer.register_forward_hook(model.forward_hook_fn))
    model.to(device)
    model.eval()

    # initialise StyleGAN2 segmentation model
    seg_model = Model(**vars(args))
    stuff = torch.load(args.segStyleGAN_weights)
    seg_model.load_state_dict(stuff['state_dict'])
    seg_model.to(device)
    seg_model.eval()

    # run
    n_steps = np.arange(int(np.ceil(args.n_samples / args.batch_size)))
    print(args.n_samples, args.batch_size, n_steps, flush=True)
    for step in tqdm(n_steps):
        with torch.no_grad():
            x = torch.randn(args.batch_size, 1, device=device)
            gen_imgs, _ = model.generator_forward(x, None)
            gen_imgs = torch.mean(gen_imgs, 1, keepdim=True)
            # some tricks for gpu mem saving
            latents = [a.float() for a in model.activations['0']]
            model.activations = {}
            latents = upsample_latents(latents, 128)
            if args.add_scan_to_latent:
                latents = torch.cat((latents, gen_imgs), 1)
            preds = seg_model.infer(latents.reshape(-1, latents.shape[1]))
            preds = preds.reshape(args.batch_size, -1, 128, 128)
            to_save = torch.cat((gen_imgs.detach().cpu(), preds.detach().cpu()), 1)
            for i in range(args.batch_size):
                with open(str(path / '{}.pkl'.format(step * args.batch_size + i)), 'wb') as f:
                    pickle.dump(to_save[i], f)



