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
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from argparse import ArgumentParser
current_file = Path(__file__)
root_dir = current_file.parent.parent.parent
print(current_file, root_dir, flush=True)
sys.path.append(str(root_dir / 'InnerEye-Generative'))
sys.path.append(str(root_dir / 'stylegan2-ada-pytorch'))
sys.path.append(str(root_dir))
from health.azure.himl import submit_to_azure_if_needed
from metrics.get_VGG_model import load_model
from loaders.prostate_loader import Prostate2DSimpleDataLoader
from models.StyleGAN2_ADA_frozen import TransferLearningStyleGAN2ADA
from helpers.loggers import AzureMLLogger, TensorboardWithImgs, LoggerCollectionWithImgs
from locations import DATASETPATH, ENVIRONMNENT, WORKSPACECONFIG, CLUSTER, DATASTORE
DATASETPATH = Path(DATASETPATH)
ENVIRONMNENT = Path(ENVIRONMNENT)
WORKSPACECONFIG = Path(WORKSPACECONFIG)


if __name__ == '__main__':
    print('{} -- {}:{}'.format(datetime.now().date(), datetime.now().hour+1, datetime.now().minute), end=' ')
    print('-- Starting up')
    seed_everything(1234)
    # args
    parser = ArgumentParser()
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--local_dataset_path", default=str(DATASETPATH / '2D/'), type=str)
    parser.add_argument("--csv_base_name", default='dataset.csv', type=str)   
    parser.add_argument("--debug", default=False, type=bool)
    parser.add_argument("--gpu", default=None, type=int)
    parser.add_argument("--azureml", '-aml', action='store_true', default=False, help="submit to azureml")

    parser = Trainer.add_argparse_args(parser)
    parser = TransferLearningStyleGAN2ADA.add_model_specific_args(parser)

    args = parser.parse_args()
    checkpoint_callback = [ModelCheckpoint(dirpath='./outputs', every_n_train_steps=1650)]
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
        args.local_dataset_path = run_info.input_datasets[0] or args.local_dataset_path
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

    # load metrics embedding
    embedder = load_model(dim_in=3)

    # initialise model
    model = TransferLearningStyleGAN2ADA(embedder, **vars(args))

    # initialise loader
    loader_gen = Prostate2DSimpleDataLoader(args.local_dataset_path, args.csv_base_name, args.batch_size, input_channels=3)

    logger = TensorboardWithImgs('./outputs/StyleGAN2')
    if args.azureml:
        AMLlogger = AzureMLLogger()
        logger = LoggerCollectionWithImgs([logger, AMLlogger])

    if args.debug:
        trainer = Trainer.from_argparse_args(args, progress_bar_refresh_rate=20, fast_dev_run=True, logger=logger)
        trainer.fit(model, loader_gen.train_dataloader(), [loader_gen.val_dataloader(), loader_gen.train_dataloader()])

    trainer = Trainer.from_argparse_args(args, progress_bar_refresh_rate=20, fast_dev_run=False, logger=logger, 
                                         callbacks=checkpoint_callback)
    trainer.fit(model, loader_gen.train_dataloader(), [loader_gen.val_dataloader(), loader_gen.train_dataloader()])
