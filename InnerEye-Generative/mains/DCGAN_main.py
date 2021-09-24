#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------

from pathlib import Path
import sys
import os
from datetime import datetime
from pytorch_lightning import seed_everything, Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from argparse import ArgumentParser
current_file = Path(__file__)
root_dir = current_file.parent.parent.parent
sys.path.append(str(root_dir / 'InnerEye-Generative'))
sys.path.append(str(root_dir.absolute()))
from health.azure.himl import submit_to_azure_if_needed
from loaders.prostate_loader import Prostate2DSimpleDataLoader
from models.DCGAN import DCGAN
from metrics.get_VGG_model import load_model
from helpers.loggers import AzureMLLogger, TensorboardWithImgs, LoggerCollectionWithImgs
from locations import DATASETPATH, ENVIRONMNENT, WORKSPACECONFIG, CLUSTER, DATASTORE
DATASETPATH = Path(DATASETPATH)
ENVIRONMNENT = Path(ENVIRONMNENT)
WORKSPACECONFIG = Path(WORKSPACECONFIG)


def cli_main(args=None):
    print('{} -- {}:{}'.format(datetime.now().date(), datetime.now().hour+1, datetime.now().minute), end=' ')
    print('-- Starting up')
    seed_everything(1234)
    # args
    parser = ArgumentParser()
    parser.add_argument("--batch_size", default=23, type=int)
    parser.add_argument("--local_dataset_path", default=str(DATASETPATH / '2D/'), type=str)
    parser.add_argument("--csv_base_name", default='dataset.csv', type=str)   
    parser.add_argument("--debug", default=False, type=bool)
    parser.add_argument("--test", default=False, action='store_true')
    parser.add_argument("--gpu", default=None, type=int)
    parser.add_argument("--azureml", '-aml', action='store_true', default=False, help="submit to azureml")
    parser = Trainer.add_argparse_args(parser)
    parser = DCGAN.add_model_specific_args(parser)

    args = parser.parse_args()
    if args.test:
        test()
        return 0, 0

    # azure ML compatibility
    checkpoint_callback = [ModelCheckpoint(dirpath='./outputs', every_n_train_steps=1650)]
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

    # run on indicated GPU:
    if args.gpu is not None and isinstance(args.gpu, int):
        # Make sure that it only uses a single GPU..
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
        args.gpus = 1

    # load metrics embedding
    embedder = load_model()

    # initialise model
    model = DCGAN(embedder, **vars(args))

    # initialise loader
    loader_gen = Prostate2DSimpleDataLoader(args.local_dataset_path, args.csv_base_name, args.batch_size, input_channels=1)
    logger = TensorboardWithImgs('./outputs', name='DCGAN')
    if args.azureml:
        AMLlogger = AzureMLLogger()
        logger = LoggerCollectionWithImgs([logger, AMLlogger])

    if args.debug:
        trainer = Trainer.from_argparse_args(args, progress_bar_refresh_rate=20, fast_dev_run=True, logger=logger)
        trainer.fit(model, loader_gen.train_dataloader(), [loader_gen.val_dataloader(), loader_gen.train_dataloader()])

    trainer = Trainer.from_argparse_args(args, progress_bar_refresh_rate=20, fast_dev_run=False, logger=logger, callbacks=checkpoint_callback)
    trainer.fit(model, loader_gen.train_dataloader(), [loader_gen.val_dataloader(), loader_gen.train_dataloader()])
    return model, trainer

def test():
    print('datasetpath: {} -- {}'.format(os.path.isdir(DATASETPATH), DATASETPATH))
    print('environment: {} -- {}'.format(os.path.isfile(ENVIRONMNENT), ENVIRONMNENT))
    print('workspace: {} -- {}'.format(os.path.isfile(WORKSPACECONFIG), WORKSPACECONFIG))

if __name__ == '__main__':
    model, trainer = cli_main()



