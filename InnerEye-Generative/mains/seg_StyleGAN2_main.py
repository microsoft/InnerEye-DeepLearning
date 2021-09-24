#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------

from pathlib import Path
import sys
import os
from argparse import ArgumentParser
from torch.utils.data import DataLoader
from datetime import datetime
from pytorch_lightning import seed_everything, Trainer
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
current_file = Path(__file__)
root_dir = current_file.parent.parent.parent
print(current_file, root_dir, flush=True)
sys.path.append(str(root_dir / 'stylegan2-ada-pytorch'))
sys.path.append(str(root_dir))
sys.path.append(str(root_dir / 'InnerEye-Generative'))
from health.azure.himl import submit_to_azure_if_needed
from loaders.gen_imgs_loader import LatentRepsDS, ValidationLatentRepsDS
from models.StyleGAN2_segment_addon import Model
from helpers.loggers import AzureMLLogger, TensorboardWithImgs, LoggerCollectionWithImgs
from locations import DATASETPATH, ENVIRONMNENT, WORKSPACECONFIG, CLUSTER, DATASTORE, GENSCANSPATH
DATASETPATH = Path(DATASETPATH)
ENVIRONMNENT = Path(ENVIRONMNENT)
WORKSPACECONFIG = Path(WORKSPACECONFIG)
GENSCANSPATH = Path(GENSCANSPATH)


if __name__ == '__main__':
    print('{} -- {}:{}'.format(datetime.now().date(), datetime.now().hour+1, datetime.now().minute), end=' ')
    print('-- Starting up')
    seed_everything(1234)
    # args
    parser = ArgumentParser()
    parser.add_argument("--batch_size", default=4096, type=int)
    parser.add_argument("--dataset_path", default=GENSCANSPATH, type=str)
    parser.add_argument("--csv_base_name", default='dataset.csv', type=str)  
    parser.add_argument("--debug", default=False, type=bool)
    parser.add_argument("--gpu", default=None, type=int)
    parser.add_argument("--azureml", '-aml', action='store_true', default=False, help="submit to azureml")
    parser.add_argument("--n_scans", '-nsc', default=16, type=int, help="Number of scans used for training")
    parser.add_argument("--n_scans_val", '-nscv', default=16, type=int, help="Number of scans used for validation")
    parser.add_argument("--n_samples_per_scan", '-samples', default=20000, type=int, 
        help="""Number of pixels to sample from each scan latent represenation.
        If the number is greater than all the pixels, the loader will load all the pixels. """)
    parser.add_argument("--preload_dataset", '-preload', action='store_false', default=True, 
        help="Whether to load the dataset items on the fly or prelod them. It is recommended to preload, as nifti files are lengthy to load.")
    parser.add_argument("--add_scan_to_latent", action='store_true', default=False,
        help="Whether to add the generator output (ie synthetic scans) to the latent representations to train the model.")
    parser.add_argument("--background_perc", default=1, help='Percentual of background samples to keep')
    parser.add_argument("--augment", action='store_true', default=False)

    parser = Trainer.add_argparse_args(parser)
    parser = Model.add_model_specific_args(parser)

    args = parser.parse_args()

    callbacks = [] 

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
        args.dataset_path = run_info.input_datasets[0] or args.dataset_path

    # run on indicated GPU:
    if args.gpu is not None and isinstance(args.gpu, int):
        # Make sure that it only uses a single GPU..
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
        args.gpus = 1

    # initialise model
    model = Model(**vars(args))
    # initialise loader
    print('{} -- {}:{}'.format(datetime.now().date(), datetime.now().hour+1, datetime.now().minute), end=' ')
    print('-- Loading datasets ...')
    if args.n_scans > 30:
        raise NotImplementedError
    train_ds = LatentRepsDS(os.path.join(args.dataset_path, 'train'), args.n_scans, args.n_samples_per_scan, 
        preload=args.preload_dataset, background_perc=float(args.background_perc), add_scan_to_latent=args.add_scan_to_latent)
    val_train_ds = ValidationLatentRepsDS(os.path.join(args.dataset_path, 'train'), [args.n_scans - 2, args.n_scans - 1], add_scan_to_latent=args.add_scan_to_latent)
    val_ds = ValidationLatentRepsDS(os.path.join(args.dataset_path, 'val'), args.n_scans_val,add_scan_to_latent=args.add_scan_to_latent)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=12)
    val_train_dl = DataLoader(val_train_ds, batch_size=args.batch_size, num_workers=12)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, num_workers=12)
    print('{} -- {}:{}'.format(datetime.now().date(), datetime.now().hour+1, datetime.now().minute), end=' ')
    print('-- ... done')
    logger = TensorboardWithImgs('./outputs/segStyleGAN2')
    
    callbacks.append(ModelCheckpoint(dirpath='./outputs/segStyleGAN2/{}/version_{}/checkpoints'.format(logger.name, logger.version), 
                                     monitor='mean_acc_no_bckgnd/val/dataloader_idx_0', mode='max', save_top_k=3, 
                                     #every_n_train_steps=50 * int(len(train_ds) / args.batch_size),
                                     auto_insert_metric_name=False,
                                     filename='epoch={epoch}-step={global_step}-val_loss={loss/val/dataloader_idx_0:.4f}' +
                                     '-mean_acc={mean_acc_no_bckgnd/val/dataloader_idx_0:.4f}'))
    callbacks.append(ModelCheckpoint(dirpath='./outputs/segStyleGAN2/{}/version_{}/checkpoints'.format(logger.name, logger.version), 
                                     #monitor='mean_acc_no_bckgnd/val/dataloader_idx_0', mode='max', save_top_k=3, 
                                     every_n_train_steps=50 * int(len(train_ds) / args.batch_size),
                                     #save_last =True,
                                     auto_insert_metric_name=False,
                                     filename='epoch={epoch}-step={global_step}-val_loss={loss/val/dataloader_idx_0:.4f}' +
                                     '-mean_acc={mean_acc_no_bckgnd/val/dataloader_idx_0:.4f}'))
    if args.azureml:
        AMLlogger = AzureMLLogger()
        logger = LoggerCollectionWithImgs([logger, AMLlogger])

    if args.debug:
        trainer = Trainer.from_argparse_args(args, progress_bar_refresh_rate=20, fast_dev_run=True, logger=logger)
        trainer.fit(model, train_dl, [val_dl, val_train_dl])

    trainer = Trainer.from_argparse_args(args, progress_bar_refresh_rate=20, 
                                         fast_dev_run=False, logger=logger, 
                                         checkpoint_callback=True, 
                                         callbacks=callbacks if len(callbacks)>0 else None)
    trainer.fit(model, train_dl, [val_dl, val_train_dl])
