#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------

from pytorch_lightning import LightningModule
from argparse import ArgumentParser
from InnerEye.ML.models.architectures.unet_2d import UNet2D
import monai
import torch
import torchvision


class Model(LightningModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.net = torch.nn.Sequential(
                torch.nn.Linear(self.hparams.dim + int(self.hparams.add_scan_to_latent), 128),
                torch.nn.ReLU(),
                torch.nn.BatchNorm1d(num_features=128),
                torch.nn.Linear(128, 32),
                torch.nn.ReLU(),
                torch.nn.BatchNorm1d(num_features=32),
                torch.nn.Linear(32, len(self.hparams.labels) + 1),

            )
        if len(self.hparams.classes_weights) != 4:
            print('Hyperparam classes weights misconfigured. All classes will be given equal weight')
            self.criterion = torch.nn.CrossEntropyLoss()
        else:
            weights = torch.FloatTensor(self.hparams.classes_weights)
            self.criterion = torch.nn.CrossEntropyLoss(weight=weights/weights.sum())
        self.optimizer_class = torch.optim.AdamW

    def configure_optimizers(self):
        optimizer = self.optimizer_class(self.parameters(), lr=self.hparams.learning_rate, weight_decay=self.hparams.weight_decay)
        return optimizer

    def infer(self, x):
        y_hat = self.net(x)
        y_pred = torch.argmax(y_hat, 1, keepdim=True)
        return y_pred

    def infer_batch(self, batch):
        x, y = batch
        y_hat = self.net(x)
        loss = self.criterion(y_hat, y.squeeze(-1)) 
        y_pred = torch.argmax(y_hat, 1, keepdim=True)
        # starts logs
        accs = torch.Tensor([(y_pred[y.squeeze(0) == i] == i).float().mean() for i in range(4)])
        metrics_dict = {}
        for key, acc in zip(['background'] + self.hparams.labels, accs):
            if not torch.isnan(acc):
                metrics_dict['labels_acc/{}_'.format(key)] = acc
        metrics_dict['mean_acc_all/'] = accs[~torch.isnan(accs)].mean()
        mean_acc_no_bckgnd = accs[1:][~torch.isnan(accs[1:])]
        if len(mean_acc_no_bckgnd) > 0:
            metrics_dict['mean_acc_no_bckgnd/'] = mean_acc_no_bckgnd.mean()
        return y_pred, loss, metrics_dict

    def training_step(self, batch, batch_idx):
        _, loss, metrics_dict = self.infer_batch(batch)
        self.log('loss/train', loss, prog_bar=True, on_epoch=True)
        metrics_dict = {key + 'train': metrics_dict[key] for key in metrics_dict}
        self.log_dict(metrics_dict, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx):
        latent, label, scan, idx = batch
        y_pred, loss, metrics_dict = self.infer_batch([latent, label])
        self.log('loss/val', loss, prog_bar=True, on_epoch=True)
        metrics_dict = {key + 'val': metrics_dict[key] for key in metrics_dict}
        self.log_dict(metrics_dict, on_epoch=True)
        return torch.stack([scan, label, y_pred], 0), idx

    def validation_epoch_end(self, outputs) -> None:
        # vis 2 images for each set
        idx = [torch.stack([o[1] for o in out]).long() for out in outputs]
        outputs = [torch.cat([o[0] for o in out], 1) for out in outputs]
        if int(outputs[0].shape[1]) % (128 * 128) != 0:
            print('only {} batches run. skipping img generation'.
                format(outputs[0].shape[1] / self.hparams.batch_size)) 
        else: 
            outputs = [o[:, _idx] for o, _idx in zip(outputs, idx)] 
            out = outputs[0].reshape(3, self.hparams.n_scans_val, 128, 128)
            grid_img = self.prep_imgs(out[0][:8], out[1][:8], out[2][:8])
            self.logger.log_image('segmented_imgs/val', grid_img, step=self.global_step)
            grid_img = self.prep_imgs(out[0][8:], out[1][8:], out[2][8:])
            self.logger.log_image('segmented_imgs/val2', grid_img, step=self.global_step)
            out = outputs[1].reshape(3, 2, 128, 128)
            grid_img = self.prep_imgs(out[0], out[1], out[2])
            self.logger.log_image('segmented_imgs/train', grid_img, step=self.global_step)


    def prep_imgs(self, img, label, pred):
        if len(self.hparams.labels) != 3:
            raise NotImplementedError(self.hparams.labels)
        # img is shape [batch, H, W] in range (-1, 1)
        img = torch.stack([img] * 3, 1)
        # img is shape [batch, 3, H, W] in range (-1, 1)
        img = (img + 1.) / 2.
        # img is shape [batch, 3, H, W] in range (0, 1)
        colours = torch.tensor([[0., 0., 1.], 
                                [0.13, 0.4, 0.], 
                                [1., .1, .1]], device=self.device) 
        # colours = [blue, green, orange], shape [n_colours, 3]
        colours = colours.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        # colours [1, n_cols, 3, 1, 1]
        # label [batch, H, W]
        label = torch.stack([label==i for i in range(4)], 1)
        # label [batch, n_labels, H, W]
        label = torch.stack([label] * 3, 2)
        # label [batch, n_labels, 3, H, W]
        clabel = (label[:, 1:] * colours).sum(1)
        # clabels [batch, 3, H, W]
        img_n_label = img * .5 + clabel
        # pred [batch, H, W]
        pred = torch.stack([pred==i for i in range(4)], 1)
        # pred [batch, n_labels, H, W]
        pred = torch.stack([pred] * 3, 2)
        # pred [batch, n_labels, 3, H, W]
        cpred = (pred[:, 1:] * colours).sum(1)
        # cpred [batch, 3, H, W]
        img_n_pred = img * .5 + cpred
        grid_img = torchvision.utils.make_grid(torch.cat((img, img_n_label, img_n_pred), 0))
        return grid_img


    def calc_metrics(self, y_hat, y):
        return monai.metrics.compute_meandice(y_hat, y, include_background=False)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--labels', nargs='+', type=str, default=['femurs', 'bladder', 'prostate'], help="""Which labels to train for. 
        Note that only the default is implemented in the dataloader & has been segmented already""") 
        parser.add_argument('--classes_weights', nargs='+', type=float, default=[.15, 13.41, 61.6, 24.85], help="""Classes weights for the cross entropy. 
        Note that these were crucial to improve results. Four are expected - background, femurs, bladder and prostate. 
        The defualts resulted in the best performance.""")
        parser.add_argument('--learning_rate', type=float, default=1e-4, help="adam: learning rate")
        parser.add_argument('--weight_decay', type=float, default=1, help="adam: weight decay")
        parser.add_argument('--dim', default=4352, type=int, help="""Number of channels. 
        Although this is a hyperparam, the number of channels is defined by the extracted GAN latent representations and hence should not be change""")

        return parser
