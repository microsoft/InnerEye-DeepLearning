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
        self.net = UNet2D(input_image_channels=1,
                   initial_feature_channels=32,
                   num_classes=len(self.hparams.labels) + 1)
        self.criterion = monai.losses.DiceCELoss(include_background=False, softmax=True)
        self.optimizer_class = torch.optim.AdamW

    def configure_optimizers(self):
        optimizer = self.optimizer_class(self.parameters(), lr=self.hparams.learning_rate, weight_decay=self.hparams.weight_decay)
        return optimizer

    def prepare_batch(self, batch):
        return batch[0].unsqueeze(1),\
               batch[1].unsqueeze(2)

    def infer_batch(self, batch):
        x, y = self.prepare_batch(batch)
        y_hat = self.net(x)
        loss = self.criterion(y_hat, y)
        y_pred = torch.argmax(y_hat, 1, keepdim=True)
        y_pred = torch.cat([y_pred==i for i in range(y_hat.shape[1])], 1)
        metrics = monai.metrics.compute_meandice(y_pred, y, include_background=False)
        non_zero_vols = torch.sum(y[:,1:], (2,3,4)) != 0
        metrics_means = [metrics[:, i][non_zero_vols[:, i]].mean() for i in range(metrics.shape[1])]
        metrics_means.append(metrics[non_zero_vols].mean())
        return y_pred, loss, metrics_means

    def training_step(self, batch, batch_idx):
        y_pred, loss, metrics_means = self.infer_batch(batch)
        self.log('loss/train', loss, prog_bar=True, on_epoch=True)
        metrics_dict = {'labels_DICE/{}_train'.format(key): val for key, val in zip(self.hparams.labels, metrics_means[:-1])}
        metrics_dict['mean_DICE/train'] = metrics_means[-1] 
        self.log_dict(metrics_dict, on_epoch=True)
        return {"loss": loss, "preds": [batch[0], batch[1], y_pred]}

    def validation_step(self, batch, batch_idx):
        y_pred, loss, metrics_means = self.infer_batch(batch)
        self.log('loss/val', loss)
        metrics_dict = {'labels_DICE/{}_test'.format(key): val for key, val in zip(self.hparams.labels, metrics_means[:-1])}
        metrics_dict['mean_DICE/test'] = metrics_means[-1]
        self.log_dict(metrics_dict, on_epoch=True)
        return batch[0], batch[1], y_pred

    def training_epoch_end(self, outputs) -> None:
        grid_img = self.prep_imgs(outputs[-1]['preds'][0], outputs[-1]['preds'][1], outputs[-1]['preds'][2])
        self.logger.log_image('segmented_imgs/train', grid_img, step=self.global_step)

    def validation_epoch_end(self, outputs) -> None:
        grid_img = self.prep_imgs(outputs[-1][0], outputs[-1][1], outputs[-1][2])
        self.logger.log_image('segmented_imgs/test', grid_img, step=self.global_step)

    def prep_imgs(self, img, label, pred):
        if len(self.hparams.labels) != 3:
            raise NotImplementedError(self.hparams.labels)
        # img is shape [batch, 1, H, W] in range (-1, 1)
        img = torch.cat([img] * 3, 1)
        # img is shape [batch, 3, H, W] in range (-1, 1)
        img = (img + 1.) / 2.
        # img is shape [batch, 3, H, W] in range (0, 1)
        colours = torch.tensor([[0.,0.,1.], [0.13,0.4,0.], [1., .1, .1]], device=self.device) 
        # colours = [blue, green, orange], shape [n_colours, 3]
        colours = colours.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        # colours [1, n_cols, 3, 1, 1]
        # label [batch, n_labels, H, W]
        label = torch.stack([label] * 3, 2)
        # label [batch, n_labels, 3, H, W]
        clabel = (label[:, 1:] * colours).sum(1)
        # clabels [batch, 3, H, W]
        img_n_label = img * .5 + clabel
        # pred [batch, n_labels, 1, H, W]
        pred = torch.cat([pred] * 3, 2)
        # pred [batch, n_labels, 3, H, W]
        cpred = (pred[:, 1:] * colours).sum(1)
        # cpred [batch, 3, H, W]
        img_n_pred = img * .5 + cpred
        grid_img = torchvision.utils.make_grid(torch.cat((img[:8], img_n_label[:8], img_n_pred[:8]), 0))
        return grid_img


    def calc_metrics(self, y_hat, y):
        return monai.metrics.compute_meandice(y_hat, y, include_background=False)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--labels', nargs='+', type=str, default=['femurs', 'bladder', 'prostate']) 
        parser.add_argument('--learning_rate', type=float, default=1e-2, help="adam: learning rate")
        parser.add_argument('--weight_decay', type=float, default=1e-2, help="adam: weight decay")
        parser.add_argument("--loss", default='SoftPlus', type=str, choices=('CE', 'SoftPlus', 'Wasserstein'))
        parser.add_argument('--regularisation', '-reg', default='R1', choices=('R1', 'gradient_penalty'), type=str)
        parser.add_argument('--save_model_manually', '-s', action='store_true', default=False)
        parser.add_argument('--mean_last', action='store_true', default=False, help='Take the mean over the three channel images')
        # todo: understand and add pl_batch_shrink=2, pl_decay=0.01, pl_weight=2
        return parser

class UNet_multi_val(Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def validation_step(self, batch, batch_idx, dataloader_idx):
        return super().validation_step(batch, batch_idx)

    def validation_epoch_end(self, _outputs) -> None:
        outputs = _outputs[0]
        grid_img = self.prep_imgs(outputs[-1][0], outputs[-1][1], outputs[-1][2])
        self.logger.log_image('segmented_imgs/test/synth', grid_img, step=self.global_step)
        outputs = _outputs[1]
        grid_img = self.prep_imgs(outputs[-1][0], outputs[-1][1], outputs[-1][2])
        self.logger.log_image('segmented_imgs/test/real', grid_img, step=self.global_step)
