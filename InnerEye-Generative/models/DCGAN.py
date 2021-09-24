#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------

# Based on https://github.com/pytorch/examples/blob/master/dcgan/main.py
from argparse import ArgumentParser
from typing import Any
import torch
import torchvision
from pytorch_lightning import LightningModule
from torch import nn, Tensor
import numpy as np
# Users should import calculate_frechet_distance from:
# https://github.com/mseitzer/pytorch-fid/blob/master/src/pytorch_fid/fid_score.py
# from metrics.FID import calculate_frechet_distance
from models.gan_loss import  WassersteinLoss,  GANBCELoss, LegacyLoss, GANBCELossDiscSmoothed
from prdc import compute_prdc


class DCGANGenerator(nn.Module):

    def __init__(self, latent_dim: int, feature_maps: int, image_channels: int) -> None:
        """
        Args:
            latent_dim: Dimension of the latent space
            feature_maps: Number of feature maps to use
            image_channels: Number of channels of the images from the dataset
        """
        super().__init__()
        self.gen = nn.Sequential(
            self._make_gen_block(latent_dim, feature_maps * 16, kernel_size=4, stride=1, padding=0),
            self._make_gen_block(feature_maps * 16, feature_maps * 8),
            self._make_gen_block(feature_maps * 8, feature_maps * 4),
            self._make_gen_block(feature_maps * 4, feature_maps * 2),
            self._make_gen_block(feature_maps * 2, feature_maps),
            self._make_gen_block(feature_maps, image_channels, last_block=True),
        )

    @staticmethod
    def _make_gen_block(
        in_channels: int,
        out_channels: int,
        kernel_size: int = 4,
        stride: int = 2,
        padding: int = 1,
        bias: bool = False,
        last_block: bool = False,
    ) -> nn.Sequential:
        if not last_block:
            gen_block = nn.Sequential(
                nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(True),
            )
        else:
            gen_block = nn.Sequential(
                nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias),
                nn.Tanh(),
            )

        return gen_block

    def forward(self, noise: Tensor) -> Tensor:
        return self.gen(noise)


class DCGANDiscriminator(nn.Module):

    def __init__(self, feature_maps: int, image_channels: int) -> None:
        """
        Args:
            feature_maps: Number of feature maps to use
            image_channels: Number of channels of the images from the dataset
        """
        super().__init__()
        self.disc = nn.Sequential(
            self._make_disc_block(image_channels, feature_maps, batch_norm=False),
            self._make_disc_block(feature_maps, feature_maps * 2),
            self._make_disc_block(feature_maps * 2, feature_maps * 4),
            self._make_disc_block(feature_maps * 4, feature_maps * 8),
            self._make_disc_block(feature_maps * 8, feature_maps * 16),
            self._make_disc_block(feature_maps * 16, 1, kernel_size=4, stride=1, padding=0, last_block=True),
        )

    @staticmethod
    def _make_disc_block(
        in_channels: int,
        out_channels: int,
        kernel_size: int = 4,
        stride: int = 2,
        padding: int = 1,
        bias: bool = False,
        batch_norm: bool = True,
        last_block: bool = False,
    ) -> nn.Sequential:
        if not last_block:
            disc_block = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias),
                nn.BatchNorm2d(out_channels) if batch_norm else nn.Identity(),
                nn.LeakyReLU(0.2, inplace=True),
            )
        else:
            disc_block = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias),
                nn.Sigmoid(),
            )

        return disc_block

    def forward(self, x: Tensor) -> Tensor:
        return self.disc(x).view(-1, 1).squeeze(1)


class DCGAN(LightningModule):
    """
    DCGAN implementation.
    Example::
        from pl_bolts.models.gans import DCGAN
        m = DCGAN()
        Trainer(gpus=2).fit(m)
    Example CLI::
        # mnist
        python dcgan_module.py --gpus 1
        # cifar10
        python dcgan_module.py --gpus 1 --dataset cifar10 --image_channels 3
    """

    def __init__(
        self,
        inception: nn.Module,
        beta1: float = 0.5,
        feature_maps_gen: int = 64,
        feature_maps_disc: int = 64,
        image_channels: int = 1,
        latent_dim: int = 100,
        learning_rate: float = 0.0002,
        optimiser_step_ratio: int = 5,
        **kwargs: Any,
    ) -> None:
        """
        Args:
            beta1: Beta1 value for Adam optimizer
            feature_maps_gen: Number of feature maps to use for the generator
            feature_maps_disc: Number of feature maps to use for the discriminator
            image_channels: Number of channels of the images from the dataset
            latent_dim: Dimension of the latent space
            learning_rate: Learning rate
        """
        super().__init__()
        self.save_hyperparameters(ignore='inception')

        # structure
        self.generator = self._get_generator()
        self.discriminator = self._get_discriminator()
    
        # optimisation
        self._get_criterion()

        # metrics
        self.embedder = inception
        self.embedder.eval()

    def _get_criterion(self):
        if self.hparams.loss == 'CE':
            self.criterion = GANBCELoss
        elif self.hparams.loss == 'Wasserstein':
            self.criterion = WassersteinLoss
        elif self.hparams.loss == 'Legacy':
            self.criterion = LegacyLoss
        elif self.hparams.loss == 'CESmoothed':
            self.criterion = GANBCELossDiscSmoothed
        else:
            raise NotImplementedError

    def _get_generator(self) -> nn.Module:
        generator = DCGANGenerator(self.hparams.latent_dim, self.hparams.feature_maps_gen, self.hparams.image_channels)
        generator.apply(self._weights_init)
        return generator

    def _get_discriminator(self) -> nn.Module:
        discriminator = DCGANDiscriminator(self.hparams.feature_maps_disc, self.hparams.image_channels)
        discriminator.apply(self._weights_init)
        return discriminator

    @staticmethod
    def _weights_init(m):
        classname = m.__class__.__name__
        if classname.find("Conv") != -1:
            torch.nn.init.normal_(m.weight, 0.0, 0.02)
        elif classname.find("BatchNorm") != -1:
            torch.nn.init.normal_(m.weight, 1.0, 0.02)
            torch.nn.init.zeros_(m.bias)

    def configure_optimizers(self):
        lr = self.hparams.learning_rate
        betas = (self.hparams.beta1, 0.999)
        optimiser_step_ratio = self.hparams.optimiser_step_ratio
        opt_disc = torch.optim.Adam(self.discriminator.parameters(), lr=lr, betas=betas)
        opt_gen = torch.optim.Adam(self.generator.parameters(), lr=lr, betas=betas)
        if optimiser_step_ratio >= 1:
            return ({'optimizer': opt_disc, 'frequency': int(optimiser_step_ratio)},
                    {'optimizer': opt_gen, 'frequency': 1})
        else:
            return ({'optimizer': opt_disc, 'frequency': 1},
                    {'optimizer': opt_gen, 'frequency': int(1/optimiser_step_ratio)})

    def forward(self, noise: Tensor) -> Tensor:
        """
        Generates an image given input noise
        Example::
            noise = torch.rand(batch_size, latent_dim)
            gan = GAN.load_from_checkpoint(PATH)
            img = gan(noise)
        """
        noise = noise.view(*noise.shape, 1, 1)
        return self.generator(noise)

    def training_step(self, batch, batch_idx, optimizer_idx):
        real = batch

        # Train discriminator
        result = None
        if optimizer_idx == 0:
            if self.hparams.loss == 'Wasserstein':
                for p in self.discriminator.parameters():
                    p.data.clamp_(self.hparams.clamp_lower, self.hparams.clamp_upper) 
            result = self._disc_step(real)

        # Train generator
        if optimizer_idx == 1:
            result = self._gen_step(real)

        return result

    def _disc_step(self, real: Tensor) -> Tensor:
        # Train with real
        real_pred = self.discriminator(real)
        # Train with fake
        fake_pred = self._get_fake_pred(real)
        disc_loss = self.criterion.discriminator_loss(real_pred, fake_pred)
        self.log('pred/real', real_pred.mean(), on_epoch=True)
        self.log('pred/fake', fake_pred.mean(), on_epoch=True)
        self.log("loss/disc", disc_loss, on_epoch=True)
        return disc_loss

    def _gen_step(self, real: Tensor) -> Tensor:
        # Train with fake
        fake_pred = self._get_fake_pred(real)
        gen_loss = self.criterion.generator_loss(fake_pred)
        self.log("loss/gen", gen_loss, on_epoch=True)
        return gen_loss

    def _get_fake_pred(self, real: Tensor) -> Tensor:
        batch_size = len(real)
        noise = self._get_noise(batch_size, self.hparams.latent_dim)
        fake = self(noise)
        fake_pred = self.discriminator(fake)
        return fake_pred

    def _get_noise(self, n_samples: int, latent_dim: int) -> Tensor:
        return torch.randn(n_samples, latent_dim, device=self.device)

    def validation_step(self, batch, batch_idx, dataloader_idx):
        if dataloader_idx == 0:
            # calculate metrics on holdout dataset 
            # real
            real_pred = self.discriminator(batch)
            # fake
            batch_size = len(batch)
            noise = self._get_noise(batch_size, self.hparams.latent_dim)
            fake = self(noise)
            fake_pred = self.discriminator(fake)

            real_loss, fake_loss = self.criterion.discriminator_loss(real_pred, fake_pred, breakdown=True)        
            self.log('pred/real_val', real_pred.mean(), on_epoch=True)
            self.log('pred/fake_val', fake_pred.mean(), on_epoch=True)
            self.log("loss/disc_real", real_loss, on_epoch=True)
            self.log("loss/disc_fake", fake_loss, on_epoch=True)
            return fake

        elif dataloader_idx == 1:
            self.embedder.to(self.device)
            with torch.no_grad():
                batch_emb = self.embedder(batch)

            batch_size = len(batch)
            noise = self._get_noise(batch_size, self.hparams.latent_dim)
            fake = self(noise)
            with torch.no_grad():
                fake_emb = self.embedder(fake)
            out = torch.stack((batch_emb, fake_emb), -1)
            return out

    def validation_epoch_end(self, outputs) -> None:
        val_out, train_out = outputs
        # use val out to save some images and train out to calculate the metrics 

        train_out = torch.cat(train_out, 0)
        train_out = np.array(train_out.detach().cpu())
        # precision, recall, density, coverage
        metrics = compute_prdc(train_out[:, :, 0], train_out[:, :, 1], self.hparams.k_neigh)
        # FID
        mus = [np.mean(train_out[:, :, i], axis=0) for i in range(2)]
        sigmas = [np.cov(train_out[:, :, i], rowvar=False) for i in range(2)]
        metrics['FID'] = calculate_frechet_distance(mus[0], sigmas[0], mus[1], sigmas[1])
        # log
        self.log_dict({'metrics/' + key: metrics[key] for key in metrics})

        # log images
        val_out = torch.cat(val_out, 0)
        grid = torchvision.utils.make_grid(val_out[-12:])
        self.logger.log_image('gen_imgs', grid, self.current_epoch)

    @staticmethod
    def add_model_specific_args(parent_parser: ArgumentParser) -> ArgumentParser:
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--beta1", default=0.5, type=float)
        parser.add_argument("--feature_maps_gen", default=64, type=int)
        parser.add_argument("--feature_maps_disc", default=64, type=int)
        parser.add_argument("--latent_dim", default=100, type=int)
        parser.add_argument("--learning_rate", default=0.0002, type=float)
        parser.add_argument("--optimiser_step_ratio", "-osr", default=1, type=float)
        parser.add_argument("--k_neigh", default=5, type=int)
        parser.add_argument("--loss", default='CE', type=str, choices=('CE', 'Wasserstein', 'GradientPenalty', 'Legacy', 'CESmoothed'))
        parser.add_argument('--clamp_lower', type=float, default=-0.01)
        parser.add_argument('--clamp_upper', type=float, default=0.01)
        return parser
