#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------

# These GAN loss functions are additions to the ones defined in
# https://github.com/dccastro/Morpho-MNIST/blob/master/models/gan_loss.py

import torch


class LegacyLoss(GANLossType):

    @staticmethod
    def discriminator_loss(real_logit, fake_logit, breakdown=False):
        criterion = torch.nn.BCELoss()

        real_gt = torch.ones_like(real_logit).to(real_logit.device)
        real_loss = criterion(real_logit, real_gt)

        fake_gt = torch.zeros_like(fake_logit).to(real_logit.device)
        fake_loss = criterion(fake_logit, fake_gt)

        if breakdown:
            return real_loss, fake_loss
        else:
            return real_loss + fake_loss

    @staticmethod
    def generator_loss(fake_logit):
        criterion = torch.nn.BCELoss()
        fake_gt = torch.ones_like(fake_logit).to(fake_logit.device)
        return criterion(fake_logit, fake_gt)


class GANBCELoss(GANLossType):

    @staticmethod
    def discriminator_loss(real_logit, fake_logit, breakdown=False):
        if breakdown:
            return -torch.log(real_logit).mean(), -torch.log(1-fake_logit).mean()
        else:
            return -torch.log(real_logit).mean() - torch.log(1-fake_logit).mean()

    @staticmethod
    def generator_loss(fake_logit):
        return -torch.log(fake_logit).mean()



class GANBCELossDiscSmoothed(GANLossType):

    @staticmethod
    def discriminator_loss(real_logit, fake_logit, breakdown=False):
        real_smooth = (torch.rand(real_logit.shape) * .3).to(real_logit.device)
        fake_smooth = (torch.rand(real_logit.shape) * .3 + .7).to(fake_logit)
        if breakdown:
            return -(torch.log(real_logit) * real_smooth).mean(), -(torch.log(1-fake_logit) * fake_smooth).mean()
        else:
            return -(torch.log(real_logit) * real_smooth).mean() - (torch.log(1-fake_logit) * fake_smooth).mean()

    @staticmethod
    def generator_loss(fake_logit):
        return -torch.log(fake_logit).mean()


class SoftPlusLoss(GANLossType):
    @staticmethod
    def discriminator_loss(real_logit, fake_logit, breakdown=False):
        if breakdown:
            return torch.nn.functional.softplus(-real_logit).mean(), torch.nn.functional.softplus(fake_logit).mean()
        else:
            return torch.nn.functional.softplus(-real_logit).mean() + torch.nn.functional.softplus(fake_logit).mean()

    @staticmethod
    def generator_loss(fake_logit):
        return torch.nn.functional.softplus(-fake_logit).mean()
