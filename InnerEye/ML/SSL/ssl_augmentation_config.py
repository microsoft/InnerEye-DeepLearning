#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
from yacs.config import CfgNode

config = CfgNode()

config.preprocess = CfgNode()
config.preprocess.center_crop_size = 224  # Images are first resized to 256x25"
config.preprocess.resize = 256  # Then at the end center cropped to 224x224

config.augmentation = CfgNode()
# Whether to apply random crop of scale config.augmentation.random_crop.scale,
# final crop is then resized to config.preprocess.resize cf. torchvision
# RandomCropResize function. If use_random_crop set to False,
# only apply resize.
config.augmentation.use_random_crop = False
# Whether to apply random horizontal flip applied with
# probability config.augmentation.random_horizontal_flip.prob.
config.augmentation.use_random_horizontal_flip = False
# Whether to apply random affine transformation parametrized by random_affine.max_angle,
# random_affine.max_horizontal_shift, random_affine.max_vertical_shift and max_shear.
config.augmentation.use_random_affine = False
# Whether to apply random color brightness and contrast transformation
# parametrized by random_color.brightness and random_color.contrast.
config.augmentation.use_random_color = False
# Whether to add gaussian noise with probability
# gaussian_noise.p_apply and standardized deviation gaussian_noise.std.
config.augmentation.add_gaussian_noise = False
# Whether to apply gamma transform with scale gamma.scale
config.augmentation.use_gamma_transform = False
# Whether to apply CutOut of a portion of the image. Size of cutout is
# parametrized by random_erasing.scale,tuple setting the minimum and maximum percentage
# of the image to remove and by random_erasing.ratio setting
# the min and max ratio of the erased patch.
config.augmentation.use_random_erasing = False
# Whether to apply elastic transforms. If True, the transformation
# is applied with probability elastic_transform.p_apply.And is parametrized
# by elastic_transform.sigma and elastic_transform.alpha.
# See ElasticTransform class for more details.
config.augmentation.use_elastic_transform = False

config.augmentation.random_crop = CfgNode()
config.augmentation.random_crop.scale = (0.9, 1.0)

config.augmentation.elastic_transform = CfgNode()
config.augmentation.elastic_transform.sigma = 4
config.augmentation.elastic_transform.alpha = 35
config.augmentation.elastic_transform.p_apply = 0.5

config.augmentation.gaussian_noise = CfgNode()
config.augmentation.gaussian_noise.std = 0.01
config.augmentation.gaussian_noise.p_apply = 0.5

config.augmentation.random_horizontal_flip = CfgNode()
config.augmentation.random_horizontal_flip.prob = 0.5

config.augmentation.random_affine = CfgNode()
config.augmentation.random_affine.max_angle = 0
config.augmentation.random_affine.max_horizontal_shift = 0.0
config.augmentation.random_affine.max_vertical_shift = 0.0
config.augmentation.random_affine.max_shear = 5

config.augmentation.random_color = CfgNode()
config.augmentation.random_color.brightness = 0.0
config.augmentation.random_color.contrast = 0.1
config.augmentation.random_color.saturation = 0.0

config.augmentation.gamma = CfgNode()
config.augmentation.gamma.scale = (0.5, 1.5)

config.augmentation.label_smoothing = CfgNode()
config.augmentation.label_smoothing.epsilon = 0.1

config.augmentation.random_erasing = CfgNode()
config.augmentation.random_erasing.scale = (0.01, 0.1)
config.augmentation.random_erasing.ratio = (0.3, 3.3)


def get_default_model_config() -> CfgNode:
    """
    Returns copy of default model config
    """
    return config.clone()
