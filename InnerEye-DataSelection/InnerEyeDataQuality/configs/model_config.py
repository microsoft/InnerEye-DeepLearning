#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------

from .config_node import ConfigNode

config = ConfigNode()

config.pretty_name = ""
config.device = 'cuda'

# cuDNN
config.cudnn = ConfigNode()
config.cudnn.benchmark = True
config.cudnn.deterministic = False

config.dataset = ConfigNode()
config.dataset.name = 'CIFAR10'
config.dataset.dataset_dir = ''
config.dataset.image_size = 32
config.dataset.n_channels = 3
config.dataset.n_classes = 10
config.dataset.num_samples = None
config.dataset.noise_temperature = 1.0
config.dataset.noise_offset = 0.0
config.dataset.noise_rate = None
config.dataset.csv_to_ignore = None
config.dataset.cxr_consolidation_noise_rate = 0.10

config.model = ConfigNode()
# options: 'cifar', 'imagenet'
# Use 'cifar' for small input images
config.model.type = 'cifar'
config.model.name = 'resnet_preact'
config.model.init_mode = 'kaiming_fan_out'
config.model.use_dropout = False

config.model.resnet = ConfigNode()
config.model.resnet.depth = 110  # for cifar type model
config.model.resnet.n_blocks = [2, 2, 2, 2]  # for imagenet type model
config.model.resnet.block_type = 'basic'
config.model.resnet.initial_channels = 16
config.model.resnet.apply_l2_norm = False  # if set to True, last activations are l2-normalised.

config.model.densenet = ConfigNode()
config.model.densenet.depth = 100  # for cifar type model
config.model.densenet.n_blocks = [6, 12, 24, 16]  # for imagenet type model
config.model.densenet.block_type = 'bottleneck'
config.model.densenet.growth_rate = 12
config.model.densenet.drop_rate = 0.0
config.model.densenet.compression_rate = 0.5

config.train = ConfigNode()
config.train.root_dir = ''
config.train.checkpoint = ''
config.train.resume_epoch = 0
config.train.restore_scheduler = True
config.train.batch_size = 128
# optimizer (options: sgd, adam)
config.train.optimizer = 'sgd'
config.train.base_lr = 0.1
config.train.momentum = 0.9
config.train.nesterov = True
config.train.weight_decay = 1e-4
config.train.start_epoch = 0
config.train.seed = 0
config.train.pretrained = False
config.train.no_weight_decay_on_bn = False
config.train.use_balanced_sampler = False

config.train.output_dir = 'experiments/exp00'
config.train.log_period = 100
config.train.checkpoint_period = 10

config.train.use_elr = False

# co_teaching defaults
config.train.use_co_teaching = False
config.train.use_teacher_model = False
config.train.co_teaching_consistency_loss = False
config.train.co_teaching_forget_rate = 0.2
config.train.co_teaching_num_gradual = 10
config.train.co_teaching_use_graph = False
config.train.co_teaching_num_warmup = 25

# self-supervision defaults
config.train.use_self_supervision = False
config.train.self_supervision = ConfigNode()
config.train.self_supervision.checkpoints = ['', '']
config.train.self_supervision.freeze_encoder = True
config.train.tanh_regularisation = 0.0

# optimizer
config.optim = ConfigNode()
# Adam
config.optim.adam = ConfigNode()
config.optim.adam.betas = (0.9, 0.999)

# scheduler
config.scheduler = ConfigNode()
config.scheduler.epochs = 160

# warm up (options: none, linear, exponential)
config.scheduler.warmup = ConfigNode()
config.scheduler.warmup.type = 'none'
config.scheduler.warmup.epochs = 0
config.scheduler.warmup.start_factor = 1e-3
config.scheduler.warmup.exponent = 4

# main scheduler (options: constant, linear, multistep, cosine)
config.scheduler.type = 'multistep'
config.scheduler.milestones = [80, 120]
config.scheduler.lr_decay = 0.1
config.scheduler.lr_min_factor = 0.001

# tensorboard
config.tensorboard = ConfigNode()
config.tensorboard.save_events = True

# train data loader
config.train.dataloader = ConfigNode()
config.train.dataloader.num_workers = 2
config.train.dataloader.drop_last = True
config.train.dataloader.pin_memory = False
config.train.dataloader.non_blocking = False

# validation data loader
config.validation = ConfigNode()
config.validation.batch_size = 256
config.validation.dataloader = ConfigNode()
config.validation.dataloader.num_workers = 2
config.validation.dataloader.drop_last = False
config.validation.dataloader.pin_memory = False
config.validation.dataloader.non_blocking = False

config.augmentation = ConfigNode()
config.augmentation.use_random_crop = True
config.augmentation.use_random_horizontal_flip = True
config.augmentation.use_random_affine = False
config.augmentation.use_label_smoothing = False
config.augmentation.use_random_color = False
config.augmentation.add_gaussian_noise = False
config.augmentation.use_gamma_transform = False
config.augmentation.use_random_erasing = False
config.augmentation.use_elastic_transform = False

config.augmentation.elastic_transform = ConfigNode()
config.augmentation.elastic_transform.sigma = 4
config.augmentation.elastic_transform.alpha = 35
config.augmentation.elastic_transform.p_apply = 0.5

config.augmentation.random_crop = ConfigNode()
config.augmentation.random_crop.scale = (0.9, 1.0)

config.augmentation.gaussian_noise = ConfigNode()
config.augmentation.gaussian_noise.std = 0.01
config.augmentation.gaussian_noise.p_apply = 0.5

config.augmentation.random_horizontal_flip = ConfigNode()
config.augmentation.random_horizontal_flip.prob = 0.5

config.augmentation.random_affine = ConfigNode()
config.augmentation.random_affine.max_angle = 0
config.augmentation.random_affine.max_horizontal_shift = 0.0
config.augmentation.random_affine.max_vertical_shift = 0.0
config.augmentation.random_affine.max_shear = 5

config.augmentation.random_color = ConfigNode()
config.augmentation.random_color.brightness = 0.0
config.augmentation.random_color.contrast = 0.1
config.augmentation.random_color.saturation = 0.1

config.augmentation.gamma = ConfigNode()
config.augmentation.gamma.scale = (0.5, 1.5)

config.augmentation.label_smoothing = ConfigNode()
config.augmentation.label_smoothing.epsilon = 0.1

config.augmentation.random_erasing = ConfigNode()
config.augmentation.random_erasing.scale = (0.01, 0.1)
config.augmentation.random_erasing.ratio = (0.3, 3.3)

config.preprocess = ConfigNode()
config.preprocess.use_resize = False
config.preprocess.use_center_crop = False
config.preprocess.center_crop_size = None
config.preprocess.histogram_normalization = ConfigNode()
config.preprocess.histogram_normalization.disk_size = 30
config.preprocess.resize = 32

# test config
config.test = ConfigNode()
config.test.checkpoint = None
config.test.output_dir = None
config.test.batch_size = 256

# test data loader
config.test.dataloader = ConfigNode()
config.test.dataloader.num_workers = 2
config.test.dataloader.pin_memory = False


def get_default_model_config() -> ConfigNode:
    return config.clone()
