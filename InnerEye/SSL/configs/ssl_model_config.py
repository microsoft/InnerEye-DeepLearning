from InnerEye.SSL.configs.config_node import ConfigNode

config = ConfigNode()

config.dataset = ConfigNode()
# Dataset name. Available choices: "CIFAR10", "RSNAKaggle"
config.dataset.name = 'CIFAR10'
# Local Path to dataset directory. Automatically set for CIFAR10.
config.dataset.dataset_dir = ''
# Number of cpu workers to use in dataloaders.
config.dataset.num_workers = None


config.train = ConfigNode()
# Training seed.
config.train.seed = None
# Batch size for both training and validation
config.train.batch_size = None
# Runs will be saved to repo_root / SSL_EXPERIMENT_DIR / output_dir.
config.train.output_dir = None
# Whether to continue training from the last checkpoint of a run.
# The checkpoint is assumed to be already placed repo_root / SSL_EXPERIMENT_DIR / output_dir and named last_checkpoint
config.train.resume_from_last_checkpoint = False
# Starting learning rate
config.train.base_lr = None


config.train.self_supervision = ConfigNode()
# Which algorithm to use for SSL training. Choices: "byol", "sim_clr".
config.train.self_supervision.type = None
# Encoder to use: resnet18, resnet50, resnet101 or densenet121.
config.train.self_supervision.encoder_name = None
# Whether to use balanced binary cross-entropy loss for linear head training.
# Weights are calculated based on statistics of the training set.
config.train.self_supervision.use_balanced_binary_loss_for_linear_head = False
# At which frequency to save checkpoints.
config.train.checkpoint_period = 200


config.scheduler = ConfigNode()
# Total number of epochs to train for.
config.scheduler.epochs = None


# The parameters below are used to specify which preprocessing and augmentations to use and their strength for
# RSNAKaggle dataset.
# WARNING: all the parameters are IGNORED for CIFAR10 training where we use the default  SimCLRTrainDataTransform from
# lightning-bolts.
config.preprocess = ConfigNode()
config.preprocess.use_center_crop = False
config.preprocess.center_crop_size = 224
config.preprocess.resize = 256

config.augmentation = ConfigNode()
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

config.augmentation.random_crop = ConfigNode()
config.augmentation.random_crop.scale = (0.9, 1.0)

config.augmentation.elastic_transform = ConfigNode()
config.augmentation.elastic_transform.sigma = 4
config.augmentation.elastic_transform.alpha = 35
config.augmentation.elastic_transform.p_apply = 0.5

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
config.augmentation.random_color.saturation = 0.0

config.augmentation.gamma = ConfigNode()
config.augmentation.gamma.scale = (0.5, 1.5)

config.augmentation.label_smoothing = ConfigNode()
config.augmentation.label_smoothing.epsilon = 0.1

config.augmentation.random_erasing = ConfigNode()
config.augmentation.random_erasing.scale = (0.01, 0.1)
config.augmentation.random_erasing.ratio = (0.3, 3.3)


def get_default_model_config() -> ConfigNode:
    """
    Returns copy of default model config
    """
    return config.clone()
