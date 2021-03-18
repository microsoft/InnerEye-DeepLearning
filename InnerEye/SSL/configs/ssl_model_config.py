from InnerEye.SSL.configs.config_node import ConfigNode

config = ConfigNode()

config.dataset = ConfigNode()
config.dataset.name = 'CIFAR10'
config.dataset.dataset_dir = ''
config.dataset.num_workers = None

config.train = ConfigNode()
config.train.seed = None
config.train.batch_size = None
config.train.output_dir = None
config.train.resume_from_last_checkpoint = False
config.train.base_lr = None
config.train.self_supervision = ConfigNode()
config.train.self_supervision.type = None
config.train.self_supervision.encoder_name = None
config.train.self_supervision.use_balanced_binary_loss_for_linear_head = False
config.train.checkpoint_period = 200

config.scheduler = ConfigNode()
config.scheduler.epochs = None

config.augmentation = ConfigNode()
config.augmentation.use_random_crop = False
config.augmentation.use_random_horizontal_flip = False
config.augmentation.use_random_affine = False
config.augmentation.use_label_smoothing = False
config.augmentation.use_random_color = False
config.augmentation.add_gaussian_noise = False
config.augmentation.use_gamma_transform = False
config.augmentation.use_random_erasing = False
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
config.preprocess.center_crop_size = 224
config.preprocess.histogram_normalization = ConfigNode()
config.preprocess.histogram_normalization.disk_size = 30
config.preprocess.resize = 32

def get_default_model_config() -> ConfigNode:
    return config.clone()
