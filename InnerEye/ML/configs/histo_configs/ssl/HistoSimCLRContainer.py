from typing import Any, Optional, Tuple, List
from yacs.config import CfgNode

from torchvision.transforms import ColorJitter, RandomHorizontalFlip, RandomGrayscale, RandomResizedCrop, GaussianBlur, Lambda, RandomApply
# from InnerEyePrivate.Histopathology.utils.data_augmentations import HEDJitter, StainNormalization

from InnerEye.ML.augmentations.transform_pipeline import ImageTransformationPipeline
from InnerEye.ML.SSL.datamodules_and_datasets.transforms_utils import DualViewTransformWrapper
from InnerEye.ML.SSL.lightning_containers.ssl_container import SSLContainer


class HistoSSLContainer(SSLContainer):
    """
    Config to train SSL model on one of the histo datasets (e.g. PANDA, CRCk). The main reason to create a
    histo specific SSL class is to overwrite the augmentations that will be applied. Augmentation can be configured by
    using a configuration yml file or by specifying the set of transformations in the _get_transforms method.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(pl_find_unused_parameters=True, **kwargs)

    def _get_transforms(self, augmentation_config: Optional[CfgNode],
                        dataset_name: str, is_ssl_encoder_module: bool) -> Tuple[Any, Any]:
        if augmentation_config:
            return super()._get_transforms(augmentation_config, dataset_name, is_ssl_encoder_module)
        else:
            # is_ssl_encoder_module will be True for ssl training, False for linear head training
            train_transforms = self.get_transforms(apply_augmentations=True)
            val_transforms = self.get_transforms(apply_augmentations=is_ssl_encoder_module)

            if is_ssl_encoder_module:
                train_transforms = DualViewTransformWrapper(train_transforms)
                val_transforms = DualViewTransformWrapper(val_transforms)
        return train_transforms, val_transforms

    @staticmethod
    def get_transforms(apply_augmentations: bool) -> ImageTransformationPipeline:
        transforms: List[Any] = []
        if apply_augmentations:
            # SimClr augmentations
            transforms = [RandomResizedCrop(size=224),
                          RandomHorizontalFlip(p=0.5),
                          RandomApply([ColorJitter(brightness=0.8, contrast=0.8, saturation=0.8, hue=0.2)], 0.8),
                          RandomGrayscale(p=0.2),
                          GaussianBlur(int(224 * 0.1) + 1)]
        else:
            # TODO Are there some transformations that we want to apply anyway?
            # not sure it will work without, DualViewTransformWrapper will call
            # an empty list
            transforms += [Lambda(lambda x: x)]
        pipeline = ImageTransformationPipeline(transforms)
        return pipeline
