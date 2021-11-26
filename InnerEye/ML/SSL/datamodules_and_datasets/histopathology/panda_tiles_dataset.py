from health_ml.data.histopathology.datasets.panda_tiles_dataset import PandaTilesDatasetReturnImageLabel
from InnerEye.ML.SSL.datamodules_and_datasets.dataset_cls_utils import InnerEyeDataClassBaseWithReturnIndex

class PandaTilesDatasetWithReturnIndex(InnerEyeDataClassBaseWithReturnIndex, PandaTilesDatasetReturnImageLabel):
    """
    Any dataset used in SSL needs to inherit from InnerEyeDataClassBaseWithReturnIndex as well as VisionData.
    This class is just a shorthand notation for this double inheritance. Please note that this class needs
    to override __getitem__(), this is why we need a separate PandaTilesDatasetReturnImageLabel.
    """
    @property
    def num_classes(self) -> int:
        return 2
