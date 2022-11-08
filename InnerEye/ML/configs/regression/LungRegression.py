import pandas as pd

from InnerEye.ML.configs.segmentation.Lung import Lung
from InnerEye.ML.utils.split_dataset import DatasetSplits


class LungRegression(Lung):
    def __init__(self) -> None:
        super().__init__(
            azure_dataset_id="lung",
            train_batch_size=2,
            test_crop_size=(32, 144, 144),
            crop_size=(32, 144, 144),
            inference_stride_size=(32, 144, 144),
            num_epochs=150,
            pl_deterministic=True,
        )

    def get_model_train_test_dataset_splits(self, dataset_df: pd.DataFrame) -> DatasetSplits:
        train = list(map(str, range(0, 10)))
        val = list(map(str, range(10, 15)))
        test = list(map(str, range(15, 20)))

        return DatasetSplits.from_subject_ids(
            df=dataset_df,
            train_ids=train,
            val_ids=val,
            test_ids=test,
        )
