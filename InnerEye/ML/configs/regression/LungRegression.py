import pandas as pd

from InnerEye.ML.configs.segmentation.Lung import Lung
from InnerEye.ML.utils.split_dataset import DatasetSplits


class LungRegression(Lung):
    """
    Model used for regression testing the Lung model. Uses the same data as the published model
    but only runs for a small number of epochs and a smaller subset of the data. This is to ensure that
    the training time is kept under 30 minutes to prevents slowing down PRs significantly.
    """

    def __init__(self) -> None:
        super().__init__(
            azure_dataset_id="lung_for_regression_test_11_2022",
            train_batch_size=3,
            num_epochs=25,
            pl_deterministic=True,
            test_crop_size=(64, 256, 256),
        )

    def get_model_train_test_dataset_splits(self, dataset_df: pd.DataFrame) -> DatasetSplits:
        train = list(map(str, range(0, 10)))
        val = list(map(str, range(10, 12)))
        test = list(map(str, range(12, 13)))

        return DatasetSplits.from_subject_ids(
            df=dataset_df,
            train_ids=train,
            val_ids=val,
            test_ids=test,
        )
