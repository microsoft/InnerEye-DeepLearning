from typing import Tuple, Any

from InnerEye.ML.Histopathology.datamodules.base_module import TilesDataModule
from InnerEye.ML.Histopathology.datasets.tcga_crck_tiles_dataset import TcgaCrck_TilesDataset
from InnerEye.ML.utils.split_dataset import DatasetSplits


class TcgaCrckTilesDataModule(TilesDataModule):
    """ TcgaCrckTilesDataModule is the child class of TilesDataModule specific to TCGA-Crck dataset
    Method get_splits() returns the train, val, test splits from the TCGA-Crck dataset
    Methods train_dataloader(), val_dataloader() and test_dataloader() override the base class methods for bag loading
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

    def get_splits(self) -> Tuple[TcgaCrck_TilesDataset, TcgaCrck_TilesDataset, TcgaCrck_TilesDataset]:
        trainval_dataset = TcgaCrck_TilesDataset(self.root_path, train=True)
        splits = DatasetSplits.from_proportions(trainval_dataset.dataset_df.reset_index(),
                                                proportion_train=0.8,
                                                proportion_test=0.0,
                                                proportion_val=0.2,
                                                subject_column=trainval_dataset.TILE_ID_COLUMN,
                                                group_column=trainval_dataset.SLIDE_ID_COLUMN,
                                                random_seed=5)

        if self.number_of_cross_validation_splits > 1:
            # Function get_k_fold_cross_validation_splits() will concatenate train and val splits
            splits = splits.get_k_fold_cross_validation_splits(self.number_of_cross_validation_splits)[self.cross_validation_split_index]

        return (TcgaCrck_TilesDataset(self.root_path, dataset_df=splits.train),
                TcgaCrck_TilesDataset(self.root_path, dataset_df=splits.val),
                TcgaCrck_TilesDataset(self.root_path, train=False))
