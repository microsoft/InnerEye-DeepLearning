from InnerEye.ML.configs.segmentation.ProstateBase import ProstateBase


class ProstatePaper(ProstateBase):
    def __init__(self) -> None:
        super().__init__(
            azure_dataset_id="89e385d5-a511-426c-9675-aa9f9e7e5a85_no_geonorm",
            save_start_epoch=20,
            disable_extra_postprocessing=True)
