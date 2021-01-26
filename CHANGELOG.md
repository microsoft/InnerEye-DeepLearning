# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

For each Pull Request, the affected code parts should be briefly described and added here in the "Upcoming" section.
Once a release is done, the "Upcoming" section becomes the release changelog, and a new empty "Upcoming" should be
created.
 
## Upcoming

### Added
- There are new commandline options for controlling PyTorch Lightning training: 
  - `max_num_gpus` controls how many GPUs are used at most for training (default: all GPUs, value -1).
  - `generate_report` controls if a HTML report will be written (default: True) 
  - `recovery_checkpoint_save_interval` determines how often a checkpoint for training recovery is saved.
- New extensions of SegmentationModelBases `HeadAndNeckBase` and `ProstateBase`. Use these classes to build your own Head&Neck or Prostate models, by just providing a list of foreground classes.
- Grouped dataset splits and k-fold cross-validation. This allows, for example, training on datasets with multiple images per subject without leaking data from the same subject across train/test/validation sets or cross-validation folds. To use this functionality, simply provide the name of the CSV grouping column (`group_column`) when creating the `DatasetSplits` object in your model config's `get_model_train_test_dataset_splits()` method. See the `InnerEye.ML.utils.split_dataset.DatasetSplits` class for details.

### Changed

- The codebase has undergone a massive refactoring, to use PyTorch Lightning as the foundation for all training. As
a consequence of that:
  - Several classes, like `ModelTrainingSteps*`, have been removed completely.
  - The final model is now always the one that is written at the end of all training epochs.
  - The commandline option `register_model_only_for_epoch` is now called `only_register_model`, and is boolean.
  - All metrics are written to AzureML and Tensorboard in a unified format. A training Dice score for 'bladder' would
  previously be called Train_Dice/bladder, now it is train/Dice/bladder.
  - Due to a different checkpoint format, it is no longer possible to use checkpoints written
  by the previous version of the code.
- The arguments of the `score.py` script changed: `data_root` -> `data_folder`, it no longer assumes a fixed
`data` subfolder. `project_root` -> `model_root`, `test_image_channels` -> `image_files`.
- By default, the visualization of patch sampling for segmentation models will run on only 1 image (down from 5).
This is because patch sampling is expensive to compute, taking 1min per large CT scan.
- Renamed `HeadAndNeckBase` to `HeadAndNeckPaper`, and `ProstateBase` to `ProstatePaper`.

### Fixed
- When registering a model, it now has a consistent folder structured, described [here](docs/deploy_on_aml.md). This
folder structure is present irrespective of using InnerEye as a submodule or not. In particular, exactly 1 Conda
environment will be contained in the model.

### Removed
- The commandline options to control which checkpoint is saved, and which is used for inference, have been removed:
`save_start_epoch`, `save_step_epochs`, `epochs_to_test`, `test_diff_epochs`, `test_step_epochs`, `test_start_epoch`
- Removed blobxfer completely. When downloading a dataset from Azure, we now use AzureML dataset downloading tools.
Please remove the following fields from your settings.yml file: 'datasets_storage_account' and 'datasets_container'. 
- Removed `ProstatePaperBase`.
- Removed ability to perform sub-fold cross validation. The parameters `number_of_cross_validation_splits_per_fold` 
and `cross_validation_sub_fold_split_index` have been removed from ScalarModelBase.

### Deprecated



## 0.1 (2020-11-13)
- This is the baseline release.
