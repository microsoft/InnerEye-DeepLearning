# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

For each Pull Request, the affected code parts should be briefly described and added here in the "Upcoming" section.
Once a release is done, the "Upcoming" section becomes the release changelog, and a new empty "Upcoming" should be
created.
 
## Upcoming

### Added
- ([#385](https://github.com/microsoft/InnerEye-DeepLearning/pull/385)) Add the ability to train a model on multiple
nodes in AzureML. Example: Add `--num_nodes=2` to the commandline arguments to train on 2 nodes.
- ([#366](https://github.com/microsoft/InnerEye-DeepLearning/pull/366)) and
  ([#407](https://github.com/microsoft/InnerEye-DeepLearning/pull/407)) add new parameters to the `score.py` script of `use_dicom` and `result_zip_dicom_name`. If `use_dicom==True` then the input file should be
  a zip of a DICOM series. This will be unzipped and converted to Nifti format before processing. The 
  result will then be converted to a DICOM-RT file, zipped and stored as `result_zip_dicom_name`.

### Changed
- ([#385](https://github.com/microsoft/InnerEye-DeepLearning/pull/385)) Starting an AzureML run now uses the
`ScriptRunConfig` object, rather than the deprecated `Estimator` object.
- ([#385](https://github.com/microsoft/InnerEye-DeepLearning/pull/385)) When registering a model, the name of the 
Python execution environment is added as a tag. This tag is read when running inference, and the execution environment
is re-used.

### Fixed

### Removed

### Deprecated

## 0.2 (2021-01-29)

### Added
- ([#323](https://github.com/microsoft/InnerEye-DeepLearning/pull/323)) There are new model configuration fields 
(and hence, commandline options), in particular for controlling PyTorch Lightning (PL) training: 
  - `max_num_gpus` controls how many GPUs are used at most for training (default: all GPUs, value -1).
  - `pl_num_sanity_val_steps` controls the PL trainer flag `num_sanity_val_steps`
  - `pl_deterministic` controls the PL trainer flags `benchmark` and `deterministic`
  - `generate_report` controls if a HTML report will be written (default: True) 
  - `recovery_checkpoint_save_interval` determines how often a checkpoint for training recovery is saved.
- ([#336](https://github.com/microsoft/InnerEye-DeepLearning/pull/336)) New extensions of SegmentationModelBases `HeadAndNeckBase` and `ProstateBase`. 
Use these classes to build your own Head&Neck or Prostate models, by just providing a 
list of foreground classes.
- ([#363](https://github.com/microsoft/InnerEye-DeepLearning/pull/363)) Grouped dataset splits and k-fold cross-validation. This allows, for example, 
training on datasets with multiple images per subject without leaking data from the 
same subject across train/test/validation sets or cross-validation folds. To use this 
functionality, simply provide the name of the CSV grouping column (`group_column`) when creating the `DatasetSplits` object in your model config's `get_model_train_test_dataset_splits()` method. See the `InnerEye.ML.utils.split_dataset.DatasetSplits` class for details.

### Changed

- ([#323](https://github.com/microsoft/InnerEye-DeepLearning/pull/323)) The codebase has undergone a massive 
refactoring, to use PyTorch Lightning as the foundation for all training. As a consequence of that:
  - Training is now using Distributed Data Parallel with synchronized `batchnorm`. The number of GPUs to use can be 
  controlled by a new commandline argument `max_num_gpus`.
  - Several classes, like `ModelTrainingSteps*`, have been removed completely.
  - The final model is now always the one that is written at the end of all training epochs.
  - The old code that options to run full image inference at multiple epochs (i.e., multiple checkpoints), this
  has been removed, alongside the respective commandline options `save_start_epoch`, `save_step_epochs`, 
  `epochs_to_test`, `test_diff_epochs`, `test_step_epochs`, `test_start_epoch`
  - The commandline option `register_model_only_for_epoch` is now called `only_register_model`, and is boolean.
  - All metrics are written to AzureML and Tensorboard in a unified format. A training Dice score for 'bladder' would
  previously be called Train_Dice/bladder, now it is train/Dice/bladder.
  - Due to a different checkpoint format, it is no longer possible to use checkpoints written
  by the previous version of the code.
- The arguments of the `score.py` script changed: `data_root` -> `data_folder`, it no longer assumes a fixed
`data` subfolder. `project_root` -> `model_root`, `test_image_channels` -> `image_files`.
- By default, the visualization of patch sampling for segmentation models will run on only 1 image (down from 5).
This is because patch sampling is expensive to compute, taking 1min per large CT scan.
- ([#336](https://github.com/microsoft/InnerEye-DeepLearning/pull/336)) Renamed `HeadAndNeckBase` to `HeadAndNeckPaper`,
and `ProstateBase` to `ProstatePaper`.

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
