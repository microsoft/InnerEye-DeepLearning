# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres
to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

For each Pull Request, the affected code parts should be briefly described and added here in the "Upcoming" section.
Once a release is done, the "Upcoming" section becomes the release changelog, and a new empty "Upcoming" should be
created.

## Upcoming

### Added

- ([#446](https://github.com/microsoft/InnerEye-DeepLearning/pull/446)) Guarding `save_outlier` so that it works when 
institution id and series id columns are missing.
- ([#441](https://github.com/microsoft/InnerEye-DeepLearning/pull/441)) Add script to move models from one AzureML workspace to another: `python InnerEye/Scripts/move_model.py`
- ([#417](https://github.com/microsoft/InnerEye-DeepLearning/pull/417)) Added a generic way of adding PyTorch Lightning
models to the toolbox. It is now possible to train almost any Lightning model with the InnerEye toolbox in AzureML,
with only minimum code changes required. See [the MD documentation](docs/bring_your_own_model.md) for details.
- ([#430](https://github.com/microsoft/InnerEye-DeepLearning/pull/430)) Update conversion to 1.0.1 InnerEye-DICOM-RT to
  add:  manufacturer, SoftwareVersions, Interpreter and ROIInterpretedTypes.
- ([#385](https://github.com/microsoft/InnerEye-DeepLearning/pull/385)) Add the ability to train a model on multiple
  nodes in AzureML. Example: Add `--num_nodes=2` to the commandline arguments to train on 2 nodes.
- ([#366](https://github.com/microsoft/InnerEye-DeepLearning/pull/366)) and
  ([#407](https://github.com/microsoft/InnerEye-DeepLearning/pull/407)) add new parameters to the `score.py` script
  of `use_dicom` and `result_zip_dicom_name`. If `use_dicom==True` then the input file should be a zip of a DICOM
  series. This will be unzipped and converted to Nifti format before processing. The result will then be converted to a
  DICOM-RT file, zipped and stored as `result_zip_dicom_name`.
- ([#416](https://github.com/microsoft/InnerEye-DeepLearning/pull/416)) Add a github action chat checks
  if `CHANGELOG.md` has been modified.
- ([#412](https://github.com/microsoft/InnerEye-DeepLearning/pull/412)) Dataset files can now have arbitrary names, and
  are no longer restricted to be called
  `dataset.csv`, via the config field `dataset_csv`. This allows to have a single set of image files in a folder, but
  multiple datasets derived from it.
- ([#391](https://github.com/microsoft/InnerEye-DeepLearning/pull/391)) Support for multilabel classification tasks.
  Multilabel models can be trained by adding the parameter `class_names` to the config for classification models.
  `class_names` should contain the name of each label class in the dataset, and the order of names should match the
  order of class label indices in `dataset.csv`.
  `dataset.csv` supports multiple labels (indices corresponding to `class_names`) per subject in the label column.
  Multiple labels should be encoded as a string with labels separated by a `|`, for example "0|2|4". Note that this PR
  does not add support for multiclass models, where the labels are mutually exclusive.
- ([#425](https://github.com/microsoft/InnerEye-DeepLearning/pull/425)) The number of layers in a Unet is no longer
  fixed at 4, but can be set via the config field `num_downsampling_paths`. A lower number of layers may be useful for
  decreasing memory requirements, or for working with smaller images.
  (The minimum image size in any dimension when using a network of n layers is 2**n.)
- ([#426](https://github.com/microsoft/InnerEye-DeepLearning/pull/426)) Flake8, mypy, and testing the HelloWorld model
  is now happening in a Github action, no longer in Azure Pipelines.
- ([#405](https://github.com/microsoft/InnerEye-DeepLearning/pull/405)) Cross-validation runs for classification models
  now also generate a report notebook summarising the metrics from the individual splits. Also includes minor formatting
  improvements for standard classification reports.
- ([#438](https://github.com/microsoft/InnerEye-DeepLearning/pull/438)) Add links and small docs to InnerEye-Gateway and InnerEye-Inference
- ([#439](https://github.com/microsoft/InnerEye-DeepLearning/pull/439)) Enable automatic job recovery from last recovery
  checkpoint in case of job pre-emption on AML. Give the possibility to the user to keep more than one recovery
  checkpoint.
- ([#442](https://github.com/microsoft/InnerEye-DeepLearning/pull/442)) Enable defining custom scalar losses
  (`ScalarLoss.CustomClassification` and `CustomRegression`), prediction targets (`ScalarModelBase.target_names`),
  and reporting (`ModelConfigBase.generate_custom_report()`) in scalar configs, providing more flexibility for defining
  model configs with custom behaviour while leveraging the existing InnerEye workflows.
- ([#445](https://github.com/microsoft/InnerEye-DeepLearning/pull/445)) Adding test coverage for the `HelloContainer`
  model with multiple GPUs
- ([#450](https://github.com/microsoft/InnerEye-DeepLearning/pull/450)) Adds the metric "Accuracy at threshold 0.5" to the classification report (`classification_crossval_report.ipynb`). 
- ([#451](https://github.com/microsoft/InnerEye-DeepLearning/pull/451)) Write a file `model_outputs.csv` with columns 
  `subject`, `prediction_target`, `label`, `model_output` and `cross_validation_split_index`. This file is not written out for sequence models.
- ([#440](https://github.com/microsoft/InnerEye-DeepLearning/pull/440)) Added support for training of self-supervised
  models (BYOL and SimCLR) based on the bring-your-own-model framework. Providing examples configurations for training
  of SSL models on CIFAR10/100 datasets as well as for chest-x-ray datasets such as NIH CHest-Xray or RSNA Pneumonia
  Detection Challenge datasets. See
  [SSL doc](https://github.com/microsoft/InnerEye-DeepLearning/blob/main/docs/self_supervised_models.md) for more
  details.
- ([#455](https://github.com/microsoft/InnerEye-DeepLearning/pull/455)) All models trained on AzureML are registered.
  The codepath previously allowed only segmentation models (subclasses of `SegmentationModelBase`) to be registered.
  Models are registered after a training run or if the `only_register_model` flag is set. Models may be legacy InnerEye
  config-based models or may be defined using the LightningContainer class.
  Additionally, the `TrainHelloWorldAndHelloContainer` job in the PR build has been split into two jobs, `TrainHelloWorld` and
  `TrainHelloContainer`. A pytest marker `after_training_hello_container` has been added to run tests after training is
  finished in the `TrainHelloContainer` job.

### Changed

- ([#385](https://github.com/microsoft/InnerEye-DeepLearning/pull/385)) Starting an AzureML run now uses the
  `ScriptRunConfig` object, rather than the deprecated `Estimator` object.
- ([#385](https://github.com/microsoft/InnerEye-DeepLearning/pull/385)) When registering a model, the name of the Python
  execution environment is added as a tag. This tag is read when running inference, and the execution environment is
  re-used.
- ([#411](https://github.com/microsoft/InnerEye-DeepLearning/pull/411)) Upgraded to PyTorch 1.8.0, PyTorch-Lightning
  1.1.8 and AzureML SDK 1.23.0
- ([#432](https://github.com/microsoft/InnerEye-DeepLearning/pull/432)) Upgraded to PyTorch-Lightning 1.2.7. Add
  end-to-end test for classification cross-validation. WARNING: upgrade PL version causes hanging of multi-node
  training.
- ([#437](https://github.com/microsoft/InnerEye-DeepLearning/pull/437)) Upgrade to PyTorch-Lightning 1.2.8.
- ([#439](https://github.com/microsoft/InnerEye-DeepLearning/pull/439)) Recovery checkpoints are now
  named `recovery_epoch=x.ckpt` instead of `recovery.ckpt` or `recovery-v0.ckpt`.
- ([#451](https://github.com/microsoft/InnerEye-DeepLearning/pull/451)) Change the signature for function `generate_custom_report` 
  in `ModelConfigBase` to take only the path to the reports folder and a `ModelProcessing` object.
- ([#458](https://github.com/microsoft/InnerEye-DeepLearning/pull/458)) Simplifying and generalizing the way we handle 
  data augmentations: The pipelining logic is now taken care of by a ImageTransformPipeline class that takes as 
  input a list of transforms to chain together. This pipeline takes of applying transforms on 3D or 2D images. 
  The user can choose to apply the same transformation for all channels (RGB example) or whether to apply different 
  transformation for each channel (if each channel represents a different 
  modality / time point for example). The pipeline can now work directly with out-of-the box torchvision transform 
  (as long as they support [..., C, H, W] inputs). This allows to get rid of nearly all of our custom augmentations 
  functions. The conversion from pipeline of image transformation to ScalarItemAugmentation is now taken care of under 
  the hood, the user does not need to call this wrapper for each config class.

### Fixed

- ([#422](https://github.com/microsoft/InnerEye-DeepLearning/pull/422)) Documentation - clarified `setting_up_aml.md`
  datastore creation instructions and fixed small typos in `hello_world_model.md`
- ([#432](https://github.com/microsoft/InnerEye-DeepLearning/pull/432)) Fixed cross-validation for classification
  models. Fixed multi-gpu metrics aggregation. Add end-to-end test for classification cross-validation. Add fix to bug
  in ddp setting when running multi-node with 1 gpu per node.
- ([#435](https://github.com/microsoft/InnerEye-DeepLearning/pull/435)) If parameter `model` in `AzureConfig` is not
  set, display an error message and terminate the run.
- ([#437](https://github.com/microsoft/InnerEye-DeepLearning/pull/437)) Fixed multi-node DDP bug in PL v1.2.8. Re-add
  end-to-end test for multi-node.
- ([#445](https://github.com/microsoft/InnerEye-DeepLearning/pull/445)) Fixed a bug when running inference for
 container models on machines with >1 GPU

### Removed
- ([#439](https://github.com/microsoft/InnerEye-DeepLearning/pull/439)) Deprecated `start_epoch` config argument.
- ([#450](https://github.com/microsoft/InnerEye-DeepLearning/pull/450)) Delete unused `classification_report.ipynb`.
- ([#455](https://github.com/microsoft/InnerEye-DeepLearning/pull/455)) Removed the AzureRunner conda environment.
  The full InnerEye conda environment is needed to submit a training job to AzureML.
-  ([#458](https://github.com/microsoft/InnerEye-DeepLearning/pull/458)) Getting rid of all the unused code for 
   RandAugment & Co. The user has now instead complete freedom to specify the set of augmentations to use.

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
- ([#336](https://github.com/microsoft/InnerEye-DeepLearning/pull/336)) New extensions of
  SegmentationModelBases `HeadAndNeckBase` and `ProstateBase`. Use these classes to build your own Head&Neck or Prostate
  models, by just providing a list of foreground classes.
- ([#363](https://github.com/microsoft/InnerEye-DeepLearning/pull/363)) Grouped dataset splits and k-fold
  cross-validation. This allows, for example, training on datasets with multiple images per subject without leaking data
  from the same subject across train/test/validation sets or cross-validation folds. To use this functionality, simply
  provide the name of the CSV grouping column (`group_column`) when creating the `DatasetSplits` object in your model
  config's `get_model_train_test_dataset_splits()` method. See the `InnerEye.ML.utils.split_dataset.DatasetSplits` class
  for details.

### Changed

- ([#323](https://github.com/microsoft/InnerEye-DeepLearning/pull/323)) The codebase has undergone a massive
  refactoring, to use PyTorch Lightning as the foundation for all training. As a consequence of that:
    - Training is now using Distributed Data Parallel with synchronized `batchnorm`. The number of GPUs to use can be
      controlled by a new commandline argument `max_num_gpus`.
    - Several classes, like `ModelTrainingSteps*`, have been removed completely.
    - The final model is now always the one that is written at the end of all training epochs.
    - The old code that options to run full image inference at multiple epochs (i.e., multiple checkpoints), this has
      been removed, alongside the respective commandline options `save_start_epoch`, `save_step_epochs`,
      `epochs_to_test`, `test_diff_epochs`, `test_step_epochs`, `test_start_epoch`
    - The commandline option `register_model_only_for_epoch` is now called `only_register_model`, and is boolean.
    - All metrics are written to AzureML and Tensorboard in a unified format. A training Dice score for 'bladder' would
      previously be called Train_Dice/bladder, now it is train/Dice/bladder.
    - Due to a different checkpoint format, it is no longer possible to use checkpoints written by the previous version
      of the code.
- The arguments of the `score.py` script changed: `data_root` -> `data_folder`, it no longer assumes a fixed
  `data` subfolder. `project_root` -> `model_root`, `test_image_channels` -> `image_files`.
- By default, the visualization of patch sampling for segmentation models will run on only 1 image (down from 5). This
  is because patch sampling is expensive to compute, taking 1min per large CT scan.
- ([#336](https://github.com/microsoft/InnerEye-DeepLearning/pull/336)) Renamed `HeadAndNeckBase` to `HeadAndNeckPaper`,
  and `ProstateBase` to `ProstatePaper`.
- ([#427](https://github.com/microsoft/InnerEye-DeepLearning/pull/427)) Move dicom loading function from SimpleITK to
  pydicom. Loading time improved by 30x.

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
