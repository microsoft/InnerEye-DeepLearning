# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres
to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

For each Pull Request, the affected code parts should be briefly described and added here in the "Upcoming" section.
Once a release is done, the "Upcoming" section becomes the release changelog, and a new empty "Upcoming" should be
created.


## Upcoming

### Added
- ([#594](https://github.com/microsoft/InnerEye-DeepLearning/pull/594)) When supplying a "--tag" argument, the AzureML jobs use that value as the display name, to more easily distinguish run.
- ([#577](https://github.com/microsoft/InnerEye-DeepLearning/pull/577)) Commandline switch `monitor_gpu` to monitor
  GPU utilization via Lightning's `GpuStatsMonitor`, switch `monitor_loading` to check batch loading times via
  `BatchTimeCallback`, and `pl_profiler` to turn on the Lightning profiler (`simple`, `advanced`, or `pytorch`)
- ([#544](https://github.com/microsoft/InnerEye-DeepLearning/pull/544)) Add documentation for segmentation model evaluation.
- ([#465](https://github.com/microsoft/InnerEye-DeepLearning/pull/465/)) Adding ability to run segmentation inference
module on test data with partial ground truth files. (Also [522](https://github.com/microsoft/InnerEye-DeepLearning/pull/522).)
- ([#502](https://github.com/microsoft/InnerEye-DeepLearning/pull/502)) More flags for fine control of when to run inference.
- ([#492](https://github.com/microsoft/InnerEye-DeepLearning/pull/492)) Adding capability for regression tests for test
jobs that run in AzureML.
- ([#509](https://github.com/microsoft/InnerEye-DeepLearning/pull/509)) Run inference on registered models (single and
  ensemble) using the parameter `model_id`.
- ([#554](https://github.com/microsoft/InnerEye-DeepLearning/pull/554)) Added a parameter `pretraining_dataset_id` to
  `NIH_COVID_BYOL` to specify the name of the SSL training dataset.
- ([#560](https://github.com/microsoft/InnerEye-DeepLearning/pull/560)) Added pre-commit hooks.
- ([#559](https://github.com/microsoft/InnerEye-DeepLearning/pull/559)) Adding the accompanying code for the ["Active label cleaning: Improving dataset quality under resource constraints"](https://arxiv.org/abs/2109.00574) paper. The code can be found in the [InnerEye-DataQuality](InnerEye-DataQuality/README.md) subfolder. It provides tools for training noise robust models, running label cleaning simulation and loading our label cleaning benchmark datasets.
- ([#589](https://github.com/microsoft/InnerEye-DeepLearning/pull/589)) Add `LightningContainer.update_azure_config()`
  hook to enable overriding `AzureConfig` parameters from a container (e.g. `experiment_name`, `cluster`, `num_nodes`).
-([#603](https://github.com/microsoft/InnerEye-DeepLearning/pull/603)) Add histopathology module
-([#614](https://github.com/microsoft/InnerEye-DeepLearning/pull/614)) Checkpoint downloading falls back to looking into AzureML if no checkpoints on disk
-([#613](https://github.com/microsoft/InnerEye-DeepLearning/pull/613)) Add additional tests for histopathology datasets


### Changed
- ([#588](https://github.com/microsoft/InnerEye-DeepLearning/pull/588)) Replace SciPy with PIL.PngImagePlugin.PngImageFile to load png files.
- ([#576](https://github.com/microsoft/InnerEye-DeepLearning/pull/576)) The console output is no longer written to stdout.txt because AzureML handles that better now
- ([#531](https://github.com/microsoft/InnerEye-DeepLearning/pull/531)) Updated PL to 1.3.8, torchmetrics and pl-bolts and changed relevant metrics and SSL code API.
- ([#555](https://github.com/microsoft/InnerEye-DeepLearning/pull/555)) Make the SSLContainer compatible with new datasets
- ([#533](https://github.com/microsoft/InnerEye-DeepLearning/pull/533)) Better defaults for inference on ensemble children.
- ([#536](https://github.com/microsoft/InnerEye-DeepLearning/pull/536)) Inference will not run on the validation set by default, this can be turned on
via the `--inference_on_val_set` flag.
- ([#548](https://github.com/microsoft/InnerEye-DeepLearning/pull/548)) Many Azure-related functions have been moved
out of the toolbox, into the separate hi-ml Python package.
- ([#502](https://github.com/microsoft/InnerEye-DeepLearning/pull/502)) Renamed command line option 'perform_training_set_inference' to 'inference_on_train_set'. Replaced command line option 'perform_validation_and_test_set_inference' with the pair of options 'inference_on_val_set' and 'inference_on_test_set'.
- ([#496](https://github.com/microsoft/InnerEye-DeepLearning/pull/496)) All plots are now saved as PNG, rather than JPG.
- ([#497](https://github.com/microsoft/InnerEye-DeepLearning/pull/497)) Reducing the size of the code snapshot that
gets uploaded to AzureML, by skipping all test folders.
- ([#509](https://github.com/microsoft/InnerEye-DeepLearning/pull/509)) Parameter `extra_downloaded_run_id` has been
  renamed to `pretraining_run_checkpoints`.
- ([#526](https://github.com/microsoft/InnerEye-DeepLearning/pull/526)) Updated Covid config to use a multiclass
  formulation. Moved functions `create_metric_computers` and `compute_and_log_metrics` from `ScalarLightning` to
  `ScalarModelBase`.
- ([#554](https://github.com/microsoft/InnerEye-DeepLearning/pull/554)) Updated report in CovidModel. Set parameters
  in the config to run inference on both the validation and test sets by default.
- ([#584](https://github.com/microsoft/InnerEye-DeepLearning/pull/584)) SSL models write the optimizer state for the linear head to the checkpoint now.
- ([#594](https://github.com/microsoft/InnerEye-DeepLearning/pull/594)) Pytorch is now non-deterministic by default. Upgrade to AzureML-SDK 1.36
- ([#566](https://github.com/microsoft/InnerEye-DeepLearning/pull/566)) Update `hi-ml` dependency to `hi-ml-azure`.
- ([#572](https://github.com/microsoft/InnerEye-DeepLearning/pull/572)) Updated to new version of hi-ml package
- ([#596](https://github.com/microsoft/InnerEye-DeepLearning/pull/596)) Add `cudatoolkit=11.1` specification to environment.yml.
- ([#605](https://github.com/microsoft/InnerEye-DeepLearning/pull/605)) Make build jobs deterministic for regression testing.

### Fixed
- ([#606](https://github.com/microsoft/InnerEye-DeepLearning/pull/606)) Bug fix: registered models do not include the hi-ml submodule
- ([#593](https://github.com/microsoft/InnerEye-DeepLearning/pull/593)) Bug fix for hi-ml 0.1.11 issue (#130): empty mount point is turned into ".", which fails the AML job
- ([#587](https://github.com/microsoft/InnerEye-DeepLearning/pull/587)) Bug fix for regression in AzureML's handling of environments: upgrade to hi-ml 0.1.11
- ([#537](https://github.com/microsoft/InnerEye-DeepLearning/pull/537)) Print warning if inference is disabled but comparison requested.
- ([#567](https://github.com/microsoft/InnerEye-DeepLearning/pull/567)) fix pillow version.
- ([#546](https://github.com/microsoft/InnerEye-DeepLearning/pull/546)) Environment and hello_world_model documentation updated
- ([#525](https://github.com/microsoft/InnerEye-DeepLearning/pull/525)) Enable --store_dataset_sample
- ([#495](https://github.com/microsoft/InnerEye-DeepLearning/pull/495)) Fix model comparison.
- ([#547](https://github.com/microsoft/InnerEye-DeepLearning/pull/547)) The parameter pl_find_unused_parameters was no longer used
to initialize the DDP Plugin.
- ([#482](https://github.com/microsoft/InnerEye-DeepLearning/pull/482)) Check bool parameter is either true or false.
- ([#475](https://github.com/microsoft/InnerEye-DeepLearning/pull/475)) Bug in AML SDK meant that we could not train
any large models anymore because data loaders ran out of memory.
- ([#472](https://github.com/microsoft/InnerEye-DeepLearning/pull/472)) Correct model path for moving ensemble models.
- ([#494](https://github.com/microsoft/InnerEye-DeepLearning/pull/494)) Fix an issue where multi-node jobs for
LightningContainer models can get stuck at test set inference.
- ([#498](https://github.com/microsoft/InnerEye-DeepLearning/pull/498)) Workaround for the problem that downloading
multiple large checkpoints can time out.
- ([#515](https://github.com/microsoft/InnerEye-DeepLearning/pull/515)) Workaround for occasional issues with dataset
mounting and running matplotblib on some machines. Re-instantiated a disabled test.
- ([#509](https://github.com/microsoft/InnerEye-DeepLearning/pull/509)) Fix issue where model checkpoints were not loaded
in inference-only runs when using lightning containers.
- ([#553](https://github.com/microsoft/InnerEye-DeepLearning/pull/553)) Fix incomplete test data module setup in Lightning inference.
- ([#557](https://github.com/microsoft/InnerEye-DeepLearning/pull/557)) Fix issue where learning rate was not set
  correctly in the SimCLR module
- ([#558](https://github.com/microsoft/InnerEye-DeepLearning/pull/558)) Fix issue with the CovidModel config where model
  weights from a finetuning run were incompatible with the model architecture created for non-finetuning runs.
- ([#604](https://github.com/microsoft/InnerEye-DeepLearning/pull/604)) Fix issue where runs on a VM would download the dataset even when a local dataset is provided.
- ([#612](https://github.com/microsoft/InnerEye-DeepLearning/pull/612)) SSL online evaluator was not doing distributed training

### Removed

- ([#577](https://github.com/microsoft/InnerEye-DeepLearning/pull/577)) Removing the monitoring of batch loading time,
  use the `BatchTimeCallback` from `hi-ml` instead
- ([#542](https://github.com/microsoft/InnerEye-DeepLearning/pull/542)) Removed Windows test leg from build pipeline.
- ([#509](https://github.com/microsoft/InnerEye-DeepLearning/pull/509)) Parameters `local_weights_path` and
  `weights_url` can no longer be used to initialize a training run, only inference runs.
- ([#526](https://github.com/microsoft/InnerEye-DeepLearning/pull/526)) Removed `get_posthoc_label_transform` in
  class `ScalarModelBase`. Instead, functions `get_loss_function` and `compute_and_log_metrics` in
  `ScalarModelBase` can be implemented to compute the loss and metrics in a task-specific manner.
- ([#554](https://github.com/microsoft/InnerEye-DeepLearning/pull/554)) Removed cryptography from list of invalid
  packages in `test_invalid_python_packages` as it is already present as a dependency in our conda environment.
- ([#596](https://github.com/microsoft/InnerEye-DeepLearning/pull/596)) Removed obsolete `TrainGlaucomaCV` from PR build.
- ([#604](https://github.com/microsoft/InnerEye-DeepLearning/pull/604)) Removed all code that downloads datasets, this is now all handled by hi-ml

### Deprecated


## 0.3 (2021-06-01)

### Added

- ([#483](https://github.com/microsoft/InnerEye-DeepLearning/pull/483)) Allow cross validation with 'bring your own' Lightning models (without ensemble building).
- ([#489](https://github.com/microsoft/InnerEye-DeepLearning/pull/489)) Remove portal query for outliers.
- ([#488](https://github.com/microsoft/InnerEye-DeepLearning/pull/488)) Better handling of missing seriesId in segmentation cross validation reports.
- ([#454](https://github.com/microsoft/InnerEye-DeepLearning/pull/454)) Checking that labels are mutually exclusive.
- ([#447](https://github.com/microsoft/InnerEye-DeepLearning/pull/447/)) Added a sanity check to ensure there are no
  missing channels, nor missing files. If missing channels in the csv file or filenames associated with channels are
  incorrect, pipeline exits with error report before running training or inference.
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
- ([#444](https://github.com/microsoft/InnerEye-DeepLearning/pull/444)) Added setup scripts and documentation to work
with the FastMRI challenge datasets.
- ([#444](https://github.com/microsoft/InnerEye-DeepLearning/pull/444)) Git-related information is now printed to the
console for easier diagnostics.
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
- ([#456](https://github.com/microsoft/InnerEye-DeepLearning/pull/456)) Adding configs to train Covid detection models.
- ([#463](https://github.com/microsoft/InnerEye-DeepLearning/pull/463)) Add arguments `dirs_recursive` and
  `dirs_non_recursive` to `mypy_runner.py` to let users specify a list of directories to run mypy on.

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
- ([#444](https://github.com/microsoft/InnerEye-DeepLearning/pull/444)) The method `before_training_on_rank_zero` of
 the `LightningContainer` class has been renamed to `before_training_on_global_rank_zero`. The order in which the
 hooks are called has been changed.
- ([#458](https://github.com/microsoft/InnerEye-DeepLearning/pull/458)) Simplifying and generalizing the way we handle
  data augmentations for classification models. The pipelining logic is now taken care of by a ImageTransformPipeline
  class that takes as input a list of transforms to chain together. This pipeline takes of applying transforms on 3D or
  2D images. The user can choose to apply the same transformation for all channels (RGB example) or whether to apply
  different transformation for each channel (if each channel represents a different
  modality / time point for example). The pipeline can now work directly with out-of-the box torchvision transform
  (as long as they support [..., C, H, W] inputs). This allows to get rid of nearly all of our custom augmentations
  functions. The conversion from pipeline of image transformation to ScalarItemAugmentation is now taken care of under
  the hood, the user does not need to call this wrapper for each config class. In models derived from ScalarModelConfig
  to change which augmentations are applied to the images inputs (resp. segmentations inputs), users can override
  `get_image_transform` (resp. `get_segmentation_transform`). These two functions replace the old
  `get_image_sample_transforms` method. See `docs/building_models.md` for more information on augmentations.

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
- ([#458](https://github.com/microsoft/InnerEye-DeepLearning/pull/458)) Getting rid of all the unused code for
   RandAugment & Co. The user has now instead complete freedom to specify the set of augmentations to use.
- ([#468](https://github.com/microsoft/InnerEye-DeepLearning/pull/468)) Removed the `KneeSinglecoil` example model

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
