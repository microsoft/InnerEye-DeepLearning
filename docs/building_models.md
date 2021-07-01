# Building Models

### Setting up training

To train new models, you can either work within the InnerEye/ directory hierarchy or create a local hierarchy beside it
and with the same internal organization (although with far fewer files).
We recommend the latter as it offers more flexibility and better separation of concerns. Here we will assume you
create a directory `InnerEyeLocal` beside `InnerEye`.

As well as your configurations (dealt with below) you will need these files:
* `settings.yml`: A file similar to `InnerEye\settings.yml` containing all your Azure settings.
The value of `extra_code_directory` should (in our example) be `'InnerEyeLocal'`, 
and model_configs_namespace should be `'InnerEyeLocal.ML.configs'`. 
* A folder like `InnerEyeLocal` that contains your additional code, and model configurations.
* A file `InnerEyeLocal/ML/runner.py` that invokes the InnerEye training runner, but that points the code to your environment and Azure
settings.
```python
from pathlib import Path
import os
from InnerEye.ML import runner


def main() -> None:
    current = os.path.dirname(os.path.realpath(__file__))
    project_root = Path(os.path.realpath(os.path.join(current, "..", "..")))
    runner.run(project_root=project_root,
               yaml_config_file=project_root / "relative/path/to/settings.yml",
               post_cross_validation_hook=None)


if __name__ == '__main__':
    main()
```

### Creating the model configuration

You will find a variety of model configurations [here](/InnerEye/ML/configs/segmentation). Those not ending
in `Base.py` reference open-sourced data and can be used as they are. Those ending in `Base.py`
are partially specified, and can be used by having other model configurations inherit from them and supply the missing
parameter values: a dataset ID at least, and optionally other values. For example, a `Prostate` model might inherit
very simply from `ProstateBase` by creating `Prostate.py` in the directory `InnerEyeLocal/ML/configs/segmentation` 
with the following contents:
```python
from InnerEye.ML.configs.segmentation.ProstateBase import ProstateBase


class Prostate(ProstateBase):
    def __init__(self) -> None:
        super().__init__(
            ground_truth_ids=["femur_r", "femur_l", "rectum", "prostate"],
            azure_dataset_id="name-of-your-AML-dataset-with-prostate-data")
```
The allowed parameters and their meanings are defined in [`SegmentationModelBase`](/InnerEye/ML/config.py).
The class name must be the same as the basename of the file containing it, so `Prostate.py` must contain `Prostate`. 
In `settings.yml`, set `model_configs_namespace` to `InnerEyeLocal.ML.configs` so this config  
is found by the runner.

A `Head and Neck` model might inherit from `HeadAndNeckBase` by creating `HeadAndNeck.py` with the following contents:
```python
from InnerEye.ML.configs.segmentation.HeadAndNeckBase import HeadAndNeckBase


class HeadAndNeck(HeadAndNeckBase):
    def __init__(self) -> None:
        super().__init__(
            ground_truth_ids=["parotid_l", "parotid_r", "smg_l", "smg_r", "spinal_cord"]
            azure_dataset_id="name-of-your-AML-dataset-with-prostate-data")
```

### Training a new model

* Set up your model configuration as above and update `azure_dataset_id` to the name of your Dataset in the AML workspace.
It is enough to put your dataset into blob storage. The dataset should be a contained in a folder at the root of the datasets container. 
The InnerEye runner will check if there is a dataset in the AzureML workspace already, and if not, generate it directly from blob storage.

* Train a new model, for example `Prostate`:
```shell script
python InnerEyeLocal/ML/runner.py --azureml --model=Prostate
```

Alternatively, you can train the model on your current machine if it is powerful enough. In
this case, you would simply omit the `azureml` flag, and instead of specifying
`azure_dataset_id` in the class constructor, you can instead use `local_dataset="my/data/folder"`,
where the folder `my/data/folder` contains a `dataset.csv` file and all the files that are referenced therein.

If your dataset is small you can try `--apply_augmentations=True` and increase the `--num_epochs=` for segmentation models. This will apply `BasicAugmentations(Transform3D[Sample])`.
This will apply these transforms: `RandomAffine(degrees=20), RandomNoise(), RandomMotion(), RandomBlur()`

### Boolean Options

Note that for command line options that take a boolean argument, and that are `False` by default, there are multiple ways of setting the option. For example alternatives to  `--azureml` include:
* `--azureml=True`, or `--azureml=true`, or `--azureml=T`, or `--azureml=t`
* `--azureml=Yes`, or `--azureml=yes`, or `--azureml=Y`, or `--azureml=y`
* `--azureml=On`, or `--azureml=on`
* `--azureml=1`

Conversely, for command line options that take a boolean argument, and that are `True` by default, there are multiple ways of un-setting the option. For example alternatives to `--no-train` include:
* `--train=False`, or `--train=false`, or `--train=F`, or `--train=f`
* `--train=No`, or `--train=no`, or `--train=N`, or `--train=n`
* `--train=Off`, or `--train=off`
* `--train=0`


### Training using multiple machines
To speed up training in AzureML, you can use multiple machines, by specifying the additional
`--num_nodes` argument. For example, to use 2 machines to train, specify:
```shell script
python InnerEyeLocal/ML/runner.py --azureml --model=Prostate --num_nodes=2
```
On each of the 2 machines, all available GPUs will be used. Model inference will always use only one machine.

For the Prostate model, we observed a 2.8x speedup for model training when using 4 nodes, and a 1.65x speedup 
when using 2 nodes.

### AzureML Run Hierarchy

AzureML structures all jobs in a hierarchical fashion:
* The top-level concept is a workspace
* Inside of a workspace, there are multiple experiments. Upon starting a training run, the name of the experiment
needs to be supplied. The InnerEye toolbox is set specifically to work with git repositories, and it automatically
sets the experiment name to match the name of the current git branch.
* Inside of an experiment, there are multiple runs. When starting the InnerEye toolbox as above, a run will be created.
* A run can have child runs - see below in the discussion about cross validation.


### K-Fold Model Cross Validation

For running K-fold cross validation, the InnerEye toolbox schedules multiple training runs in the cloud that run
at the same time (provided that the cluster has capacity). This means that a complete cross validation run usually
takes as long as a single training run.

To start cross validation, you can either modify the `number_of_cross_validation_splits` property of your model,
or supply it on the command line: provide all the usual switches, and add `--number_of_cross_validation_splits=N`, 
for some `N` greater than 1; a value of 5 is typical. This will start a 
[HyperDrive run](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-tune-hyperparameters): a parent
AzureML job, with `N` child runs that will execute in parallel. You can see the child runs in the AzureML UI in the
"Child Runs" tab.

The dataset splits for those `N` child runs will be
computed from the union of the Training and Validation sets. The Test set is unchanged. Note that the Test set can be
empty, in which case the union of all validation sets for the `N` child runs will be the full dataset.

### Recovering failed runs and continuing training

To train further with an already-created model, give the above command with additional switches like these:
```
--run_recovery_id=foo_bar:foo_bar_12345_abcd --start_epoch=120
```
The run recovery ID is of the form "experiment_id:run_id". When you trained your original model, it will have been
queued as a "Run" inside of an "Experiment". The experiment will be given a name derived from the branch name - for 
example, branch `foo/bar` will queue a run in experiment `foo_bar`. Inside the "Tags" section of your run, you should
see an element `run_recovery_id`. It will look something like `foo_bar:foo_bar_12345_abcd`.

If you are recovering a HyperDrive run, the value of `--run_recovery_id` should for the parent,
and `--number_of_cross_validation_splits` should have the same value as in the recovered run. 
For example:
```
--run_recovery_id=foo_bar:HD_55d4beef-7be9-45d7-89a5-1acf1f99078a --start_epoch=120 --number_of_cross_validation_splits=5
```

The run recovery ID of a parent HyperDrive run is currently not displayed in the "Details" section
of the AzureML UI. The easiest way to get it is to go to any of the child runs and use its
run recovery ID without the final underscore and digit.

### Testing an existing model
To evaluate an existing model on a test set, you can use models from previous runs in AzureML or from local checkpoints.

#### From a previus run in AzureML:
This is similar to continuing training using a run_recovery object, but you will need to set `--no-train`.
Thus your command should look like this:

```shell script
python Inner/ML/runner.py --azureml --model=Prostate --no-train --cluster=my_cluster_name \
   --run_recovery_id=foo_bar:foo_bar_12345_abcd --start_epoch=120
```
#### From a local checkpoint:
To evaluate a model using a local checkpoint, use the local_weights_path to specify the path to the model checkpoint 
and set train to `False`.
```shell script
python Inner/ML/runner.py --model=Prostate --no-train --local_weights_path=path_to_your_checkpoint
```

Alternatively, to submit an AzureML run to apply a model to a single image on your local disc, 
you can use the script `submit_for_inference.py`, with a command of this form:
```shell script
python InnerEye/Scripts/submit_for_inference.py --image_file ~/somewhere/ct.nii.gz --model_id Prostate:555 \
  --settings ../somewhere_else/settings.yml --download_folder ~/my_existing_folder
```

### Model Ensembles

An ensemble model will be created automatically and registered in the AzureML model registry whenever cross-validation
models are trained. The ensemble model creation is done by the child whose `cross_validation_split_index` is 0; 
you can identify this child by looking at the "Child Runs" tab in the parent run page in AzureML. 

To find the registered ensemble model, find the Hyperdrive parent run in AzureML. In the "Details" tab, there is an
entry for "Registered models", that links to the ensemble model that was just created. Note that each of the child runs
also registers a model, namely the one that was built off its specific subset of data, without taking into account
the other crossvalidation folds.

As well as registering the model, child run 0 runs the ensemble model on the validation and test sets. The results are
aggregated based on the `ensemble_aggregation_type` value in the model config,
and the generated posteriors are passed to the usual model testing downstream pipelines, e.g. metrics computation.


##### Interpreting results

Once your HyperDrive AzureML runs are completed, you can visualize the results by running the
[`plot_cross_validation.py`](/InnerEye/ML/visualizers/plot_cross_validation.py) script locally:
```shell script
python InnerEye/ML/visualizers/plot_cross_validation.py --run_recovery_id ... --epoch ...
```
filling in the run recovery ID of the parent run and the epoch number (one of the test epochs, e.g. the last epoch) 
for which you want results plotted. The script will also output several `..._outliers.txt` file with all of the outliers
across the splits and a portal query to 
find them in the production portal, and run statistical tests to compute the significance of differences between scores
across the splits and with respect to other runs that you specify. This is done for you during
 the run itself (see below), but you can use the script post hoc to compare arbitrary runs
 with each other. Details of the tests can be found
in [`wilcoxon_signed_rank_test.py`](/InnerEye/Common/Statistics/wilcoxon_signed_rank_test.py)
and [`mann_whitney_test.py`](/InnerEye/Common/Statistics/mann_whitney_test.py).

## Where are my outputs and models?

* AzureML writes all its results to the storage account you have specified. Inside of that account, you will
  find a container named `azureml`. You can access that with 
  [Azure StorageExplorer](https://azure.microsoft.com/en-us/features/storage-explorer/). The checkpoints and other 
  files of a run will be in folder `azureml/ExperimentRun/dcid.my_run_id`, where `my_run_id` is the "Run Id" visible in
  the "Details" section of the run. If you want to download all the results files or a large subset of them,
  we recommend you access them this way.
* The results can also be viewed in the "Outputs and Logs" section of the run. This is likely to be more
  convenient for viewing and inspecting single files.
* All files that the model training writes to the `./outputs` folder are automatically uploaded at the end of
  the AzureML training job, and are put into `outputs` in Blob Storage and in the run itself. 
  Similarly, what the model training writes to the `./logs` folder gets uploaded to `logs`.
* You can monitor the file system that is mounted on the compute node, by navigating to your
  storage account in Azure. In the blade, click on "Files" and, navigate through to `azureml/azureml/my_run_id`. This 
  will show all files that are mounted as the working directory on the compute VM.

The organization of the `outputs` directory is as follows:

* A `checkpoints` directory containing the checkpointed model file(s).
* For each test epoch `NNN`, a directory `epoch_NNN`, each of whose subdirectories `Test` and `Val`
contains the following:
  * A `metrics.csv` file, giving the Dice and Hausdorff scores for every structure
    of every subject in the test and validation sets respectively.
  * A `metrics_aggregates.csv` file, aggregating the information in `metrics.csv` by subject to give
    minimum, maximum, mean and standard deviation values for both Dice and Hausdorff scores.
  * A `metrics_boxplot.png` file, containing box-and-whisker plots for the same information.
  * Various files identifying the dataset and structure names.
  * A `thumbnails` directory, containing an image file for the maximal predicted slice for each
    structure of each test or validation subject.
  * For each test or validation subject, a directory containing a Nifti file for each predicted structure.
* If there are comparison runs (specified by the config parameter `comparison_blob_storage_paths`),
there will be a subdirectory named after each of those runs, each containing its own `epoch_NNN` subdirectory,
and there will be a file `MetricsAcrossAllRuns.csv` directly under `outputs`, combining the data from
the `metrics.csv` files of the current run and the comparison run(s).
* Additional files directly under `outputs`:
  * `args.txt` contains the configuration information.
  * `buildinformation.json` contains information on the build, partially overlapping with the content
  of the "Details" tab.
  * `dataset.csv` for the whole dataset (see ["Creating Datasets](creating_dataset.md) for details),
  and `test_dataset.csv`, `train_dataset.csv` and `val_dataset.csv` for those subsets of it.
  * `BaselineComparisonWilcoxonSignedRankTestResults.txt`, containing the results of comparisons
  between the current run and any specified baselines (earlier runs) to compare with. Each paragraph of that file compares two models and
  indicates, for each structure, when the Dice scores for the second model are significantly better 
  or worse than the first. For full details, see the 
  [source code](../InnerEye/Common/Statistics/wilcoxon_signed_rank_test.py).
  * A directory `scatterplots`, containing a `png` file for every pairing of the current model
  with one of the baselines. Each one is named `AAA_vs_BBB.png`, where `AAA` and `BBB` are the run IDs
  of the two models. Each plot shows the Dice scores on the test set for the models.
  * For both segmentation and classification models an IPython Notebook `report.ipynb` will be generated in the
   `outputs` directory.
    * For segmentation models, this report is based on the full image results of the model checkpoint that performed 
    the best on the validation set. This report will contain detailed metrics per structure, and outliers to help 
    model development.
    * For classification models, the report is based on the validation and test results from the last epoch. It shows
     metrics on the validation and test sets, ROC and PR Curves, and a list of the best and worst performing images
     from the test set.
    
Ensemble models are created by the zero'th child (with `cross_validation_split_index=0`) in each
cross-validation run. Results from inference on the test and validation sets are uploaded to the
parent run, and can be found in `epoch_NNN` directories as above.
In addition, various scores and plots from the ensemble and from individual child 
runs are uploaded to the parent run, in the `CrossValResults` directory. This contains:
* Subdirectories named 0, 1, 2, ... for all the child runs including the zero'th one, as well
 as `ENSEMBLE`, containing their respective `epoch_NNN` directories.
* Files `Dice_Test_Splits.png` and `Dice_Val_Splits.png`, containing box plots of the Dice scores
  on those datasets for each structure and each (component and ensemble) model. These give a visual
  overview of the results in the `metrics.csv` files detailed above. When there are many different
  structures, several such plots are created, with a different subset of structures in each one.
* Similarly, `HausdorffDistance_mm_Test_splits.png` and `HausdorffDistance_mm_Val_splits.png` contain
  box plots of Hausdorff distances.
* `MetricsAcrossAllRuns.csv` combines the data from all the `metrics.csv` files.
* `Test_outliers.txt` and `Val_outliers.txt` highlight particular outlier scores (both Dice and
  Hausdorff) in the test and validation sets respectively.
* A `scatterplots` directory and a file `CrossValidationWilcoxonSignedRankTestResults.txt`,
  for comparisons between the ensemble and its component models.

There is also a directory `BaselineComparisons`, containing the Wilcoxon test results and
scatterplots for the ensemble, as described above for single runs.

### Augmentations for classification models.

For classification models, you can define an augmentation pipeline to apply to your images input (resp. segmentations) at 
training, validation and test time. In order to define such a series of transformations, you will need to overload the 
`get_image_transform`  (resp. `get_segmention_transform`) method of your config class. This method expects you to return 
a `ModelTransformsPerExecutionMode`, that maps each execution mode to one transform function. We also provide the 
`ImageTransformationPipeline` a class that creates a pipeline of transforms, from a list of individual transforms and 
ensures the correct conversion of 2D or 3D PIL.Image or tensor inputs to the obtained pipeline.

`ImageTransformationPipeline` takes two arguments for its constructor:
 * `transforms`: a list of image transforms, in particular you can feed in standard [torchvision transforms](https://pytorch.org/vision/0.8/transforms.html) or
any other transforms as long as they support an input `[Z, C, H, W]` (where Z is the 3rd dimension (1 for 2D images), 
   C number of channels, H and W the height and width of each 2D slide - this is supported for standard torchvision 
   transforms.). You can also define your own transforms as long as they expect such a `[Z, C, H, W]` input. You can
   find some examples of custom transforms class in `InnerEye/ML/augmentation/image_transforms.py`.
* `use_different_transformation_per_channel`: if True, apply a different version of the augmentation pipeline
        for each channel. If False, applies the same transformation to each channel, separately. Default to False.
  
Below you can find an example of `get_image_transform` that would resize your input images to 256 x 256, and at
training time only apply random rotation of +/- 10 degrees, and apply some brightness distortion, 
using standard pytorch vision transforms.

```python
def get_image_transform(self) -> ModelTransformsPerExecutionMode:
    """
    Get transforms to perform on image samples for each model execution mode.
    """
    return ModelTransformsPerExecutionMode(
        train=ImageTransformationPipeline(transforms=[Resize(256), RandomAffine(degrees=10), ColorJitter(brightness=0.2)]),
        val=ImageTransformationPipeline(transforms=[Resize(256)]),
        test=ImageTransformationPipeline(transforms=[Resize(256)]))
```