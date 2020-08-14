# Building Models

### Setting up training

To train new models, you can either work within the InnerEye/ directory hierarchy or create a local hierarchy beside it
and with the same internal organization (although with far fewer files).
We recommend the latter as it offers more flexibility and better separation of concerns. Here we will assume you
create a directory `InnerEyeLocal` beside `InnerEye`.

As well as your configurations (dealt with below) you will need these files:
* `train_variables.yml`: A file similar to `InnerEye\train_variables.yml` containing all your Azure settings.
The value of `inference_code_directory` should (in our example) be `'InnerEyeLocal'`, 
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
               yaml_config_file=project_root / "relative/path/to/train_variables.yml",
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
        super().__init__(azure_dataset_id="id-of-your-blob-containing-prostate-data")
```
The allowed parameters and their meanings are defined in [`SegmentationModelBase`](/InnerEye/ML/config.py).
The class name must be the same as the basename of the file containing it, so `Prostate.py` must contain `Prostate`. 
In `train_variables.yml`, set `model_configs_namespace` to `InnerEyeLocal.ML.configs` so this config  
is found by the runner.

### Training a new model

* Set up your model configuration as above.

* Train a new model, for example `Prostate`:
```shell script
python InnerEyeLocal/ML/runner.py --submit_to_azureml=True --model=Prostate --is_train=True
```

Alternatively, you can train the model on your current machine if it is powerful enough. In
this case, you should specify `--submit_to_azureml=False`, and instead of specifying
`azure_dataset_id` in the class constructor, you can instead use `local_dataset="my/data/folder"`,
where the folder `my/data/folder` contains a `dataset.csv` file and subfolders `0`, `1`, `2`, ...,
one for each image.

### K-Fold Model Cross Validation

As for training a new model, but add the switch `--number_of_cross_validation_splits=N`, for some `N` greater than
1; a value of 5 is typical. This will trigger a HyperDrive run with each child run training on a fold from the 
union of the Training and Validation sets. The Test set is unchanged.

### Recovering and continuing training

To train further with an already-created model, give the above command with additional switches like these:
```
--run_recovery_id=foo_bar:foo_bar_12345_abcd --start_epoch=120
```
The run recovery ID is of the form "experiment_id:run_id". When you trained your original model, it will have been
queued as a "Run" inside of an "Experiment". The experiment will be given a name derived from the branch name - for 
example, branch `foo/bar` will queue a run in experiment `foo_bar`. Inside the "Tags" section of your run, you should
see an element `run_recovery_id`. It will look something like `foo_bar:foo_bar_12345_abcd`.

### Testing an existing model

As for continuing training, but set `--is_train` to `False`. Thus your command should look like this:
```shell script
python Inner/ML/runner.py --submit_to_azureml=True --model=Prostate --is_train=False --gpu_cluster_name=my_cluster_name \
   --run_recovery_id=foo_bar:foo_bar_12345_abcd --start_epoch=120
```

Alternatively, to submit an AzureML run to apply a model to a single image on your local disc, 
you can use the script `submit_for_inference.py`, with a command of this form:
```shell script
python InnerEye/Scripts/submit_for_inference.py --image_file ~/somewhere/ct.nii.gz --model_id Prostate:555 \
  --yaml_file ../somewhere_else/train_variables.yml --download_folder ~/my_existing_folder
```

### Model Ensembles

An ensemble model should be created automatically and registered in the AzureML model registry whenever cross-validation
models are trained. The ensemble model
creation is done by the child whose `cross_validation_split_index` is 0; you can identify this child by looking
at the "Child Runs" tab in the parent run page in AzureML. To find the ID of the ensemble model, look in the
driver log for the child run and search for the string "Registered model". There should be exactly two occurrences of
this string. The first is for the child model itself (each child run in fact registers one of these) and the
second is for the ensemble. 

As well as registering the model, the child run runs it on the validation and test sets. The results are aggregated 
based on the `ensemble_aggregation_type` value in the model config,
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
across the splits and with respect to other runs that you specify. Details of the tests can be found
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
  * `train_stats.csv`, containing summary statistics for each training epoch (learning rate, losses and
  Dice scores).

Ensemble models are created by the zero'th child (with `cross_validation_split_index=0`) in each
cross-validation run. They have two additional directories:

* `OTHER_RUNS` contains a subdirectory for every _other_ child run (with positive indices) and one
named `ENSEMBLE`. The child-run subdirectories each contain a `checkpoints` directory with one or more
checkpoint files, downloaded from their respective runs. The `ENSEMBLE` subdirectory contains
`epoch_NNN` subdirectories with scores for the ensemble model on the validation and test set.
* `CrossValResults` (which is also uploaded to the outputs/ directory of the parent run) 
contains the following:
  * Subdirectories named 0, 1, 2, ... for all the child runs including the zero'th one, as well
 as `ENSEMBLE`, containing their respective `epoch_NNN` directories.
  * A subdirectory named `scatterplots`, containing a `jpg` file for every pairing of component models
    and the ensemble model. Each one is named `AAA_vs_BBB.jpg`, where `AAA` and `BBB` are the run IDs
    of the two models. Each plot shows the Dice scores on the test set for the models.
    *TODO*: also has all-against-all for ensemble and each child model. Check for baseline
    comparisons when big models done.
  * A file `CrossValidationWilcoxonSignedRankTestResults.txt`, comprising the results of the Wilcoxon
    signed rank test applied to those Dice scores. Each paragraph of that file compares two models and
    indicates, for each structure, when the Dice scores for the second model are significantly better 
    or worse than the first. For full details, see the 
    [source code](../InnerEye/Common/Statistics/wilcoxon_signed_rank_test.py).
  * Files `Dice_Test_Splits.jpg` and `Dice_Val_Splits.jpg`, containing box plots of the Dice scores
    on those datasets for each structure and each (component and ensemble) model. These give a visual
    overview of the results in the `metrics.csv` files detailed above.
  * Similarly, `HausdorffDistance_mm_Test_splits.jpg` and `HausdorffDistance_mm_Val_splits.jpg` contain
    box plots of Hausdorff distances.
  * `MetricsAcrossAllRuns.csv` combines the data from all the `metrics.csv` files.
  * `Test_outliers.txt` and `Val_outliers.txt` highlight particular outlier scores (both Dice and
    Hausdorff) in the test and validation sets respectively.

### Using Tensorboard

You can use `InnerEye/Azure/monitor.py` to run a Tensorboard locally. All you need to do is call the script with the following
commandline arguments: `--run_ids job1,job2,job3`, where you provide the `run_recovery_id` of the run(s) you want to monitor.
Or you can run it with: `--experiment_name`, where you provide the name of the experiment to get all the runs in it.
You can also filter runs by type by the run's status, setting the `--filters Running,Completed` parameter to a subset of
`[Running, Completed, Failed, Canceled]`. By default Failed and Canceled runs are excluded.

To quickly access that, there is a template PyCharm run configuration `Template: Tensorboard monitoring` in the repository. Create
a copy of that, and modify the commandline arguments with your jobs to monitor.
