# Training of self-supervised models

The code present in the [InnerEye/ML/SSL](https://github.com/microsoft/InnerEye-DeepLearning/tree/main/InnerEye/ML/SSL)
folder allows you to train self-supervised models using
[SimCLR](http://proceedings.mlr.press/v119/chen20j/chen20j.pdf) or
[BYOL](https://proceedings.neurips.cc/paper/2020/file/f3ada80d5c4ee70142b17b8192b2958e-Paper.pdf). This code runs as a "
bring-your-own-model" self-contained module (
see [docs/bring_your_own_model.md](https://github.com/microsoft/InnerEye-DeepLearning/blob/main/docs/bring_your_own_model.md))
.

Here, we provide implementations for four datasets to get you kickstarted with self-supervised models:

* Toy CIFAR datasets: [CIFAR10](https://www.cs.toronto.edu/~kriz/cifar.html)
  and [CIFAR100](https://www.cs.toronto.edu/~kriz/cifar.html)
* Medical Chest-Xray
  datasets: [RSNA Pneumonia Detection Challenge](https://www.kaggle.com/c/rsna-pneumonia-detection-challenge/overview)
  data (30k scans, labels indicating presence of pneumonia),
  [NIH Chest-Xray](https://www.kaggle.com/nih-chest-xrays/data) (112k Chest-Xray scans) or
  [CheXpert](https://stanfordmlgroup.github.io/competitions/chexpert/) (228k scans).

### Multi-dataset support

During self-supervised training, a separate linear classifier is trained on top of learnt image embeddings. In this way,
users can continuously monitor the representativeness of learnt image embeddings for a given downstream classification
task. More importantly, the framework allows users to specify multiple datasets and data loaders for SimCLR/BYOL
training and evaluation. For instance, a BYOL encoder can be learnt using a dataset that does not contain any target
labels and embeddings can be evaluated throughout training on a separate dataset containing class labels. To enable this
functionality, our SSLContainer takes two dataset names parameters: ``ssl_training_dataset_name`` to indicate which
dataset to use for SSL training and ``linear_head_dataset_name`` to indicate which dataset for the classification task
used to monitor embeddings quality during training.

## Quick start guide

Here we described how to quickly start a training job with our ready made configs.

### Example 1: training a SimCLR or BYOL model on CIFAR10

To kick-off a training for a SimCLR and BYOL models on CIFAR10, simply run

```
python ML/runner.py --model=CIFAR10BYOL
python ML/runner.py --model=CIFAR10SimCLR
```

For this dataset, it will automatically take care of downloading the dataset to your machine prior to starting training.

### Example 2: training a BYOL model on Chest-Xray data

#### Step 0: Get the data

#### If you run on your local machine:

Prior to starting training a model on this dataset, you will need to download it from Kaggle to your machine:

* To use the RSNA Pneumonia Detection Challenge data: please download from
  [here](https://www.kaggle.com/c/rsna-pneumonia-detection-challenge/data?select=stage_2_train_images). Make sure to
  download all images and the `dataset.csv` file to your data folder. Please note that the labels are here merely used
  for monitoring purposes.
* To get the NIH dataset: please download from [here](https://www.kaggle.com/nih-chest-xrays/data). Make sure you
  include also the csv files in your data folder, the code assumes the dataset files and all the images lie in your data
  folder as downloaded from Kaggle. In particular, do not modify the original csv filenames (e.g. the code expects to
  find the `Data_Entry_2017.csv` file within the data directory).
* To get the CheXpert data: please download from [here](https://stanfordmlgroup.github.io/competitions/chexpert/), the
  code assumes the dataset files and all the images lie in your data folder as downloaded.

#### If you run on AML

In order to train models on AML you will need to upload the datasets listed above to your storage account and get their
dataset_id to pass to your model config.

#### Step 1: Update your model config

We provide sample configs to train models on NIH data both for BYOL and SimCLR. You can use them as they are except for
the dataset location fields:

* If you're running locally set the `local_dataset` parameter to point to your data folder.
* If you're running on AML: you need to update the `RSNA_AZURE_DATASET_ID` and `NIH_AZURE_DATASET_ID` variables to point
  to the name of the NIH and Kaggle dataset in your own workspace.

#### Step 2: Launch the training job

Example to train a SSL model with BYOL on the NIH dataset and monitor the embeddings quality on the Kaggle RSNA
Pneumonia Challenge classification task:

```
python ML/runner.py --model=NIH_RSNA_BYOL
```

## Configuring your own SSL models:

### About SSLContainer configuration

All SSL models are derived from
the [SSLcontainer](https://github.com/microsoft/InnerEye-DeepLearning/blob/main/InnerEye/ML/SSL/lightning_containers/ssl_container.py)
class. See the config class in [ML/configs/ssl](https://github.com/microsoft/InnerEye-DeepLearning/tree/main/InnerEye/ML/configs/ssl) for some examples of specific model configurations (all derived
from this container).

If you wish to create your own model config for SSL training, you will need to create a child class and parametrize it
with the following available arguments:

* `ssl_training_dataset_name`: the name of the dataset to train the SSL encoder on, a member of the SSLDatasetName
  class (don't forget to update this class if you're adding a new dataset ;)),
* `linear_head_dataset_name`: the name of the dataset to train to linear head on top of the classifier for monitoring
  purposes,
* `azure_dataset_id`: the id of the AML dataset to use for SSL training,
* `extra_azure_dataset_ids`: dataset_id to use for linear head training, expected to be provided as a list [data-id],
* `ssl_encoder`: name of the encoder to train, member of `EncoderName` class, currently supported are resnet50,
  resnet101 and densenet121,
* `ssl_training_type`: which SSL algorithm to use, member of `SSLType` choice between BYOL and SimCLR,
* `ssl_training_batch_size`: batch size of SSL training. This is the number of examples processed by a single GPU.
  Multiply this by the number of GPUs to get the effective batch size.
* `linear_head_batch_size`: batch size for linear head training (used for monitor of SSL embeddings quality). This is
  the number of examples processed by a single GPU. Multiply this by the number of GPUs to get the effective batch size.
* `ssl_augmentation_config`: path to yaml config for augmentation to use during SSL training. Only used for NIH/Kaggle
  datasets.
* `linear_head_augmentation_config`: path to yaml config for augmentation to use for linear head training. Only used for
  NIH/Kaggle datasets,
* `use_balanced_binary_loss_for_linear_head`: whether to use balanced loss for linear head training,
* `random_seed`: seed for the run,
* `num_epochs`: number of epochs to train for.

In case you wish to first test your model locally, here some optional arguments that can be useful:
* `local_dataset`: path to local dataset, if passed the azure dataset will be ignored
* `is_debug_model`: if True it will only run on the first batch of each epoch
* `drop_last`: if False (True by default) it will keep the last batch also if incomplete

### Creating your own datamodules:

To use this code with your own data, you will need to:

1. Define your own Lightening Container that inherits from `SSLContainer` as described in the paragraph above.
2. Create a dataset class that reads your new dataset, inheriting from both `VisionDataset`
   and `InnerEyeDataClassBaseWithReturnIndex`. See for example how we constructed `RSNAKaggleCXR`
   class. WARNING: the first positional argument of your dataset class constructor MUST be the data directory ("root"),
   as VisionDataModule expects this in the prepare_data step.
3. In your own container update the `_SSLDataClassMappings` member of the class so that the code knows which data class
   to associate to your new dataset name.
4. Create a yaml configuration file that contains the augmentations specific to your dataset. The yaml file will be
   consumed by the `create_transforms_from_config` function defined in the
   `InnerEye.ML.augmentations.transform_pipeline` module (see next paragraph for more details). Alternatively, overwrite
   the `_get_transforms` method. To simplify this step, we have defined a series of standard operations in
   `SSL/transforms_utils.py` . You could for example construct a transform pipeline similar to the one created
   inside `create_transform_from_config` inside your own method.
5. Update all necessary parameters in the model config (cf. previous paragraph)

Once all these steps are updated, the code in the base SSLContainer class will take care of creating the corresponding
datamodules for SSL training and linear head monitoring.

### About the augmentation configuration yaml file

The augmentations used for SSL training for all Chest-X-rays models are parametrized via a yaml config file. The path to
this config as to be passed in the model config (cf. section above). We provide two defaults
configs: ``cxr_ssl_encoder_augmentations.yaml`` is used to define the augmentations used for BYOL/SimCLR training;
the ``cxr_linear_head_augmentations.yaml`` config defines the augmentations to used for the training of the linear head
(used for monitoring purposes). The meaning of each config argument is detailed in `ssl_model_config.py`

WARNING: this file will be ignored for CIFAR examples where we use the default pl-bolts augmentations.

## Finetuning a linear head on top of a pretrained SSL model.

Alongside with the modules to train your SSL models, we also provide examplary modules that allow you to build a
classifier on top of a pretrained SSL model. The base class for these modules is `SSLClassifierContainer`. It builds on
top of the `SSLContainer` with additional command line arguments allowing you to specify where to find the checkpoint
for your pretrained model. For this you have two options:

- If you are running locally, you can provide the local path to your pretrained model checkpoint
  via `--local_weights_path`.
- If your are running on AML, use the `pretraining_run_recovery_id` field. Providing this field, will mean that AML will
  automatically download the checkpoints to the current node, will pick the latest checkpoint to build the classifier on
  top. Beware not to confuse `pretraining_run_recovery_id` with `run_recovery_id` as the latter is use to continue training on
  the same model (which is not the case here).

The code will then automatically extract the encoder part of the loaded SSL model to initialize your classifier. You can
then also choose whether you want to freeze the weights of your encoder or not via the `--freeze_encoder=True/False`
argument. By default, this is set to True.

### Example for CIFAR

We provide an example of such a classifier container for CIFAR named `SSLClassifierCIFAR`. To launch a finetuning run
for this model on CIFAR10, just run

```
python ML/runner.py --model=SSLClassifierCIFAR --pretraining_run_recovery_id={THE_ID_TO_YOUR_SSL_TRAINING_JOB}
```

### Example for CXR

Similarly, we provide class to allow you to simply start a finetuning job for CXR model in `CXRImageClassifier`. By
default, this will launch a finetuning job on the RSNA Pneumonia dataset. To start the run:

```
python ML/runner.py --model=CXRImageClassifier --pretraining_run_recovery_id={THE_ID_TO_YOUR_SSL_TRAINING_JOB}
```

or for a local run

```
python ML/runner.py --model=CXRImageClassifier --local_weights_path={LOCAL_PATH_TO_YOUR_SSL_CHECKPOINT}
```
