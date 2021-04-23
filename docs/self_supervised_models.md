# Training of self-supervised models

The code present in the SSL folder (link) allows you to train self-supervised models using
[SimCLR](http://proceedings.mlr.press/v119/chen20j/chen20j.pdf) or
[BYOL](https://proceedings.neurips.cc/paper/2020/file/f3ada80d5c4ee70142b17b8192b2958e-Paper.pdf). This code runs as a "bring-your-own-model" self-contained module (cf.
doc add-link-here).

Here, we provide implementations for four datasets to get you kickstarted with self-supervised models:

* Toy CIFAR datasets: [CIFAR10](https://www.cs.toronto.edu/~kriz/cifar.html)
  and [CIFAR100](https://www.cs.toronto.edu/~kriz/cifar.html)
* Medical Chest-Xray
  datasets: [RSNA Pneumonia Detection Challenge](https://www.kaggle.com/c/rsna-pneumonia-detection-challenge/overview)
  data (30k scans, labels indicating presence of pneumonia)
  and [NIH Chest-Xray](https://www.kaggle.com/nih-chest-xrays/data) (112k Chest-Xray scans).

### Multi-dataset support

During self-supervised training, a separate linear classifier is trained on top of learnt image embeddings. In this way,
users can continuously monitor the representativeness of learnt image embeddings for a given downstream classification
task. More importantly, the framework allows users to specify multiple datasets and data loaders for SimCLR/BYOL
training and evaluation. For instance, a BYOL encoder can be learnt using a dataset that does not contain any target
labels and embeddings can be evaluated throughout training on a separate dataset containing class labels. To enable this
functionality, our SSLContainer takes two dataset names parameters: ``ssl_training_dataset_name`` to indicate which
dataset to use for SSL training and ``classifier_dataset_name`` to indicate which dataset for the classification task
used to monitor embeddings quality during training.

## Quick start guide

Here we described how to quickly start a training job with our ready made configs.

### Example 1: training a SimCLR or BYOL model on CIFAR10 (TO UPDATE)

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
* To get the NIH dataset: please download from [here] https://www.kaggle.com/nih-chest-xrays/data. Make sure you include also the csv files in your data folder.

#### If you run on AML
In order to train models on AML you will need to upload and register the datasets listed above to your storage account and get their dataset_id to pass to your model config.

#### Step 1: Update your model config

We provide sample configs to train models on NIH data both for BYOL and SimCLR. You can use them as they are except for
the dataset location fields:

* If you're running locally set the `local_dataset` parameter to point to your data folder.
* If you're running on AML: you need to update the `RSNA_AZURE_DATASET_ID` and `NIH_AZURE_DATASET_ID` variables to point
  to the name of the NIH and Kaggle dataset in your own workspace.

#### Step 2: Launch the training job

Example to train a SSL model with BYOL on the NIH dataset and monitor the embeddings quality on the Kaggle RSNA Pneumonia Challenge classification task:

```
python ML/runner.py --model=NIH_RSNA_BYOL
```
## Configuring your own SSL models:

### About SSLContainer(add-link-here) configuration

All SSL models are derived from the SSLcontainer class. See the config class in [ML/configs/ssl](link) for some examples
of specific model configurations (all derived from this container).

If you wish to create your own model config for SSL training, you will need to create a child class and parametrize it with the following
available arguments:

* `ssl_training_dataset_name`: the name of the dataset to train the SSL encoder on, a member of the SSLDatasetName
  class (don't forget to update this class if you're adding a new dataset ;)),
* `classifier_dataset_name`: the name of the dataset to train to linear head on top of the classifier for monitoring
  purposes,
* `azure_dataset_id`: the id of the AML dataset to use for SSL training,
* `extra_azure_dataset_ids`: dataset_id to use for linear head training, expected to be provided as a list [data-id],
* `ssl_encoder`: name of the encoder to train, member of `EncoderName` class, currently supported are resnet50,
  resnet101 and densenet121,
* `ssl_training_type`: which SSL algorithm to use, member of `SSLType` choice between BYOL and SimCLR,
* `ssl_training_batch_size`: batch size of SSL training=1200,
* `ssl_training_path_augmentation_config`: path to yaml config for augmentation to use during SSL training. Only used
  for NIH/Kaggle datasets.
* `classifier_augmentations_path`: path to yaml config for augmentation to use for linear head training. Only used for
  NIH/Kaggle datasets,
* `use_balanced_binary_loss_for_linear_head`: whether to use balanced loss for linear head training,
* `random_seed`: seed for the run,
* `num_epochs`: number of epochs to train for.

### Creating your own datamodules:
To use this code with your own data, you will need to:
1.  Create a dataset class that reads your new dataset, derived from `VisionDataset`. See for example how we constructed `InnerEyeCXRDatasetBase`(link). 
2. Add a member to the `SSLDatasetName` Enum with your new dataset and update the `_SSLDataClassMappings` member of the class so that the code knows which data class to associate to your new dataset name. 
3. Update the `_get_transforms` methods to add the transform specific to your new dataset. To simplify this step, we have defined a series of standard transforms parametrized by an augmentation yaml file in `SSL/transforms_utils.py` (see next paragraph for more details). You could for example construct a transform pipeline similar to the one created with `get_cxr_ssl_transforms` for our CXR examples. 
4. Update all necessary parameters in the model config (cf. previous paragraph)

Once all these steps are updated, the code in the base SSLContainer class will take care of creating the corresponding datamodules for SSL training and linear head monitoring.

### About the augmentation configuration yaml file
The augmentations used for SSL training for all Chest-X-rays models are parametrized via a yaml config file. The path to
this config as to be passed in the model config (cf. section above). We provide two defaults
configs: ``rsna_aumgentations.yaml`` is used to define the augmentations used for BYOL/SimCLR training;
the ``linear_head.yaml`` config defines the augmentations to used for the training of the linear head (used for
monitoring purposes). The meaning of each config argument is detailed in `ssl_model_config.py`(add-link)

WARNING: this file will be ignored for CIFAR examples where we use the default pl-bolts augmentations.