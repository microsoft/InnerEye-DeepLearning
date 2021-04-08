# Training of self-supervised models

The code present in this folder allows you to train self-supervised models using
[SimCLR](http://proceedings.mlr.press/v119/chen20j/chen20j.pdf) or
[BYOL](https://proceedings.neurips.cc/paper/2020/file/f3ada80d5c4ee70142b17b8192b2958e-Paper.pdf). This code runs
outside of the traditional InnerEye codepath and is implemented as a "bring-your-own-model" self-contained module (cf.
doc add-link-here).

Here, we provide implementations for four datasets to get you kickstarted with self-supervised models:

* Toy CIFAR datasets: [CIFAR10](https://www.cs.toronto.edu/~kriz/cifar.html)
  and [CIFAR100](https://www.cs.toronto.edu/~kriz/cifar.html)
* Medical Chest-Xray
  datasets: [RSNA Pneumonia Detection Challenge](https://www.kaggle.com/c/rsna-pneumonia-detection-challenge/overview)
  data (30k scans, labels indicating presence of pneumonia)
  and [NIH Chest-Xray](https://www.kaggle.com/nih-chest-xrays/data) (112k Chest-Xray scans).

Note: To use this code with your own data, you will need to create a PyTorch lightning datamodule for your dataset,
preferably derived from "InnerEyeVisionDataModule".

### Multi-dataset Support

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
python main.py --config configs/cifar10_simclr.yaml
python main.py --config configs/cifar10_byol.yaml
```

If you're running this locally, it will automatically take care of downloading the dataset to your machine prior to
starting training.

### Example 2: training a BYOL model on Chest-Xray data

#### [Optional, only to run it locally] Step 0: Get the data

Prior to starting training a model on this dataset, you will need to download it from Kaggle to your machine:

* To use the RSNA Pneumonia Detection Challenge data: please download from
  [here](https://www.kaggle.com/c/rsna-pneumonia-detection-challenge/data?select=stage_2_train_images). Make sure to
  download all images and the `dataset.csv` file to your data folder. Please note that the labels are here merely used
  for monitoring purposes. Indeed, additionally to train the self-supervised model, we fit a linear head on top of the
  embeddings during training to monitor the embeddings quality.
* To get the NIH dataset TODO

#### Step 1: Update your model config (TO UPDATE)

We provide sample configs to train models on NIH data both for BYOL and SimCLR. You can use them as they are except for
the `local_dataset` field that needs to be updated with your local path to your data folder (if your running locally),
otherwise if running on AML you need to update the XXX and YYY variables to point to the name of the NIH and Kaggle
dataset in your workspace. After this, you're ready to train!

#### Step 3: Launch the training job (TO UPDATE)

Example to train a model on RSNA dataset with BYOL:

```
python main.py --config configs/rsna_byol.yaml
```

## About SSLContainer(add-link-here) model configuration

All SSL models are instances of the SSLcontainer class. See the config class in [ML/configs/ssl](link) for some examples
of specific model configurations (all derived from this container).

If you wish to create your own model config for SSL training, you will need to parametrize it with the following
available arguments:

* Make a list of arguments here

### About the augmentation configuration yaml file

The augmentations used for SSL training for all Chest-X-rays models are parametrized via a yaml config file. The path to
this config as to be passed in the model config (cf. section above). We provide two defaults
configs: ``rsna_aumgentations.yaml`` is used to define the augmentations used for BYOL/SimCLR training;
the ``linear_head.yaml`` config defines the augmentations to used for the training of the linear head (used for
monitoring purposes). WARNING: this file will be ignored for CIFAR examples where we use the default pl-bolts
augmentations.
