# InnerEye-DataSelection

TODO: Maybe name the root folder as `InnerEye-DataQuality` instead of DataSelection to make it consistent. And make the associated renaming in the files.

## Contents of this sub-repository:

This folder contains all the source code associated to the manuscript ["Bernhardt et al.: Active label cleaning: Improving dataset quality under resource constraints"](https://arxiv.org/abs/2109.00574).

In particular, this folder provides the tools for:
1. Label noise robust training (e.g. co-teaching, ELR, self-supervised pretraining and finetuning capabilities)
2. The label cleaning simulation benchmark proposed in the above mentioned manuscript. 
3. The model selection benchmark.
4. All the code related to our benchmark datasets "CIFAR10H" and "NoisyChestXray". 


## Installation:

Cloning the InnerEye-DeepLearning repository to your local disk and move to the InnerEye-DataSelection folder.
```
git clone https://github.com/microsoft/InnerEye-DeepLearning
cd InnerEye-DeepLearning/InnerEye-DataSelection
```

Setting up the `InnerEyeDataQuality` python environment. Note that this repository uses a specific conda environment, independent from the `InnerEye` environment. 
```
python create_environment.py
conda activate InnerEyeDataQuality
pip install -e .
```

## Benchmark datasets:

### <ins>CIFAR10H</ins>
The CIFAR10H dataset consists of samples taken from the CIFAR10 test set but all the samples have been labelled by multiple annotators.
We use the CIFAR training set as our clean test set.

### <ins>Noisy Chest-Xray</ins>
The images released as part of the [Kaggle Challenge](https://www.kaggle.com/c/rsna-pneumonia-detection-challenge/), where originally released as part of the [NIH chest x-ray dataset](https://www.nih.gov/news-events/news-releases/nih-clinical-center-provides-one-largest-publicly-available-chest-x-ray-datasets-scientific-community). Before starting the competition, 30k images have been selected as the images for competitions. The labels for these images
have then been adjudicated to label them with bounding boxes indicating "pneumonia-life opacities". This dataset uses the kaggle dataset with noisy labels
as the original labels from RSNA and the clean labels are the Kaggle labels. Originally the dataset had 14 classes, we 
created a new binary label to label each image as "pneumonia-like" or "non-pneumonia-like" depending on the original label
prior to adjudication. The original (binarized) labels along with their corresponding adjudicated label, can be created with [create_noisy_chestxray_dataset.py](InnerEyeDataQuality/datasets/noisy_cxr_benchmark_creation/create_noisy_chestxray_dataset.py) (see "How to use it" section below). The dataset class for this dataset
is the [noisy_kaggle_cxr.py](InnerEyeDataQuality/datasets/noisy_kaggle_cxr.py) file. This dataset class will automatically load the noisy labels
from the aforementioned file (provided they have been created before hand, see next section)

#### Pre-requisites for using this dataset
1. The code will assume that the RSNA Pneumonia Challenge dataset is present on your machine. You will need to download it from the [Kaggle
page](https://www.kaggle.com/c/rsna-pneumonia-detection-challenge/data?select=stage_2_train_images) first to the `dataset_dir` of your choice.
2. You will need to create the noisy dataset csv file, that contains the noisy (derived from NIH dataset) labels and their clean counterpart (from the challenge data). 
In order to do so you will need to first download the following files: 
* Detailed RNSA Challenge Annotation [RSNA_pneumonia_all_probs.csv](https://storage.googleapis.com/kaggle-forum-message-attachments/844865/15542/RSNA_pneumonia_all_probs.csv.zip), and unzip it to a  csv
* RNSA to NIH Dataset mapping: [pneumonia-challenge-dataset-mappings_2018.json](https://s3.amazonaws.com/east1.public.rsna.org/AI/2018/pneumonia-challenge-dataset-mappings_2018.json)
* Csv files from the RNSA Pneumonia Detection Challenge: [stage_2_train_labels.csv and stage_2_detailed_class_info.csv](https://www.kaggle.com/c/rsna-pneumonia-detection-challenge/data)
* Place all these 4 files in the `ROOT / InnerEyeDataQuality / datasets / noisy_cxr_benchmark_creation` folder
* Run `python InnerEyeDataQuality / datasets / noisy_cxr_benchmark_creation / create_noisy_chestxray_dataset.py`. This script will save the new dataset csv to `ROOT / InnerEyeDataQuality / datasets / noisy_chestxray_dataset.csv` as expected by [noisy_kaggle_cxr.py](InnerEyeDataQuality/datasets/noisy_kaggle_cxr.py)
3. Update the `dataset_dir` field in the corresponding  model configs. 

### Other chest X-ray datasets
#### Full Kaggle Pneumonia Detection challenge dataset
For our experiments, in particular for unsupervised pretraining we use the full Kaggle training set (stage 1) from the
[Pneumonia Challenge](https://www.kaggle.com/c/rsna-pneumonia-detection-challenge). The dataset class for this dataset
can be found in the [kaggle_cxr.py](InnerEyeDataQuality/datasets/kaggle_cxr.py) file. This dataset class loads the full 
set with binary labels based on the bounding boxes provided for the competition.

##### Pre-requisites for using this dataset
1. The code will assume that the RSNA Pneumonia Challenge dataset is present on your machine. You will need to download it from the [Kaggle
page](https://www.kaggle.com/c/rsna-pneumonia-detection-challenge/data?select=stage_2_train_images) first to the `dataset_dir` of your choice.
2. Update the `dataset_dir` field in the corresponding  model configs. 


#### NIH Chest-Xray dataset
For the pretraining of our chest xrays self-supervised models we used the full [NIH Chest-Xray Dataset](https://www.nih.gov/news-events/news-releases/nih-clinical-center-provides-one-largest-publicly-available-chest-x-ray-datasets-scientific-community#:~:text=NIH%20Clinical%20Center%20provides%20one%20of%20the%20largest,disease.%20A%20chest%20x-ray%20identifies%20a%20lung%20mass.). 

##### Pre-requisites for using this dataset
1. The code will assume that the NIH ChestXray dataset is present on your machine. You will need to download the data from its dedicated [Kaggle
page](https://www.kaggle.com/nih-chest-xrays/data) to the `dataset_dir` of your choice.
2. Update the `dataset_dir` field in the corresponding model configs. 

## Noise Robust Learning

In this section, we provide details on how to train noise robust supervised models with our codebase. The code supports in particular Co-Teaching, Early Learning Regularization (ELR), finetuning of self-supervised (SSL) pretrained models. We also provide off-the-shelf configurations matching the experiment presented in our paper for the CIFAR10H and NoisyChestXray benchmark.

### Training noise robust supervised models
The main entry point for training a supervised model is [train.py](InnerEyeDataQuality/deep_learning/train.py). 
The code requires you to provide a config file specifying the dataset to use, the training specification (batch size, scheduler etc...),
which type of training, which augmentation to use ...

#### Launch the Python training command
Train the model using the command
```python
python InnerEyeDataQuality/deep_learning/train.py  --config <path to config>
```

#### Some important config arguments:
All possible config arguments are defined in [model_config.py](InnerEyeDataQuality/configs/models/model_config.py). Here you will find a summary of the most important config arguments:
* If you want to train a model with co_teaching, you will need to set `train.use_co_teaching: True` in your config.
* If you want to finetune from a pretrained SSL checkpoint:
    * You will need to set `train.use_self_supervision: True` to tell the code to load a pretrained checkpoint.
    * You will need update the `train.self_supervision.checkpoints: [PATH_TO_SSL]` field with the checkpoints to use for initialization of your model. Note that if you want to train a co-teaching model in combination with SSL pretrained initialization your list of checkpoint needs to be of length 2. 
    * You can also choose whether to freeze the encoder or not during finetuning with `train.self_supervision.freeze_encoder` field. 
* If you want to train a model using ELR, you can set `train.use_elr: True`

#### Off the shelf configurations

For each of the dataset used in our experiments, we have defined configs to run training easily off the shelf, with the same parameters as in our experiments.

##### CIFAR10H
We provide configurations to run experiments on CIFAR10H with resp. 15% and 30% noise rate. 
* Configs for 15% noise rate experiments can be found in [configs/models/cifar10h_noise_15](InnerEyeDataQuality/configs/models/cifar10h_noise_15). In detail this folder contains configs for
    * vanilla resnet training: [InnerEyeDataQuality/configs/models/cifar10h_noise_15/resnet.yaml](InnerEyeDataQuality/configs/models/cifar10h_noise_15/resnet.yaml)
    * co-teaching resnet training:  [InnerEyeDataQuality/configs/models/cifar10h_noise_15/resnet_co_teaching.yaml](InnerEyeDataQuality/configs/models/cifar10h_noise_15/resnet_co_teaching.yaml)
    * SSL + linear head training: [InnerEyeDataQuality/configs/models/cifar10h_noise_15/resnet_self_supervision_v3.yaml](InnerEyeDataQuality/configs/models/cifar10h_noise_15/resnet_self_supervision_v3.yaml)
* Configs for 30% noise rate experiments can be found in [configs/models/cifar10h_noise_30](InnerEyeDataQuality/configs/models/cifar10h_noise_30). In detail this folder contains configs for:
    * vanilla resnet training: [InnerEyeDataQuality/configs/models/cifar10h_noise_30/resnet.yaml](InnerEyeDataQuality/configs/models/cifar10h_noise_30/resnet.yaml)
    * co-teaching resnet training:  [InnerEyeDataQuality/configs/models/cifar10h_noise_30/resnet_co_teaching.yaml](InnerEyeDataQuality/configs/models/cifar10h_noise_30/resnet_co_teaching.yaml)
    * SSL + linear head training: [InnerEyeDataQuality/configs/models/cifar10h_noise_30/resnet_self_supervision_v3.yaml](InnerEyeDataQuality/configs/models/cifar10h_noise_30/resnet_self_supervision_v3.yaml)
    * ELR training: [InnerEyeDataQuality/configs/models/cifar10h_noise_30/resnet_elr.yaml](InnerEyeDataQuality/configs/models/cifar10h_noise_30/resnet_elr.yaml)
* Examples of configs for models used in the model selection benchmark experiment can be found in the [configs/models/benchmark_3_idn](InnerEyeDataQuality/configs/models/benchmark_3_idn)

##### Noisy Chest-Xray
*Note:* To run any model on this dataset, you will need to first make sure you have the dataset ready onto your machine (see Benchmark datasets > Noisy Chest-Xray > Pre-requisite section).

* With provide configurations corresponding to our experiments on the NoisyChestXray dataset with 13% noise rate in the [configs/models/cxr](InnerEyeDataQuality/configs/models/cxr) folder:
    * Vanilla resnet training: [InnerEyeDataQuality/configs/models/cxr/resnet.yaml](InnerEyeDataQuality/configs/models/cxr/resnet.yaml)
    * Co-teaching resnet training:  [InnerEyeDataQuality/configs/models/cxr/resnet_coteaching.yaml](InnerEyeDataQuality/configs/models/cxr/resnet_coteaching.yaml)
    * Co-teaching from a pretrained SSL checkpoint: [InnerEyeDataQuality/configs/models/cxr/resnet_ssl_coteaching.yaml]([InnerEyeDataQuality/configs/models/cxr/resnet_ssl_coteaching.yaml])
    
## Label cleaning simulation benchmark
To run the label cleaning simulation you will need to run [main_simulation](InnerEyeDataQuality/main_simulation.py) with
a list of selector configs in the `--config` arguments as well as a list of seeds to use for sampling in the `--seeds` arguments. A selector config will allow you to specify which
selector to use and which model config to use for inference. All selectors config can be found in the 
[configs/selection](InnerEyeDataQuality/configs/selection) folder (more details below).

For example if you run the following command, the benchmark will be run 3 times (with 3 different seeds) for each selector specified by its selector config (here we run it with 2 selectors). The resulting simulation results will be plotted on the same graph, with results aggregated per selector. The resulting graphs will then by default be found in `ROOT/logs/main_simulation_benchmark/TIME-STAMP`.
```
python InnerEyeDataQuality/main_simulation.py --config <path/config1> <path/config2> --seeds 1 2 3
```
This will by default clean the training set associated to your config. If you wish to clean the validation set instead, please add the `--on-val-set` flag to your command.

#### More details about the selector config
Here is an example of a selector config, with details about each field:

* `selector:`
  * `type`: Which selector to use. There are several options available:
    * `PosteriorBasedSelectorJoint`: Using the ranking function proposed in the manuscript CE(posteriors, labels) - H(posteriors)
    * `PosteriorBasedSelector`: Using CE(posteriors, labels) as the ranking function
    * `GraphBasedSelector`: Graph based selection of the next samples based on the embeddings of each sample.
    * `BaldSelector`: Selection of the next sample with the BALD objective.
  * `model_name`: The name that will be used for the legend of the simulation plot
  * `model_config_path`: Path to the config file used to train the selection model.
  * `use_active_relabelling`: Whether to turn on the active component of the active learning framework. If set to True, the model will be retrained regularly during the selection process.
  * `output_directory`: Optional field where can specify the output directory to store the results in. 


#### Off-the-shelf simulation configs
* We provide the configs for various selectors for our CIFAR10H experiments in the [configs/selection/cifar10h_noise_15](InnerEyeDataQuality/configs/selection/cifar10h_noise_15) and in the [configs/selection/cifar10h_noise_30](InnerEyeDataQuality/configs/selection/cifar10h_noise_30) folders. 
* And likewise for the NoisyChestXray dataset, you can find a set of selector configs in the [configs/selection/cxr](InnerEyeDataQuality/configs/selection/cxr) folder.

## Model selection benchmark
Likewise we provide the capability to run the model selection benchmark described in the paper. In more details, this script will first evaluate the model on the original noisy validation set. Then it will load the cleaned validation labels (coming from running one selector on the noisy validation dataset) and re-run evaluation on this cleaned dataset. The script will then report metrics for all models evaluated on both noisy and cleaned data. 

Running the benchmark: 
* [Pre-requisite] Prior to running the benchmark that are a few steps to do first:
    * You will need to first train one (or several) selector model using one model config of your choice and then run the corresponding cleaning simulation on the validation set to get cleaned labels for your validation set. 
        * Note: make sure you specify a specific `output_directory` in your selector config prior to running the cleaning simulation so that the model benchmark can retrive your cleaned labels.
    * You will also need to train the classifiers models you wish to compare in your benchmark. Note this model choice is independent from the model you chose to clean your data.
* Once you have completed the previous steps, you can run the benchmark with the following command:

```
python InnerEyeDataQuality/model_selection_benchmark.py --config <path-to-model-config1> <path-to-model-config2> --curated-label-config <path-to-selector-config-used-to-clean-your-data>
```

Examples model and selector configs for this benchmark for CIFAR10H dataset can be found in [configs/models/benchmark3_idn](InnerEyeDataQuality/models/benchmark3_idn).

## Self supervised pretraining for noise robust learning
In this subfolder we also provide the code to train self-supervised models using
[SimCLR](http://proceedings.mlr.press/v119/chen20j/chen20j.pdf) or
[BYOL](https://proceedings.neurips.cc/paper/2020/file/f3ada80d5c4ee70142b17b8192b2958e-Paper.pdf). 

### General
For the unsupervised training of our models, we rely on PyTorch Lightning and Pytorch Lightining bolts. The main entry point
for model training is [InnerEyeDataQuality/deep_learning/self_supervised/main.py](InnerEyeDataQuality/deep_learning/self_supervised/main.py).
You will also need to feed in a ssl model config file to specify which dataset to use etc.. All arguments available for the config are listed in [ssl_model_config.py](InnerEyeDataQuality/deep_learning/self_supervised/configs/ssl_model_config.py)

To launch a training job simply run:
```
python InnerEyeDataQuality/deep_learning/self_supervised/main.py --config path/to/ssl_config
```

We provide configs and data module to run the self-supervised training on CIFAR10H and on the NIH Chest Xray dataset.

### Predefined ssl model configs
#### CIFAR10H
To pretrain embeddings with contrastive learning on CIFAR10H you can use the 
[cifar10h_byol.yaml](InnerEyeDataQuality/deep_learning/self_supervised/configs/cifar10h_byol.yaml) or the [cifar10h_simclr.yaml](InnerEyeDataQuality/deep_learning/self_supervised/configs/cifar10h_simclr.yaml) config files. 

#### NIH Chest-Xray 
Provided that you have downloaded the dataset as explained in the Benchmark Datasets > Other Chest Xray Datasets > NIH Datasets > Pre-requisites section, you can easily launch a unsupervised pretraining run on the full NIH dataset using the [nih_byol.yaml](InnerEyeDataQuality/deep_learning/self_supervised/configs/nih_byol.yaml) or the [nih_simclr.yaml](InnerEyeDataQuality/deep_learning/self_supervised/configs/nih_simclr.yaml)
configs. Don't forget to update the `dataset_dir` field of your config to reflect your local path.




