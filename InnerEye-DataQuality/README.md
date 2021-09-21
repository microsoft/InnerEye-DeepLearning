# InnerEye-DataQuality

## Contents of this sub-repository:

This folder contains all the source code associated to the manuscript ["Bernhardt et al.: Active label cleaning: Improving dataset quality under resource constraints"](https://arxiv.org/abs/2109.00574).

In particular, this folder provides the tools for:
1. Label noise robust training (e.g. co-teaching, ELR, self-supervised pretraining and finetuning capabilities)
2. The label cleaning simulation benchmark proposed in the above mentioned manuscript. 
3. The model selection benchmark.
4. All the code related to proposed benchmark datasets "CIFAR10H" and "NoisyChestXray". 


## Installation:

Cloning the InnerEye-DeepLearning repository to your local disk and move to the InnerEye-DataQuality folder.
```
git clone https://github.com/microsoft/InnerEye-DeepLearning
cd InnerEye-DeepLearning/InnerEye-DataQuality
```

Setting up the `InnerEyeDataQuality` python environment. Note that this repository uses a specific conda environment, independent from the `InnerEye` environment. 
```
python create_environment.py
conda activate InnerEyeDataQuality
pip install -e .
```

## Benchmark datasets:

### <ins>CIFAR10H</ins>
The [CIFAR10H dataset](https://www.nature.com/articles/s41467-020-18946-z.pdf) consists of images taken from the original CIFAR10 test set, but all the images have been labelled by multiple annotators. We use the CIFAR10 training set as the clean test-set to evaluate our trained models.

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

### <ins>Chest X-ray datasets for model pre-training</ins>
#### Full Kaggle Pneumonia Detection challenge dataset:
In a subset of experiments, for unsupervised pretraining of chest xray models, the code uses the Kaggle training set (stage 1) from the
[Pneumonia Challenge](https://www.kaggle.com/c/rsna-pneumonia-detection-challenge). The dataset class for this dataset
can be found in the [kaggle_cxr.py](InnerEyeDataQuality/datasets/kaggle_cxr.py) file. This dataset class loads the full 
set with binary labels based on the bounding boxes provided for the competition.

1. The code will assume that the RSNA Pneumonia Challenge dataset is present on your machine. You will need to download it from the [Kaggle
page](https://www.kaggle.com/c/rsna-pneumonia-detection-challenge/data?select=stage_2_train_images) first to the `dataset_dir` of your choice.
2. Update the `dataset_dir` field in the corresponding  model configs. 

#### NIH Chest-Xray dataset:
In a subset of experiments, for unsupervised pretraining of chest xray models, the code uses [NIH Chest-Xray Dataset](https://www.nih.gov/news-events/news-releases/nih-clinical-center-provides-one-largest-publicly-available-chest-x-ray-datasets-scientific-community#:~:text=NIH%20Clinical%20Center%20provides%20one%20of%20the%20largest,disease.%20A%20chest%20x-ray%20identifies%20a%20lung%20mass.). 

1. The code will assume that the NIH ChestXray dataset is present on your machine. You will need to download the data from its dedicated [Kaggle
page](https://www.kaggle.com/nih-chest-xrays/data) to the `dataset_dir` of your choice.
2. Update the `dataset_dir` field in the corresponding model configs.

## Noise Robust Learning

In this section, we provide details on how to train noise robust supervised models with this repository. The code supports in particular Co-Teaching, Early Learning Regularization (ELR), finetuning of self-supervised (SSL) pretrained models. We also provide off-the-shelf configurations matching the experiment presented in the paper for the CIFAR10H and NoisyChestXray benchmark.

### Training noise robust supervised models
The main entry point for training a supervised model is [train.py](InnerEyeDataQuality/deep_learning/train.py). 
The code requires you to provide a config file specifying the dataset to use, the training specification (batch size, scheduler etc...),
which type of training, which augmentation to use. To launch a training job use the following command:
```python
python InnerEyeDataQuality/deep_learning/train.py  --config <path to config>
```

Please check the [Readme](InnerEyeDataQuality/configs/README.md) file to learn more about how to configure model training experiments.

## Label cleaning simulation benchmark
To run the label cleaning simulation you will need to run [main_simulation](InnerEyeDataQuality/main_simulation.py) with
a list of selector configs in the `--config` arguments as well as a list of seeds to use for sampling in the `--seeds` arguments. A selector config will allow you to specify which
selector to use and which model config to use for inference. All selectors config can be found in the 
[configs/selection](InnerEyeDataQuality/configs/selection) folder (more details below).

For example if you run the following command, the benchmark will be run 3 times (with 3 different seeds) for each selector specified by its selector config (here we run it with 2 selectors). The resulting simulation results will be plotted on the same graph, with results aggregated per selector. The resulting graphs will then by default be found in `ROOT/logs/main_simulation_benchmark/TIME-STAMP`.
```
python InnerEyeDataQuality/main_simulation.py --config <path/config1> <path/config2> --seeds 1 2 3
```
This will by default clean the training set associated to your config. If you wish to clean the validation set instead, please add the `--on-val-set` flag to your command. Please check the [Readme](InnerEyeDataQuality/configs/README.md) file to learn more about how to configure the simulation benchmark for label cleaning.

## Model selection benchmark
The repository also provides the capability to run the model selection benchmark described in the paper. In particular, the [script](InnerEyeDataQuality/model_selection_benchmark.py) will first evaluate the model on the original noisy validation set. Then it will load the cleaned validation labels (coming from running one selector on the noisy validation dataset) and re-run evaluation on this cleaned dataset. The script will then report metrics for all models evaluated on both noisy and cleaned data. 

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
In this subfolder you will find the source code to pre-train models using
[SimCLR](http://proceedings.mlr.press/v119/chen20j/chen20j.pdf) or
[BYOL](https://proceedings.neurips.cc/paper/2020/file/f3ada80d5c4ee70142b17b8192b2958e-Paper.pdf) self-supervision methods.

### General
For the unsupervised training of the models, we rely on PyTorch Lightning and Pytorch Lightining bolts. The main entry point
for model training is [InnerEyeDataQuality/deep_learning/self_supervised/main.py](InnerEyeDataQuality/deep_learning/self_supervised/main.py).
You will also need to feed in a ssl model config file to specify which dataset to use etc.. All arguments available for the config are listed in [ssl_model_config.py](InnerEyeDataQuality/deep_learning/self_supervised/configs/ssl_model_config.py)

To launch a training job simply run:
```
python InnerEyeDataQuality/deep_learning/self_supervised/main.py --config path/to/ssl_config
```
Please check the [Readme](InnerEyeDataQuality/configs/README.md) file to learn more about how to configure the self-supervised (SSL) training.





