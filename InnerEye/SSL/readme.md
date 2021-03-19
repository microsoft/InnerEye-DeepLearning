# Training of self-supervised models

The code present in this folder allows you to train self-supervised models using
[SimCLR](http://proceedings.mlr.press/v119/chen20j/chen20j.pdf) or
[BYOL](https://proceedings.neurips.cc/paper/2020/file/f3ada80d5c4ee70142b17b8192b2958e-Paper.pdf). 


We provide datamodules and configs examples for two datasets: [CIFAR10](https://www.cs.toronto.edu/~kriz/cifar.html) and [RSNA Pneumonia Detection Challenge](https://www.kaggle.com/c/rsna-pneumonia-detection-challenge/overview) to get you kickstarted with unsupervised models. To use this code with your own data, you simply need to adapt one of the existing [configs](configs) 
and create a PyTorch lightning datamodule for your dataset. Details about the meaning of each config arguments can be found 
in the [ssl_model_config.py](configs/ssl_model_config.py) file.


### Example 1: training a SimCLR model on CIFAR10
To kick-off a training for a SimCLR model on CIFAR10, simply run
```
python main.py --config configs/cifar10_simclr.yaml
```
This will automatically take care of downloading the dataset to your machine prior to starting training.


### Example 2: training a BYOL model on RSNA Pneumonia Detection Challenge

##### Step 1: Get the data
Prior to starting training a model on this dataset, you will need to download it from Kaggle to your machine:
* To use the RSNA Pneumonia Detection Challenge data: please download from 
  [here](https://www.kaggle.com/c/rsna-pneumonia-detection-challenge/data?select=stage_2_train_images). 
  Make sure to download all images and the `dataset.csv` file to your data folder. Please note that the labels are here 
  merely used for monitoring purposes. Indeed, additionally to train the self-supervised model, we fit a linear head on top
  of the embeddings during training to monitor the embeddings quality.
  
##### Step 2: Update the configs
We provide sample configs to train models on RSNA Pneumonia data both for BYOL and SimCLR. You can use them as they are except for 
the `dataset.dataset_dir` field that needs to be updated with your local path to your data folder. 
After this, you're ready to train!

Example to train a model on RSNA dataset with BYOL:
```
python main.py --config configs/rsna_byol.yaml
```

