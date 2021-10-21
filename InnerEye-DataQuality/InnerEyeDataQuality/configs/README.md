## Config arguments for model training:
All possible config arguments are defined in [model_config.py](InnerEyeDataQuality/configs/models/model_config.py). Here you will find a summary of the most important config arguments:
* If you want to train a model with co_teaching, you will need to set `train.use_co_teaching: True` in your config.
* If you want to finetune from a pretrained SSL checkpoint:
    * You will need to set `train.use_self_supervision: True` to tell the code to load a pretrained checkpoint.
    * You will need update the `train.self_supervision.checkpoints: [PATH_TO_SSL]` field with the checkpoints to use for initialization of your model. Note that if you want to train a co-teaching model in combination with SSL pretrained initialization your list of checkpoint needs to be of length 2. 
    * You can also choose whether to freeze the encoder or not during finetuning with `train.self_supervision.freeze_encoder` field. 
* If you want to train a model using ELR, you can set `train.use_elr: True`

### CIFAR10H
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

### Noisy Chest-Xray
*Note:* To run any model on this dataset, you will need to first make sure you have the dataset ready onto your machine (see Benchmark datasets > Noisy Chest-Xray > Pre-requisite section).

* With provide configurations corresponding to the experiments on the NoisyChestXray dataset with 13% noise rate in the [configs/models/cxr](InnerEyeDataQuality/configs/models/cxr) folder:
    * Vanilla resnet training: [InnerEyeDataQuality/configs/models/cxr/resnet.yaml](InnerEyeDataQuality/configs/models/cxr/resnet.yaml)
    * Co-teaching resnet training:  [InnerEyeDataQuality/configs/models/cxr/resnet_coteaching.yaml](InnerEyeDataQuality/configs/models/cxr/resnet_coteaching.yaml)
    * Co-teaching from a pretrained SSL checkpoint: [InnerEyeDataQuality/configs/models/cxr/resnet_ssl_coteaching.yaml]([InnerEyeDataQuality/configs/models/cxr/resnet_ssl_coteaching.yaml])
<br/><br/>

## Config arguments for label cleaning simulation:

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
* We provide the configs for various selectors for the CIFAR10H experiments in the [configs/selection/cifar10h_noise_15](InnerEyeDataQuality/configs/selection/cifar10h_noise_15) and in the [configs/selection/cifar10h_noise_30](InnerEyeDataQuality/configs/selection/cifar10h_noise_30) folders. 
* And likewise for the NoisyChestXray dataset, you can find a set of selector configs in the [configs/selection/cxr](InnerEyeDataQuality/configs/selection/cxr) folder.
<br/><br/>

## Configs for self-supervised model training:

CIFAR10H: To pretrain embeddings with contrastive learning on CIFAR10H you can use the 
[cifar10h_byol.yaml](InnerEyeDataQuality/deep_learning/self_supervised/configs/cifar10h_byol.yaml) or the [cifar10h_simclr.yaml](InnerEyeDataQuality/deep_learning/self_supervised/configs/cifar10h_simclr.yaml) config files. 

Chest X-ray: Provided that you have downloaded the dataset as explained in the Benchmark Datasets > Other Chest Xray Datasets > NIH Datasets > Pre-requisites section, you can easily launch a unsupervised pretraining run on the full NIH dataset using the [nih_byol.yaml](InnerEyeDataQuality/deep_learning/self_supervised/configs/nih_byol.yaml) or the [nih_simclr.yaml](InnerEyeDataQuality/deep_learning/self_supervised/configs/nih_simclr.yaml)
configs. Don't forget to update the `dataset_dir` field of your config to reflect your local path.
