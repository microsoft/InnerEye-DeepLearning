
# Trained model for hippocampal segmentation

## Purpose

This documentation describes how to use our pre-trained model to segment the left and right hippocampi from brain MRI scans. The model was trained on 998 MRI scan + segmentation pairs from the [ADNI](https://adni.loni.usc.edu/) dataset. This data is publicly available via their website, but users must sign a Data Use Agreement in order to gain access. We do not provide access to the data. The following description assumes the user has their own dataset to evaluate/ retrain the model on.

## Terms of use

Please note that this model is intended for research purposes only. You are responsible for the performance, the necessary testing, and if needed any regulatory clearance for any of the models produced by this toolbox.

---

## Usage

The following instructions assume you have completed the preceding setup steps in the [InnerEye README](https://github.com/microsoft/InnerEye-DeepLearning/), in particular, [Setting up Azure Machine Learning](https://github.com/microsoft/InnerEye-DeepLearning/blob/main/docs/setting_up_aml.md).

### Create an Azure ML Dataset

To evaluate this model on your own data, you will first need to register an [Azure ML Dataset](https://docs.microsoft.com/en-us/azure/machine-learning/v1/how-to-create-register-datasets). You can follow the instructions in the InnerEye repo for [creating datasets](https://github.com/microsoft/InnerEye-DeepLearning/blob/main/docs/creating_dataset.md) in order to do this.

## Downloading the model

The saved weights from the trained Hippocampus model can be downloaded along with the source code used to train it from [our github releases page](https://github.com/microsoft/hi-ml/releases).

### Registering a model in Azure ML

To evaluate the model in Azure ML, you must first [resgister an Azure ML Model](https://docs.microsoft.com/en-us/python/api/azureml-core/azureml.core.model.model?view=azure-ml-py#remarks). To register the Hippocampus model in your AML Workspace, unpack the source code downloaded in the previous step and follow InnerEye's [instructions to upload models to Azure ML](https://github.com/microsoft/InnerEye-DeepLearning/blob/main/docs/move_model.md).

```shell
WORKSPACE="fill with your workspace name"
GROUP="fill with your resource group name"
SUBSCRIPTION="fill with your subscription ID"

python InnerEye/Scripts/move_model.py \
    --action upload \
    --path . \
    --workspace_name $WORKSPACE \
    --resource_group $GROUP \
    --subscription_id $SUBSCRIPTION \
    --model_id Hippocampus:118
```

### Evaluating the model

You can evaluate the model either in Azure ML or locally using the downloaded checkpoint files. These 2 scenarios are described in more detail, along with instructions in [testing an existing model](https://github.com/microsoft/InnerEye-DeepLearning/blob/main/docs/building_models.md#testing-an-existing-model).

For example, to evalute the model on your Dataset in Azure ML:

```shell
DATASET_ID="fill with your dataset name"

python InnerEye/ML/runner.py \
    --azure_dataset_id $DATASET_ID \
    --model Hippocampus \
    --model_id Hippocampus:111 \
    --experiment_name evaluate_hippocampus_model \
    --azureml \
    --no-train
```

### Deploy with InnerEye Gateway

To deploy this model, see the instructions in the [InnerEye README](https://github.com/microsoft/InnerEye-DeepLearning/).

---

# Hippocampal Segmentation Model Card

## Model details

- Organiation: Biomedical Imaging Team at Micrsoft Research, Cambridge UK.
- Model date: 5th July 2022.
- Model version: 0.1 . Currently only one version of the model has been released. We do not expect to update the model frequently.
- Model type: 3d UNET ensemble of 3d UNet. Training details are as described in [this paper](https://jamanetwork.com/journals/jamanetworkopen/fullarticle/2773292).
- Training details: 5 fold ensemble model. Trained on a subsection of the ADNI dataset (described in detail below)
- License: The model is released under MIT license as described [here](https://github.com/microsoft/InnerEye-DeepLearning/blob/main/LICENSE).
- Contect: innereyedev@microsoft.com.

## Limitations

This model has been trained on a subset of the ADNI dataset. There have been various phases of ADNI spanning different time periods. In this Model Card we refer to the original, or ADNI 1, stud.y This dataset comprises scans and metadata from patients between the ages of 55-90 from 57 different sites across the US and Canada [source](https://adni.loni.usc.edu/study-design/#background-container). Therefore a major limitation of this model would be the ability to generalise to patients outside of this demographic. Another limitation is that  The MRI protocol for ADNI1 (which was collected between 2004-2009) focused on imaging on 1.5T scanners [source](https://adni.loni.usc.edu/methods/mri-tool/mri-analysis/). Modern scanners may have higher field strengths and therefore different levels of contrast which could lead to different performance from the results we report.

The results of this model have not been validated by clinical experts. We expect the user to evaluate the result

## Intended Uses

This model is for research purposes only. It is intended to be used for the task of segmentating hippocampi from brain MRI scans. Any other task is out of scope for this model.

## About the data

The model was trained on 998 pairs of MRI segmentation + segmentation. The model was further validated on 127 pairs of images and tested on 125 pairs. A further 317 pairs were retained as a held-out test set for the final evaluation of the model, which is what we report performance on.

All of this data comes from the Alzheimers Disease Neuroimaging Initiative study [link to website](https://adni.loni.usc.edu/). The data is publicly available, but requires signing a Data Use Agreement before access is granted.

## About the ground-truth segmentations

 The segmentations were also downloaded from the ADNI dataset. They were created semi-automatically using software from [Medtronic Surgical Navigation Technologies](https://www.medtronic.com/us-en/healthcare-professionals/products/neurological/surgical-navigation-systems.html). Further information is available on the [ADNI website](https://adni.loni.usc.edu/).

## Metrics

Note that due to the ADNI Data Usage Agreement we are only able to share aggregate-level metrics from our evaluation. Evaluation is performed on a heldout test set of 252 pairs of MRI + segmentation pairs from the ADNI dataset.

Dice Score for Left and Right hippocampus respectively on a held-out test set of :

![hippocampus_metrics_boxplot.png](hippocampus_metrics_boxplot.png)
| Structure     | count   | DiceNumeric_mean | DiceNumeric_std | DiceNumeric_min | DiceNumeric_max | HausdorffDistance_mm_mean | HausdorffDistance_mm_std | HausdorffDistance_mm_min | HausdorffDistance_mm_max | MeanDistance_mm_mean | MeanDistance_mm_std | MeanDistance_mm_min | MeanDistance_mm_max |
|---------------|---------|------------------|-----------------|-----------------|-----------------|---------------------------|--------------------------|--------------------------|--------------------------|----------------------|---------------------|---------------------|---------------------|
| hippocampus_L | 252     | 0.918            | 0.022           | 0.819           | 0.953           | 2.206                     | 0.812                    | 1.206                    | 6.964                    | 0.168                | 0.054               | 0.096               | 0.399               |
| hippocampus_R | 252     | 0.918            | 0.024           | 0.816           | 0.954           | 2.185                     | 0.706                    | 1.206                    | 4.966                    | 0.168                | 0.055               | 0.094               | 0.433               |
|               |         |                  |                 |                 |                 |                           |                          |                          |                          |                      |                     |                     |                     |
---
