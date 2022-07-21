
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

To evaluate the model in Azure ML, you must first [resgister an Azure ML Model](https://docs.microsoft.com/en-us/python/api/azureml-core/azureml.core.model.model?view=azure-ml-py#remarks). To register the Hippocampus model in your AML Workspace, unpack the source code downloaded in the previous step and follow the instructions to run InnerEye's [move_model.py](https://github.com/microsoft/InnerEye-DeepLearning/blob/main/docs/move_model.md) script in the `-a=upload` direction.

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
    --model_id Hippocampus:111
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
