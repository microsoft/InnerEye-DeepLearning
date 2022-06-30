# Trained model for hippocampal segmentation
## Purpose
[This folder](TODO: ADD FOLDER) contains the saved checkpoints and associated code for a model to segment the left and right hippocampi from brain MRI scans.

## Terms of use
Please note that this model is intended for research purposes only. You are responsible for the performance, the necessary testing, and if needed any regulatory clearance for any of the models produced by this toolbox.

---

## Usage
### Create an Azure ML Dataset
To evaluate this model on your own data, you will first need to register an [Azure ML Dataset](https://docs.microsoft.com/en-us/azure/machine-learning/v1/how-to-create-register-datasets). You can follow the instructions in the InnerEye repo for [creating datasets](https://github.com/microsoft/InnerEye-DeepLearning/blob/main/docs/creating_dataset.md) in order to do this.

### Registering a model in Azure ML
To evaluate the model in Azure ML, you must first [resgister an Azure ML Model](https://docs.microsoft.com/en-us/python/api/azureml-core/azureml.core.model.model?view=azure-ml-py#remarks). To register the Hippocampus in your AML Workspace, follow the instructions to run InnerEye's [move_model.py](https://github.com/microsoft/InnerEye-DeepLearning/blob/main/docs/move_model.md) script in the `-a=upload` direction.

i.e.
```shell
python InnerEye/Scripts/move_model.py -a upload --path ./ --workspace_name <workspace_name> --resource_group <resource_group> --subscription_id <subscription_id> --model_id Hippocampus:111
```
### Evaluating the model
You can evaluate the model either in Azure ML (for this you will need to see the section below on how to register an Azure ML model), or locally using the downloaded checkpoint files. These 2 scenarios are described in more detail, along with instructions in  [testing an existing model](https://github.com/microsoft/InnerEye-DeepLearning/blob/main/docs/building_models.md#testing-an-existing-model).

E.g. to evalute the model on your own Dataset in Azure ML:
```shell
python InnerEye/ML/runner.py --azure_dataset_id=<my dataset id> --model=Hippocampus --model_id=Hippocampus:111 --experiment_name evaluate_hippocampus_model --azureml  --no-train
```

### Deploy with InnerEye Gateway
To deploy this model, see the instructions for [InnerEye-Gateway](https://github.com/microsoft/InnerEye-Gateway)
