# Model Deployment

Note: At present, only segmentation model have a clearly defined deployment path. If you have a deployment
path for classification model that you would like to see supported, please file an issue on Github.

After training of a segmentation model, all code that was used in training, plus the last checkpoint of the model,
are packaged up into a folder and then registered in AzureML for later use. This package of files is then available
in AzureML in the "Models" section of the workspace, see also the 
[documentation](https://docs.microsoft.com/en-us/python/api/azureml-core/azureml.core.model.model?view=azure-ml-py). 
Each model has a name, a numeric version, and apart from the core files also contains tags and properties.

When training a segmentation model with the InnerEye toolbox, you can see all files that make up the model in the
`outputs/final_model` folder, acessible via the UI in the "Outputs + logs" tab. All those files will be packaged
up into an AzureML model that has the same name as the InnerEye model you are training. The model name and version
will be written to a tag of the training run: In the AzureML run overview page, you can see a tag called `model_id`,
with a value that looks like `Prostate:892` if the `Prostate` model in version 892 was registered in this run.

In AzureML, navigate to the "Models" section, then locate the model that has just been registered. In the "Artifacts"
tab, you can inspect the files that have been registered. This will have a structure like this:
```
final_model/
├──score.py
├──environment.yml
├──model_inference_config.json
├──InnerEye/
|  ├── Azure/
|  ├── Common/
|  ├── ML/
├──checkpoints/
|  ├── 1_checkpoint.pth.tar
├──YourCode/
```

- `score.py`: This is a Python script that can execute the trained model. Check `score.py` in the repository root for 
a description of its commandline arguments.
- `environment.yml`: A Conda environment file that contains the packages required to run the code. If you are using
InnerEye as a submodule and have added a second Conda file, the file in the registered model will be the merged
version of those 2 Conda files.
- `model_inference_config.json`: Contains the names of anatomical structures that the model produces, and the paths
to all PyTorch checkpoint files. This file is used by `score.py`
- `InnerEye/`: A full copy of the InnerEye toolbox at the time of training.
- `YourCode/`: Optional. This is only present if you are using InnerEye as a submodule, and have specified that you
have a second folder with code that you would like to deploy alongside the InnerEye submodule (the 
`extra_code_directory` commandline argument).
- `checkpoints/`: A folder with 1 or more PyTorch checkpoint files. Multiple checkpoint files are only present if
the model comes out of an ensemble training run.
