# Move a model to other workspace

The InnerEye models on AzureML are composed of two parts in order to reproduce the same settings at inference and
training time:

- Model: The model is registered in the AzureML registry and contains the code used at training time and pytorch
  checkpoint
- Environment: The Azure ML environment used to train the model. This contains the docker image with all the
  dependencies that were used for training

If you want to export a model from one Workspace to another you can use the following command to download and upload a model
from an AzureML workspace. This script does not use settings.yml, it uses interactive authentication, and the workspace specified in the
parameters. The model will be written to the path in --path parameter with two folders one for the `MODEL` and one for the `ENVIRONMENT` files.

- Download to
  path: `python InnerEye/Scripts/move_model.py -a download --path ./ --workspace_name "<name>" --resource_group "<name>" --subscription_id "<sub_id>" --model_id "name:version"`

- Upload from
  path: `python InnerEye/Scripts/move_model.py - upload --path ./ --workspace_name "<name>" --resource_group "<name>" --subscription_id "<sub_id>" --model_id "name:version"`