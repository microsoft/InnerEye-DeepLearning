# Move a model to other workspace

The InnerEye models on AzureML are composed of two parts in order to reproduce the same settings at inference and
training time:

- Model: The model is registered in the AzureML registry and contains the code used at training time and pytorch
  checkpoint
- Environment: The Azure ML environment used to train the model. This contains the docker image with all the 
  dependencies that were used for training

If you want to export a model from one Workspace to another you can use the following command to download and upload a
model from an AzureML workspace. This script does not use settings.yml, it uses interactive authentication, and the
workspace specified in the parameters.

- Download to path:

`python InnerEye/Scripts/move_model.py -a download --path ./ --workspace_name "<name>" --resource_group "<name>" --subscription_id "<sub_id>" --model_id "name:version"`

 The model will be downloaded to a subfolder called `name_version` (based on the `--model_id` parameter) of the path in `--path` parameter. For example, with `--model_id "my_model:100` this will be `my_model_100`. The subfolder will contain two folders: one for the `MODEL` and one for the `ENVIRONMENT` files.

- Upload from path:

`python InnerEye/Scripts/move_model.py -a upload --path ./ --workspace_name "<name>" --resource_group "<name>" --subscription_id "<sub_id>" --model_id "name:version"`

The model is expected to be in the format above, i.e. contained in a subfolder called `name_version` of the path in `--path` parameter. The model will be automatically given the next available `id` once uploaded to the workspace. For example, if `my_model:100` already exists then it will be called `my_model:101`.

Once in place you may want to run inference on the model with partial test data, i.e. test data from patients for whom
some of the labels are missing. Normally inference on partial test data would raise an exception. To allow inference to continue over partial test data add the flag `--allow_incomplete_labels` to your inference call, for example `python InnerEye/ML/runner.py --allow_incomplete_labels --train=False --azureml --model=<model_name> --run_recovery_id=<recovery_id>`