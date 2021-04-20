# Move a model to other workspace

The InnerEye models on AzureML are composed of two parts in order to reproduce the same settings at inference and
training time:

- Model: The model is registed in the AzureML registry and contains the code used at training time and pytorch
  checkpoint
- Environment: The Azure ML environment used to train the model. This contains the docker image with all the
  dependencies that were used for training

If you want to export a model from one Workspace to another you can use:

- Export to
  disk: `python InnerEye/Scripts/move_model.py --action "export" --path ./ --workspace_name "<name>" --resource_group "<name>" --subscription_id "<sub_id>" --model_id "name:version"`

- Import from
  disk: `python InnerEye/Scripts/move_model.py --action "import" --path ./ --workspace_name "<name>" --resource_group "<name>" --subscription_id "<sub_id>" --model_id "name:version"`