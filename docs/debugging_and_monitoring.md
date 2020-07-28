# Debugging and Monitoring Jobs

### Debugging setup on local machine

For full debugging of any non-trivial model, you will need a GPU. Some basic debugging can also be carried out on
standard Linux or Windows machines.

There are two main entry points into the code:

* [`InnerEye/Azure/azure_runner.py`](/InnerEye/Azure/azure_runner.py) is triggered via the training build definition,
[`azure-pipelines/train.yaml`](/azure-pipelines/train.yaml), and can also be run from the command line. 
This queues a run in AzureML.
* The run itself executes [`InnerEye/ML/runner.py`](/InnerEye/ML/runner.py).

For both runner scripts, you need to provide a list of arguments and secrets. To simplify debugging, these are pulled
automatically from [`Inner/train_variables.yml`](/InnerEye/train_variables.yml) and from a local secrets file. 
When running on a Windows machine, the secrets are expected in `c:\temp\InnerEyeTestVariables.txt`. On Linux, they 
should be in `~/InnerEyeTestVariables.txt`. The secrets file is expected to contain at least a line of the form
```
APPLICATION_KEY=<app key for your AML workspace>
```

For developing and running your own models, you will probably find it convenient to create your own variants of
`runner.py` and `train_variables.yml`, as detailed in the page on [model building](building_models.md).

To quickly access both runner scripts for local debugging, we created template PyCharm run configurations, called
"Template: Azure runner" and "Template: ML runner". If you want to execute the runners on your machine, then
create a copy of the template run configuration, and change the arguments to suit your needs.

### Shorten training run time for debugging

Here are a few hints how you can reduce the complexity of training if you need to debug an issue. In most cases,
you should then be able to rely on a CPU machine.
* Reduce the number of feature channels in your model. If you run a UNet, for example, you can set 
`feature_channels = [1]` in your model definition file.
* Train only for a single epoch. You can set `--num_epochs=1` via the commandline or the `more_switches` variable
if you start your training via a build definition. This will only create a model checkpoint at epoch 1, and ignore
the values you have set for `test_save_epoch` and other related parameters.
* Restrict the dataset to a minimum, by setting `--restrict_subjects=1` on the commandline. This will cap all of
training, validation, and test set to at most 1 subject. To specify different numbers of training, validation
and test images, you can provide a comma-separated list, e.g. `--restrict_subjects=4,1,2`.

With the above settings, you should be able to get a model training run to complete on a CPU machine in a few minutes.


### Verify your changes using a simplified fast model

If you made any changes to the code that submits experiments (either `azure_runner.py` or `runner.py` or code
imported by those), validate them using a model training run in Azure. You can queue a model training run for the 
simplified `BasicModel2Epochs` model.
