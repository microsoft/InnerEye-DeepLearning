# Debugging and Monitoring

## Monitoring in the AzureML Portal

The AzureML portal provides a powerful suite of tools for monitoring all aspects of your experiments including hardware analytics, training metrics and job outputs. InnerEye-DeepLearning is already configured to be fully compatible with all of these. To view this portal simply navigate to your [AzureML workspace](ml.azure.com) and select your experiment/run in the "Jobs" sub-menu.

### Training Outputs + Reports

Under the "Outputs + logs" tab you will find all the files output by your job:

- The arguments used for your job in `args.txt`.
- CSV files detailing your input dataset splits (`dataset.csv`, `train_dataset.csv`, `test_dataset.csv`, `val_dataset.csv`).
- In the `logs/` and `azureml-logs/` folders you can find all the log files output by your job.
  - The most important of these is the `azureml-logs/70_driver_log.txt` which contains information that is especially useful for debugging failed jobs.
- Under the `outputs/` folder you will find:
  - Each epoch's training metrics under `Train/` and `Val/` for training and validation respectively.
  - The most recent training checkpoint under `checkpoints/`.
  - Outputs from the epoch with the lowest validation loss under `best_validation_epoch/`.
  - The final report on the completed model under `reports/`. This is especially useful as it contains a full breakdown of a variety of metrics which are produced by a full inference pass after training is completed.
- For training tasks you will find a copy of the trained model (also registered to AzureML) in the `final_model/` folder (or `final_ensemble_model/` for ensemble models).

### Metrics

Under the "Metrics" tab you will be able to view all metrics logged by your job. This includes, but is not limited to:

- Train and validation loss.
- DICE scores for individual structures on segmentation tasks.
- Voxel/Pixel counts.
- Epoch number.

### Hardware Analytics

Under the "Monitoring" tab you will be able to view a range of hardware metrics. This includes, but is not limited to:

- GPU Utilisation.
- GPU Memory Usage.
- GPU Energy Usage.
- CPU Utilisation.

## Using TensorBoard to monitor AzureML jobs

* **Existing jobs**: execute [`InnerEye/Azure/tensorboard_monitor.py`](https://github.com/microsoft/InnerEye-DeepLearning/tree/main/InnerEye/Azure/tensorboard_monitor.py)
with either an experiment id `--experiment_name` or a list of run ids `--run_ids job1,job2,job3`.
If an experiment id is provided then all of the runs in that experiment will be monitored. Additionally You can also
filter runs by type by the run's status, setting the `--filters Running,Completed` parameter to a subset of
`[Running, Completed, Failed, Canceled]`. By default Failed and Canceled runs are excluded.

To quickly access this script from PyCharm, there is a template PyCharm run configuration
`Template: Tensorboard monitoring` in the repository. Create a copy of that, and modify the commandline
arguments with your jobs to monitor.

* **New jobs**: when queuing a new AzureML job, pass `--tensorboard`, which will automatically start a new TensorBoard
session, monitoring the newly queued job.

## Resource Monitor

GPU and CPU usage can be monitored throughout the execution of a run (local and AML) by setting the monitoring interval
for the resource monitor eg: `--monitoring_interval_seconds=5`. This will spawn a separate process at the start of the
run which will log both GPU and CPU utilization and memory consumption. These metrics will be written to AzureML as
well as a separate TensorBoard logs file under `Diagnostics`.

## Debugging setup on local machine

For full debugging of any non-trivial model, you will need a GPU. Some basic debugging can also be carried out on
standard Linux or Windows machines.

The main entry point into the code is [`InnerEye/ML/runner.py`](https://github.com/microsoft/InnerEye-DeepLearning/tree/main/InnerEye/ML/runner.py). The code takes its
configuration elements from commandline arguments and a settings file,
[`InnerEye/settings.yml`](https://github.com/microsoft/InnerEye-DeepLearning/tree/main/InnerEye/settings.yml).

A password for the (optional) Azure Service
Principal is read from `InnerEyeTestVariables.txt` in the repository root directory. The file
is expected to contain a line of the form

```text
APPLICATION_KEY=<app key for your AML workspace>
```

For developing and running your own models, you will probably find it convenient to create your own variants of
`runner.py` and `settings.yml`, as detailed in the page on [model building](building_models.md).

To quickly access both runner scripts for local debugging, we created template PyCharm run configurations, called
"Template: Azure runner" and "Template: ML runner". If you want to execute the runners on your machine, then
create a copy of the template run configuration, and change the arguments to suit your needs.

## Shorten training run time for debugging

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

## Verify your changes using a simplified fast model

If you made any changes to the code that submits experiments (either `azure_runner.py` or `runner.py` or code
imported by those), validate them using a model training run in Azure. You can queue a model training run for the
simplified `BasicModel2Epochs` model.

## Debugging on an AzureML node

It is sometimes possible to get a Python debugging (pdb) session on the main process for a model
training run on an  AzureML compute cluster, for example if a run produces unexpected output,
or is silent what seems like an unreasonably long time. For this to work, you will need to
have created the cluster with ssh access enabled; it is not currently possible to add this
after the cluster is created. The steps are as follows.

* From the "Details" tab in the run's page, note the Run ID, then click on the target name under
"Compute target".
* Click on the "Nodes" tab, and identify the node whose "Current run ID" is that of your run.
* Copy the connection string (starting "ssh") for that node, run it in a shell, and when prompted,
supply the password chosen when the cluster was created.
* Type "bash" for a nicer command shell (optional).
* Identify the main python process with a command such as

```shell
ps aux | grep 'python.*runner.py' | egrep -wv 'bash|grep'
```

You may need to vary this if it does not yield exactly one line of output.

* Note the process identifier (the value in the PID column, generally the second one).
* Issue the commands

```shell
kill -TRAP nnnn
nc 127.0.0.1 4444
```

where `nnnn` is the process identifier. If the python process is in a state where it can
accept the connection, the "nc" command will print a prompt from which you can issue pdb
commands.

Notes:

* The last step (kill and nc) can be successfully issued at most once for a given process.
Thus if you might want a colleague to carry out the debugging, think carefully before
issuing these commands yourself.
* This procedure will not work on processes other than the main "runner.py" one, because
only that process has the required trap handling set up.
