# Debugging and Monitoring Jobs

### Using TensorBoard to monitor AzureML jobs

* **Existing jobs**: execute [`InnerEye/Azure/tensorboard_monitor.py`](/InnerEye/Azure/tensorboard_monitor.py) 
with either an experiment id `--experiment_name` or a list of run ids `--run_ids job1,job2,job3`. 
If an experiment id is provided then all of the runs in that experiment will be monitored. Additionally You can also 
filter runs by type by the run's status, setting the `--filters Running,Completed` parameter to a subset of
`[Running, Completed, Failed, Canceled]`. By default Failed and Canceled runs are excluded.

To quickly access this script from PyCharm, there is a template PyCharm run configuration 
`Template: Tensorboard monitoring` in the repository. Create a copy of that, and modify the commandline 
arguments with your jobs to monitor.

* **New jobs**: when queuing a new AzureML job, pass `--tensorboard=True`, which will automatically start a new TensorBoard
session, monitoring the newly queued job. 

### Resource Monitor
GPU and CPU usage can be monitored throughout the execution of a run (local and AML) by setting the monitoring interval 
for the resource monitor eg: `--monitoring_interval_seconds=5`. This will spawn a separate process at the start of the
run which will log both GPU and CPU utilization and memory consumption. These metrics will be written to AzureML as
well as a separate TensorBoard logs file under `Diagnostics`.

### Debugging setup on local machine

For full debugging of any non-trivial model, you will need a GPU. Some basic debugging can also be carried out on
standard Linux or Windows machines.

The main entry point into the code is [`InnerEye/ML/runner.py`](/InnerEye/ML/runner.py). The code takes its 
configuration elements from commandline arguments and a settings file, 
[`InnerEye/settings.yml`](/InnerEye/settings.yml). 

A password for the (optional) Azure Service 
Principal is read from `InnerEyeTestVariables.txt` in the repository root directory. The file 
is expected to contain a line of the form
```
APPLICATION_KEY=<app key for your AML workspace>
```

For developing and running your own models, you will probably find it convenient to create your own variants of
`runner.py` and `settings.yml`, as detailed in the page on [model building](building_models.md).

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


# Debugging on an AzureML node

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
```shell script
ps aux | grep 'python.*runner.py' | egrep -wv 'bash|grep'
```
You may need to vary this if it does not yield exactly one line of output.
* Note the process identifier (the value in the PID column, generally the second one).
* Issue the commands
```shell script
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
