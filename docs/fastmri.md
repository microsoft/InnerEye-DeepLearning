# Working with FastMRI models

The InnerEye toolbox supports models built for the [fastMRI challenge](https://fastmri.org/). The challenge supports
research into novel methods for reconstructing magnetic resonance images from undersampled data, with the ultimate
goal of speeding up MR acquisition and reducing cost.

Building even the baseline models for this challenge is computationally demanding, and can run for several days on
a single GPU box. With the help of the InnerEye toolbox and distributed training in Azure Machine Learning, this time
can be reduced dramatically.

In order to work with the challenge data in Azure, you will need to
- Register for the challenge
- Have the InnerEye toolbox set up with Azure as described [here](setting_up_aml.md)
- Download and prepare the challenge data, or use the script that we provide here to bulk download directly from
AWS into Azure blob storage.

## Registering for the challenge
In order to download the dataset, you need to register [here](https://fastmri.org/dataset/).

You will shortly receive an email with links to the dataset. In that email, there are two sections containing  
scripts to download the data, like this:
```
To download Knee MRI files, we recommend using curl with recovery mode turned on:
curl -C "https://....amazonaws.com/knee_singlecoil_train.tar.gz?AWSAccessKeyId=...Expires=1610309839" --output knee_singlecoil_train.tar.gz"
...
```
There are two sections of that kind, one for the knee data and one for the brain data. Copy and paste *all* the lines
with `curl` commands into a text file, for example called `curl.txt`. In total, there should be 10 lines with `curl` 
commands for the knee data, and 7 for the brain data (including the SHA256 file).

## Download the dataset directly to blob storage via Azure Data Factory

We are providing a script that will bulk download all files in the FastMRI dataset from AWS to Azure blob storage.
To start that script, you need
- The file that contains all the `curl` commands to download the data (see above). The downloading script will 
extract all the AWS access tokens from the `curl` commands.
- The connection string to the Azure storage account that stores your dataset. 
  - To get that, navigate to the [Azure Portal](https://portal.azure.com), and search for the storage account 
  that you created to hold your datasets (Step 4 in [AzureML setup](setting_up_aml.md)). 
  - On the left hand navigation, there is a section "Access Keys", select that and copy out the connection string 
  (sanity check: it should look something like `DefaultEndpointsProtocol=....==;EndpointSuffix=core.windows.net`)
- The Azure location where the Data Factory should be created (for example "westeurope"). The Data Factory should 
  live in the same Azure location as your AzureML workspace and storage account. To check the location, 
  find the workspace in the [Azure Portal](https://portal.azure.com), the location is shown on the overview page.

Then run the script to download the dataset as follows, providing the path the the file with the curl commands
and the connection string as commandline arguments, enclosed in quotes:
`python InnerEye/Scripts/prepare_fastmri.py --curl curl.txt --connection_string "<your_connection_string>"` --location westeurope

This script will
- Authenticate against Azure either using the Service Principal credentials that you set up in Step 3 of the
 [AzureML setup](setting_up_aml.md), or your own credentials. To use the latter, you need to be logged in via the Azure
 command line interface (CLI), available [here](https://docs.microsoft.com/en-us/cli/azure/) for all platforms.
- Create an Azure Data Factory in the same resource group as the AzureML workspace.
- Create pipelines to download the datasets in compressed form to the `datasets` container in the storage account
you supplied, and uncompress them.
- Run all the pipelines and delete the Data Factory.

This whole process can take a few hours to complete. It will print progress information every 30 seconds to the console.
Alternatively, find the Data Factory "fastmri-copy-data" in your Azure portal, and click on the "Monitor" icon to 
drill down into all running pipelines.

Once the script is complete, you will have the following datasets in Azure blob storage:
- `knee_singlecoil`, `knee_multicoil`, and `brain_multicoil` with all files unpacked
- `knee_singlecoil_compressed`, `knee_multicoil_compressed`, and `brain_multicoil_compressed` with the `.tar` and 
`.tar.gz` files as downloaded. NOTE: The raw challenge data files all have a `.tar.gz` extension, even though some
of them are plain (uncompressed) `.tar` files. The pipeline corrects these mistakes and puts the files into blob storage
with their corrected extension.
- The DICOM files are stored in the folders `knee_DICOMs` and `brain_DICOMs` (uncompressed) and 
`knee_DICOMs_compressed` and `brain_DICOMs_compressed` (as `.tar` files)


### Troubleshooting the data downloading
If you see a runtime error saying "The subscription is not registered to use namespace 'Microsoft.DataFactory'", then 
follow the steps described [here](https://stackoverflow.com/a/48419951/5979993), to enable DataFactory for your
subscription.


## Running a FastMri model with InnerEye

The Azure Data Factory that downloaded the data has put it into the storage account you supplied on the commandline.
If set up correctly, this is the Azure storage account that holds all datasets used in your AzureML workspace.
Hence, after the downloading completes, you are ready to use the InnerEye toolbox to submit an AzureML job that uses
the FastMRI data.

There are 2 example models already coded up in the InnerEye toolbox, defined in 
[fastmri_varnet.py](../InnerEye/ML/configs/other/fastmri_varnet.py): `KneeMulticoil` and 
`BrainMulticoil`. As with all InnerEye models, you can start a training run by specifying the name of the class 
that defines the model, like this:
```shell script
python InnerEye/ML/runner.py --model KneeMulticoil --azureml --num_nodes=4
```
This will start an AzureML job with 4 nodes training at the same time. Depending on how you set up your compute
cluster, this will use a different number of GPUs: For example, if your cluster uses ND24 virtual machines, where 
each VM has 4 Tesla P40 cards, training will use a total of 16 GPUs.

As common with multiple nodes, training time will not scale linearly with increased number of nodes. The following
table gives a rough overview of time to train 1 epoch of the FastMri model in the InnerEye toolbox 
on our cluster (`Standard_ND24s` nodes with 4 Tesla P40 cards):

| Step | 1 node (4 GPUs) | 2 nodes (8 GPUs) | 4 nodes (16 GPUs) | 8 nodes (32 GPUs) |
| --- | --- | --- | --- | --- |
| Download training data (1.25 TB) | 22min | 22min | 22min | 22min |
| Train and validate 1 epoch | 4h 15min | 2h 13min | 1h 6min | 34min |
| Evaluate on test set | 30min | 30min | 30min | 30min |
| Total time for 1 epoch | 5h 5min | 3h 5min | 1h 58min | 1h 26min |
| Total time for 50 epochs | 9 days | 4.6 days | 2.3 days | 1.2 days|

Note that the download times depend on the type of Azure storage account that your workspace is using. We recommend 
using Premium storage accounts for optimal performance.

You can avoid the time to download the dataset, by specifying that the data is always read on-the-fly from the network.
For that, just add the `--use_dataset_mount` flag to the commandline. This may impact training throughput if
the storage account cannot provide the data quick enough - however, we have not observed a drop in GPU utilization even
when training on 8 nodes in parallel. For more details around dataset mounting please refer to the next section.


## Performance considerations for BrainMulticoil

Training a FastMri model on the `brain_multicoil` dataset is particularly challenging because the dataset is larger.
Downloading the dataset can - depending on the types of nodes - already make the nodes go out of disk space.

The InnerEye toolbox has a way of working around that problem, by reading the dataset on-the-fly from the network, 
rather than downloading it at the start of the job. You can trigger this behaviour by supplying an additional 
commandline argument `--use_dataset_mount`, for example:
```shell script
python InnerEye/ML/runner.py --model BrainMulticoil --azureml --num_nodes=4 --use_dataset_mount
```
With this flag, the InnerEye training script will start immediately, without downloading data beforehand. 
However, the fastMRI data module generates a cache file before training, and to build that, it needs to traverse the 
full dataset. This will lead to a long (1-2 hours) startup time before starting the first epoch, while it is
creating this cache file. This can be avoided by copying the cache file from a previous run into to the dataset folder. 
More specifically, you need to follow these steps:
* Start a training job, training for only 1 epoch, like
```shell script
python InnerEye/ML/runner.py --model BrainMulticoil --azureml --use_dataset_mount --num_epochs=1
```
* Wait until the job starts has finished creating the cache file - the job will print out a message 
"Saving dataset cache to dataset_cache.pkl", visible in the log file `azureml-logs/70_driver_log.txt`, about 1-2 hours
after start. At that point, you can cancel the job. 
* In the "Outputs + logs" section of the AzureML job, you will now see a file `outputs/dataset_cache.pkl` that has 
been produced by the job. Download that file.
* Upload the file `dataset_cache.pkl` to the storage account that holds the fastMRI datasets, in the `brain_multicoil` 
folder that was previously created by the Azure Data Factory. You can do that via the Azure Portal or Azure Storage
 Explorer. Via the Azure Portal, you can search for the storage account that holds your data, then select 
 "Data storage: Containers" in the left hand navigation. You should see a folder named `datasets`, and inside of that
 `brain_multicoil`. Once in that folder, press the "Upload" button at the top and select the `dataset_cache.pkl` file.
* Start the training job again, this time you can start multi-node training right away, like this:
```shell script
python InnerEye/ML/runner.py --model BrainMulticoil --azureml --use_dataset_mount --num_nodes=8. This new
```
This job should pick up the existing cache file, and output a message like "Copying a pre-computed dataset cache 
file ..."

The same trick can of course be applied to other models as well (`KneeMulticoil`).


# Running on a GPU machine

You can of course run the InnerEye fastMRI models on a reasonably large machine with a GPU for development and 
debugging purposes. Before running, we recommend to download the datasets using a tool 
like [azcopy](http://aka.ms/azcopy) into a folder, for example the `datasets` folder at the repository root.

To use `azcopy`, you will need the access key to the storage account that holds your data - it's the same storage
account that was used when creating the Data Factory that downloaded the data.
- To get that, navigate to the [Azure Portal](https://portal.azure.com), and search for the storage account 
that you created to hold your datasets (Step 4 in [AzureML setup](setting_up_aml.md)). 
- On the left hand navigation, there is a section "Access Keys". Select that and copy out one of the two keys (_not_
the connection strings). The key is a base64 encoded string, it should not contain any special characters apart from 
`+`, `/`, `.` and `=`

Then run this script in the repository root folder:
```shell script
mkdir datasets
azcopy --source-key <storage_account_key> --source https://<your_storage_acount>.blob.core.windows.net/datasets/brain_multicoil --destination datasets/brain_multicoil --recursive
```
Replace `brain_multicoil` with any of the other datasets names if needed.

If you follow these suggested folder structures, there is no further change necessary to the models. You can then
run, for example, the `BrainMulticoil` model by dropping the `--azureml` flag like this:
```shell script
python InnerEye/ML/runner.py --model BrainMulticoil
```
The code will recognize that an Azure dataset named `brain_multicoil` is already present in the `datasets` folder,
and skip the download.

If you choose to download the dataset to a different folder, for example `/foo/brain_multicoil`, you will need to
make a small adjustment to the model in [fastmri_varnet.py](../InnerEye/ML/configs/other/fastmri_varnet.py),
and add the `local_dataset` argument like this:
```python
class BrainMulticoil(FastMri):
    def __init__(self) -> None:
        super().__init__()
        self.azure_dataset_id = "brain_multicoil"
        self.local_dataset = Path("/foo/brain_multicoil")
        self.dataset_mountpoint = "/tmp/brain_multicoil"
```
