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

Then run the script to download the dataset as follows, providing the path the the file with the curl commands
and the connection string as commandline arguments, enclosed in quotes:
`python InnerEye/Scripts/prepare_fastmri.py --curl curl.txt --connection_string "<your_connection_string"`

This script will
- Authenticate against Azure either using the Service Principal credentials that you set up in Step 3 of the
 [AzureML setup](setting_up_aml.md), or your own credentials.
- Create an Azure Data Factory in the same resource group as the AzureML workspace
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

There is an example model already included in the InnerEye toolbox, that uses the `knee_multicoil` dataset. Please
check out [fastmri_varnet.py](../InnerEye/ML/configs/other/fastmri_varnet.py). As with all InnerEye models, you can
start a training run by specifying the name of the class that defines the model, like this:
```shell script
python InnerEye/ML/runner.py --model FastMri --azureml=True --num_nodes=4
```
This will start an AzureML job with 4 nodes training at the same time. Depending on how you set up your compute
cluster, this will use a different number of GPUs: For example, if your cluster uses ND24 virtual machines, where 
each VM has 4 Tesla P40 cards, training will use a total of 16 GPUs.

As common with multiple nodes, training time will not scale linearly with increased number of nodes. The following
table gives a rough overview of time to train 1 epoch of the FastMri model in the InnerEye toolbox 
on our cluster (4 Tesla P40 cards per node):

| Step | 1 node (4 GPUs) | 2 nodes (8 GPUs) | 4 nodes (16 GPUs) | 8 nodes (32 GPUs) |
| --- | --- | --- | --- |
| Download training data (1.25 TB) | 22min | 22min | 22min | 22min |
| Train and validate 1 epoch | 4h 15min | 2h 13min | 1h 6min | 34min |
| Evaluate on test set | 30min | 30min | 30min | 30min |
| Total time for 1 epoch | 5h 7min | 3h 5min | 1h 58min | 1h 26min |
| Total time for 50 epochs | 8 days | 4.6 days | 2.3 days | 1.2 days|

Note that the download times depend on the type of Azure storage account that your workspace is using. We recommend 
using Premium storage accounts for optimal performance.
