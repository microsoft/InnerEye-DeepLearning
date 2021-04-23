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
There are two sections of that kind, one for the knee data and one for the brain data. Copy and paste all the line
with `curl` commands into a text file, for example called `curl.txt`. In total, there should be 10 lines with `curl` 
commands for the knee data, and 7 for the brain data (including the SHA256 file).

## Download the dataset directly to blob storage via Azure Data Factory

We are providing a script that will bulk download all files in the FastMRI dataset from AWS to Azure blob storage.
To start that script, you need
- The file that contains all the `curl` commands to download the data, see above. The script will extract all the
 AWS access tokens from the `curl` commands.
- The connection string to the Azure storage account that stores your dataset. To get that, navigate to the 
[Azure Portal](https://portal.azure.com), and search for the storage account that you created to hold your datasets
(Step 4 in [AzureML setup](setting_up_aml.md)). On the left hand navigation, there is a section "Access Keys", select
that and copy out the connection string (it will look something like 
`DefaultEndpointsProtocol=....==;EndpointSuffix=core.windows.net`)

Then run the script to download the dataset as follows, providing the path the the file with the curl commands
and the connection string as commandline arguments, enclosed in double quotes:
`python InnerEye/Scripts/prepare_fastmri.py --curl curl.txt --connection_string "<your_connection_string"`

This script will
- Authenticate against Azure either using the Service Principal credentials that you set up in Step 3 of the
 [AzureML setup](setting_up_aml.md), or your own credentials.
- Create an Azure Data Factory in the same resource group as the AzureML workspace
- Create pipelines to download the datasets in compressed form to the `datasets` container in the storage account
you supplied, copying all of them into the `fastmri_compressed` folder
- Create pipelines to download the datasets and uncompress them, going into the `datasets` container in the 
storage account you supplied, with subfolders `knee_singlecoil`, `knee_multicoil`, and `brain_multicoil`.
- Run all the pipelines and delete the Data Factory.
This can take a few minutes to complete.


# Troubleshooting
If you see a runtime error saying "The subscription is not registered to use namespace 'Microsoft.DataFactory'", then 
follow the steps described [here](https://stackoverflow.com/a/48419951/5979993), to enable DataFactory for your
subscription.
