# Training a Hello World segmentation model

In the configs folder, you will find a config file called [HelloWorld.py](../InnerEye/ML/configs/segmentation/HelloWorld.py)
We have created this file to demonstrate how to:

1. Subclass SegmentationModelBase which is the base config for all segmentation model configs
1. Configure the UNet3D implemented in this package
1. Configure Azure HyperDrive based parameter search

This model can be trained from the commandline from the root of the repo: `python InnerEye/ML/runner.py --model=HelloWorld`.
When used like this, it will use dummy 3D scans as the training data, that are included in this repository. Training will run
on your local dev machine.

In order to get this model to train in AzureML, you need to upload the data to blob storage. This can be done via
[Azure Storage Explorer](https://azure.microsoft.com/en-gb/features/storage-explorer/) or via the
[Azure commandline tools](https://docs.microsoft.com/en-us/cli/azure/). Please find the detailed instructions for both
options below.

Before uploading, you need to know what storage account you have set up to hold the data for your AzureML workspace, see
[Step 4 in the Azure setup](setting_up_aml.md): For the upload you need to know the name of that storage account.

## Option 1: Upload via Azure Storage explorer

First install [Azure Storage Explorer](https://azure.microsoft.com/en-gb/features/storage-explorer/).

When starting Storage Explorer, you need to [log in to Azure](https://docs.microsoft.com/en-gb/azure/vs-azure-tools-storage-manage-with-storage-explorer?tabs=windows).

* Select your subscription in the left-hand navigation, and then the storage account that you set up earlier.
* There should be a section "Blob Containers" for that account.
* Right-click on "Blob Containers", and choose "Create Blob Container". Give that container the name "datasets"
* Click on the newly created container "datasets". You should see no files present.
* Press "Upload" / "Upload folder"
* As the folder to upload, select `<repo_root>/Tests/ML/test_data/train_and_test_data`
* As the destination directory, select `/hello_world`.
* Start the upload. Press the "Refresh" button after a couple of seconds, you should now see a folder `hello_world`, and inside of it, a subfolder `train_and_test_data`.
* Press "Upload" / "Upload files".
* Choose `<repo_root>/Tests/ML/test_data/dataset.csv`, and `/hello_world` as the destination directory.
* Start the upload and refresh.
* Verify that you now have files `/hello_world/dataset.csv` and `/hello_world/train_and_test_data/id1_channel1.nii.gz`

## Option 2: Upload via the Azure CLI

First, install the [Azure commandline tools](https://docs.microsoft.com/en-us/cli/azure/).

Run the following in the command prompt:

```shell
az login
az account list
```

If the `az account list` command returns more than one subscription, run `az account set --name "your subscription name"`

The code below assumes that you are uploading to a storage account that has the name
`stor_acct`, please replace with your actual storage account name.

```shell
cd <your_repository_root>
az storage container create --account-name stor_acct --name datasets
az storage blob upload --account-name stor_acct --container-name datasets --file ./Tests/ML/test_data/dataset.csv --name hello_world/dataset.csv
az storage blob upload-batch --account-name stor_acct --destination datasets --source ./Tests/ML/test_data/train_and_test_data --destination-path hello_world/train_and_test_data
```

## Create an AzureML datastore

A "datastore" in AzureML lingo is an abstraction for the ML systems to access files that can come from different places. In our case, the datastore points to a storage container to which we have just uploaded the data.

Instructions to create the datastore are given
[in the AML setup instructions](setting_up_aml.md) in step 5.

## Run the HelloWorld model in AzureML

Double-check that you have copied your Azure settings into the settings file, as described
[in the AML setup instructions](setting_up_aml.md) in step 6.

Then execute:

```shell
conda activate InnerEye
python InnerEye/ML/runner.py --model=HelloWorld --azureml
```
