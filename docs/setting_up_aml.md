# How to setup Azure Machine Learning for InnerEye


In order to be able to train models on Azure Machine Learning (AML) you will need to setup your environment in the 
Azure Portal first. In this document we will walk you through this process step-by-step.

In short, you will need to:
* Set up an Azure Machine Learning Workspace.
* Register your application to create a Service Principal Object.
* Set up a storage account to store your data.
* Create a compute cluster to run your experiments.
* Update your [train_variables.yml](/InnerEye/train_variables.yml) file and KeyVault with your own credentials.

Once you're done with these steps, you will be ready for the next steps described in [Creating a dataset](https://github.com/microsoft/InnerEye-createdataset), 
[Building models in Azure ML](building_models.md) and 
[Sample segmentation and classification tasks](sample_tasks.md).


### Step 1: Create an AML workspace

Prerequisite: an Azure account and a corresponding Azure subscription. See the [Get started with Azure](https://azure.microsoft.com/en-us/get-started/) page
for more information on how to set up your account and your subscription. Here are more detailed instructions on how to
[manage accounts and subscriptions with Azure](https://docs.microsoft.com/en-us/azure/cost-management-billing/manage/).

Assuming you have an Azure account and an Azure subscription, to create an AML workspace you will need to:
1. Connect to the [Azure portal](https://aka.ms/portal) with your account.
2. At the top of the home page, you will see a list of Azure services (alternatively you can also use the search bar).
You will need to select "Machine Learning" and click `+ Create`. You will then have to select your subscription, 
and create a new `Resource Group`. Then, give a name to your workspace a name, such as
`MyInnerEye-Workspace`, and choose the correct Region suitable for your location as well as the 
desired `Workspace Edition`. Finish by clicking on `Review + Create` and then `Create`. You can find more details about how to set up an AML workspace in 
the Azure documentation [here](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-manage-workspace).


### Step 2: Register your application to create a Service Principal Authentication object. 
You will need to register your application in Azure to be able to create a Service Principal Authentication object. This
will allow you to access all resources linked to your newly created Azure ML workspace with a single secret key after you
have finished the setup. You can find more information about application registrations and service principal objects
[here](https://docs.microsoft.com/en-us/azure/active-directory/develop/app-objects-and-service-principals).

To register the application:

 1. Navigate back to [aka.ms/portal](https://aka.ms/portal)
 1. Navigate to `App registrations` (use the top search bar to find it).
 1. Click on `+ New registration` on the top left of the page.
 1. Choose a name for your application e.g. `MyInnerEye-ServicePrincipal` and click `Register`.
 1. Once it is created you will see your application in the list appearing under `App registrations`. This step might take 
 a few minutes. Click on the resource to access its properties. In particular, you will need the application ID. 
 You can find this ID in the `Overview` tab (accessible from the list on the left of the page). Note it down for later.
 1. You need to create an application secret to access the resources managed by this service principal. 
 On the pane on the left find `Certificates & Secrets`. Click on `+ New client secret` (bottom of the page), note down your token. 
 Warning: this token will only appear once at the creation of the token, you will not be able to re-display it again later. 
 Copy this token to your password manager or keep it in a secure location where you will be able to retrieve it for the
 next steps.
 
Now that your service principal is created, you need to give permission for it to access and manage your AML workspace. 
To do so:
1. Go to your AML workspace. To find it you can type the name of your workspace in the search bar above.
2. On the left of the page go to `Access control`. Then click on `+ Add` > `Add role assignment`. A pane will appear on the
 the right. Select `Role > Contributor` and leave `Assign access`. Finally in the `Select` field type the name
of your Service Principal and select it. Finish by clicking `Save` at the bottom of the pane.

Your Service Principal is now all set!

### Step 3: Get the key of the AML storage account
When you created your AML workspace, a [storage account](https://docs.microsoft.com/en-us/azure/storage/common/storage-account-overview)
 was automatically created for you. This storage account will be used to save all results of your experiments that will 
 be displayed in the Azure Dashboard. In order to let the code write to this storage account you will need 
 to retrieve the access key for this account. 

1. Navigate to your AML workspace (created in step 1) by typing its name in the top search bar.
2. In the `Overview` pane, at the top right, you will find a field called `Storage` with a storage account linked to it.
3. Click on the storage account name to open it. For the next steps you will
need to retrieve the `storage account ID`. For this go to the `Properties` tab of the storage account. There you will find
the `Storage account resource ID`. Save this value somewhere for the next steps.
4. You will also need to access the key of this storage account. You can find the access keys in the `Access keys` tab of 
the storage account (in the left pane). You will need to temporarily save the value of the first key for the next step 
in a secure location, preferably in your password manager. 

### Step 4: Create a storage account for your datasets.
In order to train your model in the cloud, you will need to upload your datasets to Azure. For this, you will have two options:
 * Store your datasets in the storage account linked to your AML workspace (see Step 3 above).
 * Create a new storage account whom you will only use for dataset storage purposes. 

You will need to create a blob container called `datasets` in whichever account you choose. InnerEye will look for datasets
in this blob.

If you want to create a new storage account:

0. Go to [aka.ms/portal](https://aka.ms/portal)
1. Type `storage accounts` in the top search bar and open the corresponding page.
2. On the top of the page click on `+ Add`.
3. Select your subscription and the resource group that you created earlier.
4. Specify a name for your storage account.
5. Choose a location suitable for you.
6. Click create.
7. Once your resource is created you can access it by typing its name in the top search bar. You will then need to retrieve the storage account ID 
and the access key of this storage account following the same instructions as you did for the dataset storage account (cf. Step 3.7 above).
Be careful not to mix up the `dataset storage account` and the AML `storage account` IDs and keys.

### Step 5: Create a compute cluster for your experiments
In order to be able to run experiments you will need to create a compute cluster attached to your AML workspace. In order
to do so follow the steps described [here](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-set-up-training-targets#set-up-in-azure-machine-learning-studio).
Note down the name of your compute target.

We recommend using [low priority](https://docs.microsoft.com/en-us/azure/batch/batch-low-pri-vms) clusters, they only cost a fraction of the dedicated VMs.
As a reference, the Prostate model and the Head and Neck model require VMs with 4 GPUs with at least 16GB of memory per GPU, for example Standard_ND24s, Standard_NC24s_v3 or Standard_NC24s_v2.

### Step 6: Create a datastore
You will need to create a datastore in AzureML. Go to the `Datastores` tab in AML, and then click `+ New datastore`. 
Create a datastore called `innereyedatasets`. In the fields for storage account, type in your dataset storage account name,
and under blob container, type `datasets` (this is the blob you created in Step 4).

### Step 7: Update the variables in `train_variables.yml`
The [train_variables.yml](/InnerEye/train_variables.yml) file is used to store your Azure Credentials. In order to be able to
train your model you will need to update this file using your own credentials.
1. You will first need to retrieve your `tenant_id`. You can find your tenant id by navigating to
`Azure Active Directory > Properties > Tenant ID` (use the search bar above to access the `Azure Active Directory` 
resource. Copy and paste the GUID to the `tenant_id` field of the `.yml` file. More information about Azure tenants can be found 
[here](https://docs.microsoft.com/en-us/azure/active-directory/develop/quickstart-create-new-tenant).
2. You then need to retrieve your subscription id. In the search bar look for `Subscriptions`. Then in the subscriptions list,
look for the subscription you are using for your workspace. Copy the value of the `Subscription ID` in the corresponding 
field of [train_variables.yml](/InnerEye/train_variables.yml).
3. Copy the application ID of your service principal that you retrieved earlier (cf. Step 2.4) to the `application_id` field.
4. In the `storage_account:` field copy the ID of the AML storage account (retrieved in Step 3).
5. Similarly in the `datasets_storage_account:` field copy the ID of the dataset storage account (retrieved in Step 4). If
you chose not to create a separate account for your dataset in Step 4, then specify the same value as in the 
`storage_account` field, to tell the code to use the same storage account.
6. Update the `resource_group:` field with your resource group name (created in Step 1).
7. Update the `workspace_region:` and `workspace-name:` fields according to the values you chose in Step 1.
8. Update the `gpu_cluster_name:` field with the name of your own compute cluster (Step 5).

Leave all other fields as they are for now.

To securely store your keys for your two storage accounts, we recommend using 
[KeyVault](https://azure.microsoft.com/en-gb/services/key-vault/). Your AML workspace comes with a KeyVault automatically. 
To only thing you need to do is to add your storage accounts keys in this key vault. 
To make it easier for you we have created a script called [write_secrets.py](/InnerEye/Scripts/write_secrets.py). 
You simply need to run this script with the correct arguments and it will update all secrets for you.

### Step 8: Save your application secret in your local machine
Last but not least, in order to be able to authenticate to your AML workspace via the Service Principal 
that you created earlier, you need to set the environment variable `APPLICATION_KEY` to the value of
the application secret that you retrieved in Step 2.5.

You should be all set now! 

You can now go to the next step [Creating a dataset](https://github.com/microsoft/InnerEye-createdataset) to learn
how to upload and make your dataset ready for training. 
