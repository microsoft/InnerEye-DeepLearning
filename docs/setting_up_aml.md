# How to setup Azure Machine Learning for InnerEye


In order to be able to train models on Azure Machine Learning (AML) you will need to setup your environment in the 
Azure Portal first. In this document we will walk you through this process step-by-step.

In short, you will need to:
* Set up an Azure Machine Learning (AzureML) Workspace
* Create a compute cluster to run your experiments.
* Optional: Register your application to create a Service Principal Object.
* Optional: Set up a storage account to store your datasets. You may already have such a storage account, or you may
want to re-use the storage account that is created with the AzureML workspace - in both cases, you can skip this step.
* Update your [train_variables.yml](/InnerEye/train_variables.yml) file and KeyVault with your own credentials.

Once you're done with these steps, you will be ready for the next steps described in [Creating a dataset](https://github.com/microsoft/InnerEye-createdataset), 
[Building models in Azure ML](building_models.md) and 
[Sample segmentation and classification tasks](sample_tasks.md).

**Prerequisite**: an Azure account and a corresponding Azure subscription. See the 
[Get started with Azure](https://azure.microsoft.com/en-us/get-started/) page
for more information on how to set up your account and your subscription. Here are more detailed instructions on how to
[manage accounts and subscriptions with Azure](https://docs.microsoft.com/en-us/azure/cost-management-billing/manage/).

## Automatic Deployment

Click on this link to automatically create
- an AzureML workspace
- an associated storage account that will hold all training results
- and a computer cluster for training.
This replaces steps 1 and 2 below.

[![Deploy To Azure](https://raw.githubusercontent.com/Azure/azure-quickstart-templates/master/1-CONTRIBUTION-GUIDE/images/deploytoazure.svg?sanitize=true)](https://portal.azure.com/#create/Microsoft.Template/uri/https%3A%2F%2Fgithub.com%2FMicrosoft%2FInnerEye-DeepLearning%2Fblob%2Fantonsc%2Fdeploy%2Fazure-pipelines%2Fazure_deployment_template.json)

- You will be asked to create a new `Resource Group`, a logical grouping that will hold all the Azure resources that
the script will create. In doing that, you will need to choose a location where all your Azure resources live - here,
pick a location that is compliant with the legal requirements that your own datasets have (for example, your data may
need to be kept inside of the UK)
- Then choose a name for your AzureML workspace. Use letters and numbers only, because other resources will be created
using the workspace name as a prefix.

You can invoke the deployment also by going to [Azure](https://ms.portal.azure.com/#create/Microsoft.Template), 
selecting "Build your own template", and in the editor upload the 
[json template file](/azure-pipelines/azure_deployment_template.json) included in the repository.

### Step 1: Create an AzureML workspace

You can skip this if you have chosen automatic deployment above.

Assuming you have an Azure account and an Azure subscription, to create an AzureML workspace you will need to:
1. Connect to the [Azure portal](https://aka.ms/portal) with your account.
2. At the top of the home page, you will see a list of Azure services (alternatively you can also use the search bar).
You will need to select "Machine Learning" and click `+ Create`. You will then have to select your subscription, 
and create a new `Resource Group`. Then, give your workspace a name, such as
`MyInnerEye-Workspace`, and choose the correct Region suitable for your location as well as the 
desired `Workspace Edition`. Finish by clicking on `Review + Create` and then `Create`. 

You can find more details about how to set up an AzureML workspace in 
the Azure documentation [here](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-manage-workspace).

### Step 2: Create a compute cluster for your experiments

In order to be able to run experiments you will need to create a compute cluster attached to your AzureML workspace.
You can skip this if you have chosen automatic deployment above.

We recommend using [low priority](https://docs.microsoft.com/en-us/azure/batch/batch-low-pri-vms) clusters, since 
they only cost a fraction of the dedicated VMs.
As a reference, the Prostate model and the Head and Neck model require VMs with 4 GPUs with at least 16GB of memory
per GPU, for example `Standard_ND24s`, `Standard_NC24s_v3` or `Standard_NC24s_v2`.

You need to ensure that your Azure subscription actually has a quota for accessing GPU machines. To see your quota,
find your newly created AzureML workspace in the [Azure portal](http://portal.azure.com), using the search bar at the
top. Then choose "Usage and Quotas" in the left hand navigation. You should see your actual core usage and your quota,
like "0/100" meaning that you are using 0 nodes out of a quota of 100. If you don't see a quota for both dedicated AND
low priority nodes, click on the "Request Quota" button at the bottom of the page to create a ticket with Azure support.

Details about creating compute clusters can be found 
[here](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-set-up-training-targets#set-up-in-azure-machine-learning-studio).
Note down the name of your compute cluster - this will later go into the `cluster` entry of your settings
file `train_variables.yml`.


### Step 3 (Optional): Create a Service Principal Authentication object.

Training runs in AzureML can be submitted either under the name of user who started it, or as a generic identity called
"Service Principal". Using the generic identity is essential if you would like to submit training runs from code,
for example from within an Azure pipeline. But even if you are not planning to submit training runs from code, you
may choose to use a Service Principal because it can make access management easier.

If you would like the training runs to have the identity of the user who started it, there's nothing special to do - 
when you first try to submit a job, you will be prompted to authenticate in the browser.

If you would like to use Service Principal, you will need to create it in Azure first, and then store its password
in a file inside your repository. This
will allow you to access all resources linked to your newly created Azure ML workspace with a single secret key after you
have finished the setup. You can find more information about application registrations and service principal objects
[here](https://docs.microsoft.com/en-us/azure/active-directory/develop/app-objects-and-service-principals).

To create the Service Principal:

 1. Navigate back to [aka.ms/portal](https://aka.ms/portal)
 1. Navigate to `App registrations` (use the top search bar to find it).
 1. Click on `+ New registration` on the top left of the page.
 1. Choose a name for your application e.g. `MyInnerEye-ServicePrincipal` and click `Register`.
 1. Once it is created you will see your application in the list appearing under `App registrations`. This step might take 
 a few minutes. Click on the resource to access its properties. In particular, you will need the application ID. 
 You can find this ID in the `Overview` tab (accessible from the list on the left of the page). 
 Note it down for later - this will go into the `application_id` entry of your settings
file `train_variables.yml`.
 1. You need to create an application secret to access the resources managed by this service principal. 
 On the pane on the left find `Certificates & Secrets`. Click on `+ New client secret` (bottom of the page), note down your token. 
 Warning: this token will only appear once at the creation of the token, you will not be able to re-display it again later. 
 1. Save your application secret in your local machine:
     1. You can either set the environment variable `APPLICATION_KEY` to the application secret you just generated.
     1. Or you can create a file called `InnerEyeTestVariables.txt` in the root directory of your git repository. 
     That file should contain a single line of the form `APPLICATION_KEY=TheApplicationSecretYouJustCreated`
 1. You will need to share this application secret with your colleagues if they also want to use Service Principal
 authentication. They will also either need to set the environment variable, or create the text file with the secret.
 
Now that your service principal is created, you need to give permission for it to access and manage your AzureML workspace. 
To do so:
1. Go to your AzureML workspace. To find it you can type the name of your workspace in the search bar above.
2. On the left of the page go to `Access control`. Then click on `+ Add` > `Add role assignment`. A pane will appear on the
 the right. Select `Role > Contributor` and leave `Assign access`. Finally in the `Select` field type the name
of your Service Principal and select it. Finish by clicking `Save` at the bottom of the pane.

Your Service Principal is now all set!


### Step 4: Create a storage account for your datasets.
In order to train your model in the cloud, you will need to upload your datasets to Azure. For this, you will have two options:
 * Store your datasets in the storage account linked to your AzureML workspace (see Step 1 above).
 * Create a new storage account whom you will only use for dataset storage purposes. 

If you want to create a new storage account:

0. Go to [aka.ms/portal](https://aka.ms/portal)
1. Type `storage accounts` in the top search bar and open the corresponding page.
2. On the top of the page click on `+ Add`.
3. Select your subscription and the resource group that you created earlier.
4. Specify a name for your storage account, and a location suitable for your data.
6. Click create.
7. Once your resource is created you can access it by typing its name in the top search bar. 

Once your datasets storage account is ready, find it in the Azure portal, and choose "Containers" in the left hand
navigation. Create a new container called "datasets". Inside of this container, each dataset will later occupy one
folder.

### Step 5: Create a datastore
You will now need to create a datastore in AzureML. 

- For that, you need to know the account key for the storage account
that holds your datasets - this can be the name of  the storage account created in Step 4, or the storage account that came with the 
AzureML workspace in Step 1. 
- Find the storage account in the Azure.
- In the left hand pane, choose "Access keys", and copy one of the keys to the clipboard.
- Now go to your AzureML workspace to create the datastore. In the left hand navigation pane, choose "Datastores".
- Click `+ New datastore`. 
- Create a datastore called `innereyedatasets`. 
- In the fields for storage account, type in your dataset storage account
name (newly created or the AzureML one). 
- In blob containers, choose "datasets" from the dropdown. If the dropdown is empty, go back to Step 4 and 
create a container called "datasets".
- Choose "Authentication Type: Account Key" and paste the storage account key from the clipboard into the
"Account Key" field.
- Done!

If you want to make use of super fast download of datasets for local debugging, you can also store the account key
on your local machine:
- Create a file called `InnerEyeTestVariables.txt` in the root directory of your git repository, and add a line
`DATASETS_ACCOUNT_KEY=TheKeyThatYouJustCopied`.
- Copy the name of the datasets storage account into the field `datasets_storage_account` of your settings file
`train_variables.yml`.


### Step 6: Update the variables in `train_variables.yml`
The [train_variables.yml](/InnerEye/train_variables.yml) file is used to store your Azure setup. In order to be able to
train your model you will need to update this file using the settings for your Azure subscription.
1. You will first need to retrieve your `tenant_id`. You can find your tenant id by navigating to
`Azure Active Directory > Properties > Tenant ID` (use the search bar above to access the `Azure Active Directory` 
resource. Copy and paste the GUID to the `tenant_id` field of the `.yml` file. More information about Azure tenants can be found 
[here](https://docs.microsoft.com/en-us/azure/active-directory/develop/quickstart-create-new-tenant).
2. You then need to retrieve your subscription id. In the search bar look for `Subscriptions`. Then in the subscriptions list,
look for the subscription you are using for your workspace. Copy the value of the `Subscription ID` in the corresponding 
field of [train_variables.yml](/InnerEye/train_variables.yml).
3. Copy the application ID of your Service Principal that you retrieved earlier (cf. Step 3) to the `application_id` field.
If you did not set up a Service Principal, fill that with an empty string or leave out altogether.
6. Update the `resource_group:` field with your resource group name (created in Step 1).
7. Update the `workspace_name:` field with the name of the AzureML workspace created in Step 1.
8. Update the `cluster:` field with the name of your own compute cluster (Step 2). If you chose automatic
deployment, this cluster will be called "NC24-LowPrio"

Leave all other fields as they are for now.

## Done!
You should be all set now! 

You can now go to the next step [Creating a dataset](https://github.com/microsoft/InnerEye-createdataset) to learn
how to upload and make your dataset ready for training. 
