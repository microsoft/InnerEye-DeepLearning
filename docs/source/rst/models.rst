Pre-Trained Models
==================

InnerEye-DeepLearning currently has two pre-trained models avaiable for use
in segmentation tasks. This page describes how to set up and use these models.
For specific information on the models, please refer to the relevant model card:

.. toctree::
   :maxdepth: 1

   ../md/hippocampus_model.md
   ../md/lung_model.md


Terms of use
------------

Please note that all models provided by InnerEye-DeepLearning are intended for
research purposes only. You are responsible for the performance, the necessary testing,
 and if needed any regulatory clearance for any of the models produced by this toolbox.

Usage
-----

The following instructions assume you have completed the preceding setup
steps in the `InnerEye
README <https://github.com/microsoft/InnerEye-DeepLearning/>`__, in
particular, `Setting up Azure Machine Learning <setting_up_aml.md>`__.

Create an AzureML Dataset
~~~~~~~~~~~~~~~~~~~~~~~~~

To evaluate pre-trained models on your own data, you will first need to register
an `Azure ML
Dataset <https://docs.microsoft.com/en-us/azure/machine-learning/v1/how-to-create-register-datasets>`__.
You can follow the instructions in the for `creating
datasets <creating_dataset.md>`__ in order to do this.

Downloading the models
~~~~~~~~~~~~~~~~~~~~~~

The saved weights for each model can be found in their respective :ref:`model cards<Pre-Trained Models>`.
You will need to download the weights and source code for the model that you wish to use.

Registering a model in Azure ML
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To evaluate the model in Azure ML, you must first `register an Azure ML
Model <https://docs.microsoft.com/en-us/python/api/azureml-core/azureml.core.model.model?view=azure-ml-py#remarks>`__.
To register the pre-trained model in your AML Workspace, unpack the
source code downloaded in the previous step and follow InnerEye's
`instructions to upload models to Azure ML <move_model.md>`__.

Run the following from a folder that contains both the ``ENVIRONMENT/``
and ``MODEL/`` folders (these exist inside the downloaded model files):

.. code:: shell

   WORKSPACE="fill with your workspace name"
   GROUP="fill with your resource group name"
   SUBSCRIPTION="fill with your subscription ID"

   python InnerEye/Scripts/move_model.py \
       --action upload \
       --path . \
       --workspace_name $WORKSPACE \
       --resource_group $GROUP \
       --subscription_id $SUBSCRIPTION \
       --model_id <Model Name>:<Model Version>

Evaluating the model
~~~~~~~~~~~~~~~~~~~~

You can evaluate the model either in Azure ML or locally using the
downloaded checkpoint files. These 2 scenarios are described in more
detail, along with instructions in `testing an existing
model <building_models.md#testing-an-existing-model>`__.

For example, to evaluate the model on your Dataset in Azure ML, run the
following from within the directory ``*/MODEL/final_ensemble_model/``

.. code:: shell

   CLUSTER="fill with your cluster name"
   DATASET_ID="fill with your dataset name"

   python InnerEye/ML/runner.py \
       --azure_dataset_id $DATASET_ID \
       --model <Model Name> \
       --model_id <Model Name>:<Model Version> \
       --experiment_name <experiement name> \
       --azureml \
       --no-train \
       --cluster $CLUSTER
       --restrict_subjects=0,0,+

Deploy with InnerEye Gateway
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To deploy a model using the InnerEye Gateway, see the instructions in the `Gateway Repo <https://github.com/microsoft/InnerEye-Gateway/>`__.
