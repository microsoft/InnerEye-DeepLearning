# InnerEye-DeepLearning

[![Build Status](https://innereye.visualstudio.com/InnerEye/_apis/build/status/InnerEye-DeepLearning/InnerEye-DeepLearning-PR?branchName=master)](https://innereye.visualstudio.com/InnerEye/_build?definitionId=112&branchName=master)

## Overview

This is a deep learning toolbox to train models on medical images (or more generally, 3D images). 
It integrates seamlessly with cloud computing in Azure.
 
On the modelling side, this toolbox supports 
- Segmentation models
- Classification and regression models
- Sequence models

Classification, regression, and sequence models can be built with only images as inputs, or a combination of images
and non-imaging data as input. This supports typical use cases on medical data where measurements, biomarkers, 
or patient characteristics are often available in addition to images.

On the user side, this toolbox focusses on enabling machine learning teams to achieve more. It is cloud-first, and
relies on [Azure Machine Learning Services (AzureML)](https://docs.microsoft.com/en-gb/azure/machine-learning/) for execution,
bookkeeping, and visualization. Taken together, this gives:
- **Traceability**: AzureML keeps a full record of all experiments that were executed, including a snapshot of
the code. Tags are added to the experiments automatically, that can later help filter and find old experiments.
- **Transparency**: All team members have access to each other's experiments and results.
- **Reproducibility**: Two model training runs using the same code and data will result in exactly the same metrics. All
 sources of randomness like multithreading are controlled for.
- **Cost reduction**: Using AzureML, all compute (virtual machines, VMs) is requested at the time of starting the 
training job, and freed up at the end. Idle VMs will not incur costs. In addition, Azure low priority 
nodes can be used to further reduce costs (up to 80% cheaper).
- **Scale out**: Large numbers of VMs can be requested easily to cope with a burst in jobs.

Despite the cloud focus, all training and model testing works just as well on local compute, which is important for
model prototyping, debugging, and in cases where the cloud can't be used.

In addition, our toolbox supports:
 - Cross-validation using AzureML's built-in support, where the models for 
individual folds are trained in parallel. This is particularly important for the long-running training jobs
often seen with medical images. 
- Hyperparameter tuning using
[Hyperdrive](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-tune-hyperparameters).
- Building ensemble models.
- Easy creation of new models via a configuration-based approach, and inheritance from an existing
architecture.
 
Once training in AzureML is done, the models can be deployed from within AzureML or via 
[Azure Stack Hub](https://azure.microsoft.com/en-us/products/azure-stack/hub/).


## Getting started

1. [Setting up your environment](docs/environment.md)
1. [Setting up Azure Machine Learning](docs/setting_up_aml.md)
1. [Creating a dataset](docs/creating_dataset.md)
1. [Building models in Azure ML](docs/building_models.md)
1. [Sample Segmentation and Classification tasks](docs/sample_tasks.md)
1. [Debugging and monitoring models](docs/debugging_and_monitoring.md)

## More information

1. [Project InnerEye](https://www.microsoft.com/en-us/research/project/medical-image-analysis/)
1. [Testing](docs/testing.md)
1. [How to do pull requests](docs/pull_requests.md)
1. [Contributing](docs/contributing.md)

## References

Please cite the following references if you find the image segmentation framework useful in your research:

[Augmenting Image Guided Radiotherapy Workflows with Deep Learning](). JAMA Network Open (Under Review), 2020.    
Oktay O, Nanavati J, Schwaighofer A, Carter D, Bristow M, Tanno R, Jena R, Barnett G, Noble D, Rimmer Y, Glocker B, O’Hara K, Bishop C, Alvarez-Valle J, and Nori A.

[3D U-Net: Learning Dense Volumetric Segmentation from Sparse Annotation](https://arxiv.org/abs/1606.06650). MICCAI, Springer, 2016.  
Özgün Çiçek, Ahmed Abdulkadir, Soeren S. Lienkamp, Thomas Brox, Olaf Ronneberger

## Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.
