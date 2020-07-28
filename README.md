# InnerEye-DeepLearning

[![Build Status](https://innereye.visualstudio.com/InnerEye/_apis/build/status/InnerEye-DeepLearning/InnerEye-DeepLearning-PR?branchName=master)](https://innereye.visualstudio.com/InnerEye/_build?definitionId=112&branchName=master)

## Overview

This is a medical imaging Deep-Learning library to train and deploy models on [Azure Machine Learning Services](https://docs.microsoft.com/en-gb/azure/machine-learning/) or [Azure Stack Hub](https://azure.microsoft.com/en-us/products/azure-stack/hub/).

We support segmentation, classification and regression models. We enable users to take a dataset and train high performance models with ensembles in a few simple steps. These models can then be deployed on Azure Machine Learning Services or Azure Stack Hub.

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
