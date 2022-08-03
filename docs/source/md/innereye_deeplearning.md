# InnerEye-DeepLearning

[![Build Status](https://innereye.visualstudio.com/InnerEye/_apis/build/status/InnerEye-DeepLearning/InnerEye-DeepLearning-PR?branchName=main)](https://innereye.visualstudio.com/InnerEye/_build?definitionId=112&branchName=main)

## Overview

This is a deep learning toolbox to train models on medical images (or more generally, 3D images).
It integrates seamlessly with cloud computing in Azure.

On the modelling side, this toolbox supports

- Segmentation models
- Classification and regression models
- Adding cloud support to any PyTorch Lightning model, via a [bring-your-own-model setup](bring_your_own_model.md)

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
model prototyping, debugging, and in cases where the cloud can't be used. In particular, if you already have GPU
machines available, you will be able to utilize them with the InnerEye toolbox.

In addition, our toolbox supports:

- Cross-validation using AzureML's built-in support, where the models for
individual folds are trained in parallel. This is particularly important for the long-running training jobs
often seen with medical images.
- Hyperparameter tuning using
[Hyperdrive](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-tune-hyperparameters).
- Building ensemble models.
- Easy creation of new models via a configuration-based approach, and inheritance from an existing
architecture.

Once training in AzureML is done, the models can be deployed from within AzureML.

## Quick Setup

This quick setup assumes you are using a machine running Ubuntu with Git, Git LFS, Conda and Python 3.7+ installed. Please refer to the [setup guide](environment.md) for more detailed instructions on getting InnerEye set up with other operating systems and installing the above prerequisites.

### Instructions

1. Clone the InnerEye-DeepLearning repo by running the following command:

   ```shell
   git clone --recursive https://github.com/microsoft/InnerEye-DeepLearning & cd InnerEye-DeepLearning
   ```

2. Create and activate your conda environment:

   ```shell
   conda env create --file environment.yml && conda activate InnerEye
   ```

3. Verify that your installation was successful by running the HelloWorld model (no GPU required):

   ```shell
   python InnerEye/ML/runner.py --model=HelloWorld
   ```

If the above runs with no errors: Congratulations! You have successfully built your first model using the InnerEye toolbox.

If it fails, please check the
[troubleshooting page on the Wiki](https://github.com/microsoft/InnerEye-DeepLearning/wiki/Issues-with-code-setup-and-the-HelloWorld-model).

## Other Documentation

Further detailed instructions, including setup in Azure, are here:

1. [Setting up your environment](environment.md)
1. [Setting up Azure Machine Learning](setting_up_aml.md)
1. [Training a simple segmentation model in Azure ML](hello_world_model.md)
1. [Creating a dataset](creating_dataset.md)
1. [Building models in Azure ML](building_models.md)
1. [Sample Segmentation and Classification tasks](sample_tasks.md)
1. [Debugging and monitoring models](debugging_and_monitoring.md)
1. [Model diagnostics](model_diagnostics.md)
1. [Move a model to a different workspace](move_model.md)
1. [Working with FastMRI models](fastmri.md)
1. [Active label cleaning and noise robust learning toolbox](https://github.com/microsoft/InnerEye-DeepLearning/blob/1606729c7a16e1bfeb269694314212b6e2737939/InnerEye-DataQuality/README.md)
1. [Using InnerEye as a git submodule](innereye_as_submodule.md)

## Deployment

We offer a companion set of open-sourced tools that help to integrate trained CT segmentation models with clinical
software systems:

- The [InnerEye-Gateway](https://github.com/microsoft/InnerEye-Gateway) is a Windows service running in a DICOM network,
that can route anonymized DICOM images to an inference service.
- The [InnerEye-Inference](https://github.com/microsoft/InnerEye-Inference) component offers a REST API that integrates
with the InnnEye-Gateway, to run inference on InnerEye-DeepLearning models.

Details can be found [here](deploy_on_aml.md).

![deployment.png](../images/deployment.png)

## More information

1. [Project InnerEye](https://www.microsoft.com/en-us/research/project/medical-image-analysis/)
1. [Releases](releases.md)
1. [Changelog](CHANGELOG.md)
1. [Testing](testing.md)
1. [How to do pull requests](pull_requests.md)
1. [Contributing](contributing.md)

## Licensing

[MIT License](https://github.com/microsoft/InnerEye-DeepLearning/blob/main/LICENSE)

**You are responsible for the performance, the necessary testing, and if needed any regulatory clearance for
 any of the models produced by this toolbox.**

## Acknowledging usage of Project InnerEye OSS tools

When using Project InnerEye open-source software (OSS) tools, please acknowledge with the following wording:

> This project used Microsoft Research's Project InnerEye open-source software tools ([https://aka.ms/InnerEyeOSS](https://aka.ms/InnerEyeOSS)).

## Contact

If you have any feature requests, or find issues in the code, please create an
[issue on GitHub](https://github.com/microsoft/InnerEye-DeepLearning/issues).

Please send an email to InnerEyeInfo@microsoft.com if you would like further information about this project.

## Publications

Oktay O., Nanavati J., Schwaighofer A., Carter D., Bristow M., Tanno R., Jena R., Barnett G., Noble D., Rimmer Y., Glocker B., O’Hara K., Bishop C., Alvarez-Valle J., Nori A.: Evaluation of Deep Learning to Augment Image-Guided Radiotherapy for Head and Neck and Prostate Cancers. JAMA Netw Open. 2020;3(11):e2027426. [doi:10.1001/jamanetworkopen.2020.27426](https://pubmed.ncbi.nlm.nih.gov/33252691/)

Bannur S., Oktay O., Bernhardt M, Schwaighofer A., Jena R., Nushi B., Wadhwani S., Nori A., Natarajan K., Ashraf S., Alvarez-Valle J., Castro D. C.: Hierarchical Analysis of Visual COVID-19 Features from Chest Radiographs. ICML 2021 Workshop on Interpretable Machine Learning in Healthcare. [https://arxiv.org/abs/2107.06618](https://arxiv.org/abs/2107.06618)

Bernhardt M., Castro D. C., Tanno R., Schwaighofer A., Tezcan K. C., Monteiro M., Bannur S., Lungren M., Nori S., Glocker B., Alvarez-Valle J., Oktay. O: Active label cleaning for improved dataset quality under resource constraints. [https://www.nature.com/articles/s41467-022-28818-3](https://www.nature.com/articles/s41467-022-28818-3). Accompagnying code [InnerEye-DataQuality](https://github.com/microsoft/InnerEye-DeepLearning/blob/1606729c7a16e1bfeb269694314212b6e2737939/InnerEye-DataQuality/README.md)

## Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit [https://cla.opensource.microsoft.com](https://cla.opensource.microsoft.com).

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

## Maintenance

This toolbox is maintained by the [Microsoft Medical Image Analysis team](https://www.microsoft.com/en-us/research/project/medical-image-analysis/).
