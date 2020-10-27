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
 
Once training in AzureML is done, the models can be deployed from within AzureML or via 
[Azure Stack Hub](https://azure.microsoft.com/en-us/products/azure-stack/hub/).


## Getting started

We recommend using our toolbox with Linux or with the Windows Subsystem for Linux (WSL2). Much of the core 
functionality works fine on Windows, but PyTorch's full feature set is only available on Linux. Read [more about
WSL here](docs/WSL.md).

Clone the repository into a subfolder of the current directory:
```shell script
git clone https://github.com/microsoft/InnerEye-DeepLearning
cd InnerEye-DeepLearning
git lfs install
git lfs pull
```
After that, you need to set up your Python environment:
- Install `conda` or `miniconda` for your operating system. 
- Create a Conda environment from the `environment.yml` file in the repository root, and activate it:
```shell script
conda env create --file environment.yml
conda activate InnerEye
``` 
- If environment creation fails with odd error messages on a Windows machine, please [continue here](docs/WSL.md).

Now try to run the HelloWorld segmentation model - that's a very simple model that will train for 2 epochs on any
machine, no GPU required. You need to set the `PYTHONPATH` environment variable to point to the repository root first. 
Assuming that your current directory is the repository root folder, on Linux `bash` that is: 
```shell script
export PYTHONPATH=`pwd`
python InnerEye/ML/runner.py --model=HelloWorld
```
(Note the "backtick" around the `pwd` command, this is not a standard single quote!)

On Windows:
```shell script
set PYTHONPATH=%cd%
python InnerEye/ML/runner.py --model=HelloWorld
```

If that works: Congratulations! You have successfully built your first model using the InnerEye toolbox.

If it fails, please check the 
[troubleshooting page on the Wiki](https://github.com/microsoft/InnerEye-DeepLearning/wiki/Issues-with-code-setup-and-the-HelloWorld-model).

Further detailed instructions, including setup in Azure, are here:
1. [Setting up your environment](docs/environment.md)
1. [Training a Hello World segmentation model](docs/hello_world_model.md)
1. [Setting up Azure Machine Learning](docs/setting_up_aml.md)
1. [Creating a dataset](docs/creating_dataset.md)
1. [Building models in Azure ML](docs/building_models.md)
1. [Sample Segmentation and Classification tasks](docs/sample_tasks.md)
1. [Debugging and monitoring models](docs/debugging_and_monitoring.md)
1. [Model diagnostics](docs/model_diagnostics.md)

## More information

1. [Project InnerEye](https://www.microsoft.com/en-us/research/project/medical-image-analysis/)
1. [Testing](docs/testing.md)
1. [How to do pull requests](docs/pull_requests.md)
1. [Contributing](docs/contributing.md)

## Licensing

[MIT License](LICENSE)

**You are responsible for the performance, the necessary testing, and if needed any regulatory clearance for
 any of the models produced by this toolbox.**

## Contact

Please send an email to InnerEyeInfo@microsoft.com if you would like further information about this project.

If you have any feature requests, or find issues in the code, please create an 
[issue on GitHub](https://github.com/microsoft/InnerEye-DeepLearning/issues).

If you are interested in using the InnerEye Deep Learning Toolkit to develop your own products and services,
please email InnerEyeCommercial@microsoft.com. We can also provide input on using the toolbox with 
[Azure Stack Hub](https://azure.microsoft.com/en-us/products/azure-stack/hub/), a hybrid cloud solution
that allows for on-premise medical image analysis that complies with data handling regulations.


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


## Credits

This toolbox is maintained by the 
[Microsoft InnerEye team](https://www.microsoft.com/en-us/research/project/medical-image-analysis/), 
and has received valuable contributions from a number
of people outside our team. We would like to thank in particular our interns, 
[Yao Quin](http://cseweb.ucsd.edu/~yaq007/), [Zoe Landgraf](https://www.linkedin.com/in/zoe-landgraf-a2212293),
[Padmaja Jonnalagedda](https://www.linkedin.com/in/jspadmaja/),
[Mathias Perslev](https://github.com/perslev), as well as the AI Residents 
[Patricia Gillespie](https://www.microsoft.com/en-us/research/people/t-pagill/) and
[Guilherme Ilunga](https://gilunga.github.io/).
