# InnerEye-DeepLearning

[![Build Status](https://innereye.visualstudio.com/InnerEye/_apis/build/status/InnerEye-DeepLearning/InnerEye-DeepLearning-PR?branchName=main)](https://innereye.visualstudio.com/InnerEye/_build?definitionId=112&branchName=main)

InnerEye-DeepLearning (IE-DL) is a toolbox for easily training deep learning models on 3D medical images. Simple to run both locally and in the cloud with [AzureML](https://docs.microsoft.com/en-gb/azure/machine-learning/), it allows users to train and run inference on the following:

- Segmentation models.
- Classification and regression models.
- Any PyTorch Lightning model, via a [bring-your-own-model setup](docs/source/md/bring_your_own_model.md).

In addition, this toolbox supports:

- Cross-validation using AzureML, where the models for individual folds are trained in parallel. This is particularly important for the long-running training jobs often seen with medical images.
- Hyperparameter tuning using [Hyperdrive](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-tune-hyperparameters).
- Building ensemble models.
- Easy creation of new models via a configuration-based approach, and inheritance from an existing architecture.

## Documentation

For all documentation, including setup guides and APIs, please refer to the [IE-DL Read the Docs site](https://innereye-deeplearning.readthedocs.io/#).

## Quick Setup

This quick setup assumes you are using a machine running Ubuntu with Git, Git LFS, Conda and Python 3.7+ installed. Please refer to the [setup guide](docs/source/md/environment.md) for more detailed instructions on getting InnerEye set up with other operating systems and installing the above prerequisites.

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

## Full InnerEye Deployment

We offer a companion set of open-sourced tools that help to integrate trained CT segmentation models with clinical
software systems:

- The [InnerEye-Gateway](https://github.com/microsoft/InnerEye-Gateway) is a Windows service running in a DICOM network,
that can route anonymized DICOM images to an inference service.
- The [InnerEye-Inference](https://github.com/microsoft/InnerEye-Inference) component offers a REST API that integrates
with the InnerEye-Gateway, to run inference on InnerEye-DeepLearning models.

Details can be found [here](docs/source/md/deploy_on_aml.md).

![docs/deployment.png](docs/source/images/deployment.png)

## Benefits of InnerEye-DeepLearning

In combiniation with the power of AzureML, InnerEye provides the following benefits:

- **Traceability**: AzureML keeps a full record of all experiments that were executed, including a snapshot of the code. Tags are added to the experiments automatically, that can later help filter and find old experiments.
- **Transparency**: All team members have access to each other's experiments and results.
- **Reproducibility**: Two model training runs using the same code and data will result in exactly the same metrics. All sources of randomness are controlled for.
- **Cost reduction**: Using AzureML, all compute resources (virtual machines, VMs) are requested at the time of starting the training job and freed up at the end. Idle VMs will not incur costs. Azure low priority nodes can be used to further reduce costs (up to 80% cheaper).
- **Scalability**: Large numbers of VMs can be requested easily to cope with a burst in jobs.

Despite the cloud focus, InnerEye is designed to be able to run locally too, which is important for model prototyping, debugging, and in cases where the cloud can't be used. Therefore, if you already have GPU machines available, you will be able to utilize them with the InnerEye toolbox.

## Licensing

[MIT License](/LICENSE)

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

Bernhardt M., Castro D. C., Tanno R., Schwaighofer A., Tezcan K. C., Monteiro M., Bannur S., Lungren M., Nori S., Glocker B., Alvarez-Valle J., Oktay. O: Active label cleaning for improved dataset quality under resource constraints. [https://www.nature.com/articles/s41467-022-28818-3](https://www.nature.com/articles/s41467-022-28818-3). Accompanying code [InnerEye-DataQuality](https://github.com/microsoft/InnerEye-DeepLearning/blob/1606729c7a16e1bfeb269694314212b6e2737939/InnerEye-DataQuality/README.md)

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
