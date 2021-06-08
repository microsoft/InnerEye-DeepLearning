# Sample Tasks

This document contains two sample tasks for the classification and segmentation pipelines. 

The document will walk through the steps in [Training Steps](building_models.md), but with specific examples for each task.
Before trying tp train these models, you should have followed steps to set up an [environment](environment.md) and [AzureML](setting_up_aml.md)

## Sample classification task: Glaucoma Detection on OCT volumes
 
This example is based on the paper [A feature agnostic approach for glaucoma detection in OCT volumes](https://arxiv.org/pdf/1807.04855v3.pdf).

### Downloading and preparing the dataset
1. The dataset is available [here](https://zenodo.org/record/1481223#.Xs-ehzPiuM_) <sup>[[1]](#1)</sup>.

1. After downloading and extracting the zip file, run the [create_glaucoma_dataset_csv.py](https://github.com/microsoft/InnerEye-DeepLearning/blob/main/InnerEye/Scripts/create_glaucoma_dataset_csv.py)
 script on the extracted folder.
```
python create_dataset_csv.py /path/to/extracted/folder
```
This will convert the dataset to csv form and create a file `dataset.csv`.
 
1.  Upload this folder (with the images and `dataset.csv`) to Azure Blob Storage. For details on creating a storage account, 
see [Setting up AzureML](setting_up_aml.md#step-4-create-a-storage-account-for-your-datasets). The dataset should go
into a container called `datasets`, with a folder name of your choice (`name_of_your_dataset_on_azure` in the 
description below).

### Setting up training

You have two options for running the Glaucoma model:
- You can directly work on a fork of the InnerEye repository. Create a config file `InnerEye/ML/configs/MyGlaucoma.py`
 which extends the GlaucomaPublic class like this:
```python
from InnerEye.ML.configs.classification.GlaucomaPublic import GlaucomaPublic

class MyGlaucomaModel(GlaucomaPublic):
    def __init__(self) -> None:
        super().__init__()
        self.azure_dataset_id="name_of_your_dataset_on_azure"
``` 
The value for `self.azure_dataset_id` should match the dataset upload location, called 
`name_of_your_dataset_on_azure` above. 

Once that config is in place, you can start training in AzureML via
```
python InnerEye/ML/runner.py --model=MyGlaucomaModel --azureml=True
```

As an alternative to working with a fork of the repository, you can use InnerEye-DeepLearning via a submodule. 
Please check [here](innereye_as_submodule.md) for details.


## Sample segmentation task: Segmentation of Lung CT
 
This example is based on the [Lung CT Segmentation Challenge 2017](https://wiki.cancerimagingarchive.net/display/Public/Lung+CT+Segmentation+Challenge+2017) <sup>[[2]](#2)</sup>.

### Downloading and preparing the dataset

1. The dataset <sup>[[3]](#3)[[4]](#4)</sup> can be downloaded [here](https://wiki.cancerimagingarchive.net/display/Public/Lung+CT+Segmentation+Challenge+2017#021ca3c9a0724b0d9df784f1699d35e2).
1. The next step is to convert the dataset from DICOM-RT to NIFTI. Before this, place the downloaded dataset in another
 parent folder, which we will call `datasets`. This file structure is expected by the converison tool.
1. Use the [InnerEye-CreateDataset](https://github.com/microsoft/InnerEye-createdataset) to create a NIFTI dataset
 from the downloaded (DICOM) files.
After installing the tool, run
```batch
InnerEye.CreateDataset.Runner.exe dataset --datasetRootDirectory=<path to the 'datasets' folder> --niftiDatasetDirectory=<output folder name for converted dataset> --dicomDatasetDirectory=<name of downloaded folder inside 'datasets'> --geoNorm 1;1;3
```
Now, you should have another folder under `datasets` with the converted Nifti files.
The `geonorm` tag tells the tool to normalize the voxel sizes during conversion.
1.  Upload this folder (with the images and dataset.csv) to Azure Blob Storage. For details on creating a storage account, 
see [Setting up AzureML](setting_up_aml.md#step-4-create-a-storage-account-for-your-datasets). All files should go
into a folder in the `datasets` container, for example `my_lung_dataset`.
1. You can then modify the example model configuration in [Lung.py](../InnerEye/ML/configs/segmentation/Lung.py), and
add the `azure_dataset_id` field, so that it looks like:
```python
class Lung(SegmentationModelBase):
    def __init__(self, **kwargs: Any) -> None:
        fg_classes = ["spinalcord", "lung_r", "lung_l", "heart", "esophagus"]
        fg_display_names = ["SpinalCord", "Lung_R", "Lung_L", "Heart", "Esophagus"]
        super().__init__(
            azure_dataset_id="my_lung_dataset",
            architecture="UNet3D",
            feature_channels=[32],
...
```
If you are using InnerEye as a submodule, please add a new configuration that is a subclass of `Lung`, and set
the `azure_dataset_id` field there, as described for the Glaucoma model [here](innereye_as_submodule.md).
1. You can now run the following command to start a job on AzureML:
```
python InnerEye/ML/runner.py --azureml=True --model=Lung
```
See [Model Training](building_models.md) for details on training outputs, resuming training, testing models and model ensembles.
 
### References

<a id="1">[1]</a>
Ishikawa, Hiroshi. (2018). OCT volumes for glaucoma detection (Version 1.0.0) [Data set]. Zenodo. http://doi.org/10.5281/zenodo.1481223  

<a id="2">[2]</a>
Yang, J. , Veeraraghavan, H. , Armato, S. G., Farahani, K. , Kirby, J. S., Kalpathy-Kramer, J. , van Elmpt, W. , Dekker, A. , Han, X. , Feng, X. , Aljabar, P. , Oliveira, B. , van der Heyden, B. , Zamdborg, L. , Lam, D. , Gooding, M. and Sharp, G. C. (2018), 
Autosegmentation for thoracic radiation treatment planning: A grand challenge at AAPM 2017. Med. Phys.. . [doi:10.1002/mp.13141](https://doi.org/10.1002/mp.13141)  

<a id="3">[3]</a>
Yang, Jinzhong; Sharp, Greg; Veeraraghavan, Harini ; van Elmpt, Wouter ; Dekker, Andre; Lustberg, Tim; Gooding, Mark. (2017). 
Data from Lung CT Segmentation Challenge. The Cancer Imaging Archive. http://doi.org/10.7937/K9/TCIA.2017.3r3fvz08  

<a id="4">[4]</a>
Clark K, Vendt B, Smith K, Freymann J, Kirby J, Koppel P, Moore S, Phillips S, Maffitt D, Pringle M, Tarbox L, Prior F. 
The Cancer Imaging Archive (TCIA): Maintaining and Operating a Public Information Repository, Journal of Digital Imaging, Volume 26, Number 6, December, 2013, pp 1045-1057. ([paper](http://link.springer.com/article/10.1007%2Fs10278-013-9622-7))
