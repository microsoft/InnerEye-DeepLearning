# Lung Segmentation Model

## Purpose

This model is designed to perform segmentation of CT scans of human torsos. It is trained to identify 5 key structures: left lung, right lung, heart, spinalcord and esophagus.

## Download

The lung segmentation model can be downloaded from [this release](https://github.com/microsoft/InnerEye-DeepLearning/releases/tag/v0.8).

## Connected Components

It is possible to apply connected components as a post-processing step, and by default this is performed on the 3 largest structures: both lungs and the heart. To alter this behaviour, update the property `largest_connected_component_foreground_classes` of the Lung class in `InnerEye/ML/configs/segmentation/Lung.py`.

## Model Card

### Model Details

- Organisation: Biomedical Imaging Team at Microsoft Research, Cambridge UK.
- Model date: 31st October 2022.
- Model version: 1.0.
- Model type: ensemble of 3D UNet. Training details are as described in [this paper](https://jamanetwork.com/journals/jamanetworkopen/fullarticle/2773292).
- Training details: 5 fold ensemble model. Trained on the [LCTSC 2017 dataset](https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=24284539) (described in detail below).
- License: The model is released under MIT license as described [here](https://github.com/microsoft/InnerEye-DeepLearning/blob/main/LICENSE).
- Contact: innereyeinfo@microsoft.com.

### Terms of use

Please note that all models provided by InnerEye-DeepLearning are intended for research purposes only. You are responsible for the performance, the necessary testing, and if needed any regulatory clearance for any of the models produced by this toolbox.

### Limitations

The dataset used for training contains only 60 scans, 10 of which are withheld for testing. This limited amount of training data means that the model underperforms on the smaller structures (esophagus and spinalcord) and may not yet generalise well to data samples from outside the dataset.

Furthermore, the dataset description does not contain details on the population of patients used for creating the dataset. Therefore it is not possible to assess whether this model is suitable for use on a target population outside of the dataset.

### Intended Uses

This model is intended for research purposes only. It is intended to be used as a starting-point for more challenging segmentation tasks or training using more thorough and comprehensive segmentation tasks.

### Training Data

This model is trained on the [LCTSC 2017 dataset](https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=24284539). For a detailed description on this data, including the contouring guidelines, see [this page](https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=24284539#242845396723d79f9909442996e4dd0af5e56a30).

The following steps were carrried out to create the dataset used for training this model:

1. Download the DICOM dataset from the above LCTSC 2017 link.
1. Use the [InnerEye-CreateDataset tool](https://github.com/microsoft/InnerEye-CreateDataset) to run the following command on the data:

   ```shell
     .\InnerEye.CreateDataset.Runner.exe dataset --datasetRootDirectory=<path_to_DICOM_data> --niftiDatasetDirectory=lung_nifti --dicomDatasetDirectory=LCTSC --geoNorm 1 1 3 --groundTruthDescendingPriority esophagus spinalcord lung_r lung_l heart
   ```

1. Upload and register NIFTI dataset to Azure by following the [dataset creation](creating_dataset.md) guide.

### Metrics

Metrics for the withheld test data (first 10 scans in the dataset), can be seen in the following table:

| Structure     | count   | DiceNumeric_mean | DiceNumeric_std | DiceNumeric_min | DiceNumeric_max | HausdorffDistance_mm_mean | HausdorffDistance_mm_std | HausdorffDistance_mm_min | HausdorffDistance_mm_max | MeanDistance_mm_mean | MeanDistance_mm_std | MeanDistance_mm_min | MeanDistance_mm_max |
|---------------|---------|------------------|-----------------|-----------------|-----------------|---------------------------|--------------------------|--------------------------|--------------------------|----------------------|---------------------|---------------------|---------------------|
| lung_l        | 10      | 0.984            | 0.009           | 0.958           | 0.990           | 11.642                    | 4.868                    | 6.558                    | 19.221                   | 0.344                | 0.266               | 0.167               | 1.027               |
| lung_r        | 10      | 0.983            | 0.009           | 0.960           | 0.991           | 10.764                    | 3.307                    | 6.325                    | 16.156                   | 0.345                | 0.200               | 0.160               | 0.797               |
| spinalcord    | 10      | 0.860            | 0.050           | 0.756           | 0.912           | 27.213                    | 22.015                   | 12.000                   | 81.398                   | 1.750                | 2.167               | 0.552               | 7.209               |
| heart         | 10      | 0.935            | 0.015           | 0.908           | 0.953           | 17.550                    | 14.796                   | 9.000                    | 17.550                   | 2.022                | 0.661               | 1.456               | 3.299               |
| esophagus     | 10      | 0.728            | 0.128           | 0.509           | 0.891           | 23.503                    | 25.679                   | 6.173                    | 72.008                   | 3.207                | 4.333               | 0.409               | 13.991              |
|               |         |                  |                 |                 |                 |                           |                          |                          |                          |                      |                     |                     |                     |
