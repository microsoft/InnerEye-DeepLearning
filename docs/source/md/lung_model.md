# Lung Segmentation Model

## Purpose

This model is designed to perform segmentation of CT scans of human torsos. It is trained to identify 5 key structures: left lung, right lung, heart, spinalcord and esophagus.

## Terms of Use

This model is intended for research purposes only. You are responsible for the performance, the necessary testing, and if needed any regulatory clearance for any of the models produced by this toolbox.

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

### Limitations

The dataset used for training contains only 60 scans, 10 of which are withheld for testing. This limited amount of training data means that the model may not yet generalise well to data samples from outside the dataset and underperforms on the smaller structures (esophagus and spinalcord).

### Intended Uses

This model is intended for research purposes only. It is intended to be used as a starting-point for more challenging segmentation tasks or training using more thorough and comprehensive segmentation tasks.

### Training Data

This model is trained on the [LCTSC 2017 dataset](https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=24284539). For a detailed description on this data, including the contouring guidelines, see [this page](https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=24284539#242845396723d79f9909442996e4dd0af5e56a30).

To create the dataset used for training, we utilised the [InnerEye-CreateDataset tool](https://github.com/microsoft/InnerEye-CreateDataset). The dataset was processed using the following command:

```shell
  .\InnerEye.CreateDataset.Runner.exe dataset --datasetRootDirectory=C:\Users\a-phessey\Datasets\lung_dicom --niftiDatasetDirectory=lung_nifti --dicomDatasetDirectory=LCTSC --geoNorm 1 1 3 --groundTruthDescendingPriority esophagus spinalcord lung_r lung_l heart
```

### Metrics

Metrics for the withheld test data (first 10 scans in the dataset), can be seen in the following table:

| Structure     | count   | DiceNumeric_mean | DiceNumeric_std | DiceNumeric_min | DiceNumeric_max | HausdorffDistance_mm_mean | HausdorffDistance_mm_std | HausdorffDistance_mm_min | HausdorffDistance_mm_max | MeanDistance_mm_mean | MeanDistance_mm_std | MeanDistance_mm_min | MeanDistance_mm_max |
|---------------|---------|------------------|-----------------|-----------------|-----------------|---------------------------|--------------------------|--------------------------|--------------------------|----------------------|---------------------|---------------------|---------------------|
| lung_r        | 10      | 0.982            | 0.010           | 0.957           | 0.990           | 11.107                    | 3.382                    | 6.782                    | 16.734                   | 0.366                | 0.213               | 0.177               | 0.902               |
| lung_l        | 10      | 0.980            | 0.009           | 0.958           | 0.989           | 11.457                    | 3.389                    | 8.246                    | 18.028                   | 0.359                | 0.261               | 0.191               | 1.040               |
| heart         | 10      | 0.953            | 0.015           | 0.902           | 0.953           | 13.384                    | 3.101                    | 8.186                    | 17.975                   | 1.983                | 0.574               | 1.519               | 3.301               |
| spinalcord    | 10      | 0.848            | 0.044           | 0.763           | 0.894           | 29.362                    | 21.370                   | 10.050                   | 78.473                   | 1.888                | 2.029               | 0.631               | 6.710               |
| esophagus     | 10      | 0.701            | 0.108           | 0.489           | 0.809           | 27.945                    | 27.767                   | 12.728                   | 80.214                   | 3.991                | 5.735               | 1.108               | 18.489              |
|               |         |                  |                 |                 |                 |                           |                          |                          |                          |                      |                     |                     |                     |
