# Model Diagnostics

The InnerEye toolbox has extensive reporting about the model building process, as well as the performance
of the final model. Our goal is to provide as much insight as possible about the critical steps (and 
pitfalls) of building a model.

## Patch sampling for segmentation models

When building a segmentation model, one of the crucial steps is how equally-shaped crops are taken from
the raw medical image, that are later fed into the model training. An outline of that process is
given [here](https://github.com/microsoft/InnerEye-DeepLearning/wiki/Adjusting-and-tuning-a-segmentation-model).

At the start of training, the toolbox inspects the first 10 images of the training set. For each of them,
1000 random crops are drawn at random, similar to how they would be drawn during training. From that, a
heatmap is constructed, where each voxel value contains how often that specific voxels was actually contained
in the random crop (a value between 0 and 1000). The heatmap is stored as a Nifti file, alongside the 
original scan, in folder `outputs/patch_sampling/`. When running inside AzureML, navigate to the 
"Outputs" tab, and go to the folder (see screenshot below).

In addition, for each patient, 3 thumbnail images are generated, that overlay the heatmap on top of the
scan. Dark red indicates voxels that are sampled very often, transparent red indicates voxels that are used
infrequently.

Example thumbnail when viewed in the AzureML UI:
![](screenshot_azureml_patch_sampling.png)
