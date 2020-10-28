# Dataset Creation
This document describes the dataset formats used by InnerEye for segmentation and classification tasks. After creating 
the dataset, upload it to AzureML blob storage (as described in the 
[AzureML documentation](setting_up_aml.md#step-4-create-a-storage-account-for-your-datasets))

## Segmentation Datasets
This section walks through the process of creating a dataset in the format expected by the InnerEye package. 
However, if your dataset is in DICOM-RT format, you should instead use the 
[InnerEye-CreateDataset](https://github.com/microsoft/InnerEye-CreateDataset) tool.
After creating the dataset, you can also [analyze](#analysing-segmentation-datasets) the structures in it.

Segmentation datasets should have the input scans and ground truth segmentations in Nifti format.

InnerEye expects segmentation datasets to have the following structure:
 * Each subject has one or more scans, and one or more segmentation masks. There should be one segmentation mask for
   each ground truth structure (anatomical structure that the model should segment)
 * For convenience, scans and ground truth masks for different subjects can live in separate folders, but that's not a must.
 * Inside the root folder for the dataset, there should be a file `dataset.csv`, containing the following fields 
 at minimum:
    * `subject`: A unique positive integer assigned to every patient
    * `channel`: The imaging channel or ground truth structure described by this row. 
    * `filePath`: Path to the file for this scan or structure. We support nifti (nii, nii.gz), numpy (npy, npz) and hdf5(h5).
        * For HDF5 path suffix with | separator
            * For images <path>|<dataset_name>|<channel index>
            * For segmentation binary <path>|<dataset_name>|<channel index>
            * For segmentation multimap <path>|<dataset_name>|<channel index>|<multimap value>
                * Multimaps are encoded as 0=background and integers for each class.
            * The expected dimensions: (channel, Z, Y, X)
        * For numpy or nifti just the expected format is just the path to the files. 
            * For images can be encoded as float32 with dimensions (X, Y, Z)
            * For segmentations should be encoded as binary masks with dimensions (X, Y, Z) 
    
    Additional supported fields include `acquisition_date`, `institutionId`, `seriesID` and `tags` (meant for miscellaneous labels). 

For example, for a CT dataset with two structures `heart` and `lung` to be segmented, the dataset folder
could look like:

```
root_folder
├──dataset.csv
├──subjectID1/
│  ├── ct.nii.gz
│  ├── heart.nii.gz
│  └── lung.nii.gz
├──subjectID2/
|  ├── ct.nii.gz
|  ├── heart.nii.gz
|  ├── lung.nii.gz
├──...
```

The `dataset.csv` for this dataset would look like:
```
subject,filePath,channel
1,subjectID1/ct.nii.gz,ct
1,subjectID1/heart.nii.gz,structure1
1,subjectID1/lung.nii.gz,structure2
2,subjectID2/ct.nii.gz,ct
2,subjectID2/heart.nii.gz,structure1
2,subjectID2/lung.nii.gz,structure2
```

The images must adhere to these constraints:
* All images (across all subjects) must have already undergone geometric normalization, i.e., all images must have
approximately the same voxel size.
* All images for a particular subject must have the same dimensions. In the above example, if `subjectID1/ct.nii.gz`
has size 200 x 256 x 256, then `subjectID1/heart.nii.gz` and `subjectID1/lung.nii.gz` must have exactly the
same dimensions.
* It is usually not required that images for different subjects have the same dimensions.

All these constraints are automatically checked and guaranteed if the raw data is in DICOM format and you are using
the [InnerEye-CreateDataset](https://github.com/microsoft/InnerEye-CreateDataset) tool to convert them to Nifti
format. Geometric normalization can also be turned on as a pre-processing step.
 
For the above dataset structure for heart and lung segmentation, you would then create a model configuration that 
contains at least the following fields:
```python
class HeartLungModel(SegmentationModelBase):
    def __init__(self) -> None:
        super().__init__(
            azure_dataset_id="folder_name_in_azure",
            local_dataset="/home/me/folder_name_on_local_vm",
            image_channels=["ct"],
            ground_truth_ids=["heart", "lung"],
            # Segmentation architecture
            architecture="UNet3D",
            feature_channels=[32],
            # Size of patches that are used for training, as (z, y, x) tuple 
            crop_size=(64, 224, 224),
            # Reduce this if you see GPU out of memory errors
            train_batch_size=8,
            # Size of patches that are used when evaluating the model
            test_crop_size=(128, 512, 512),
            inference_stride_size=(64, 256, 256),
            # Use CT Window and Level as image pre-processing
            norm_method=PhotometricNormalizationMethod.CtWindow,
            level=40,
            window=400,
            # Learning rate settings
            l_rate=1e-3,
            min_l_rate=1e-5,
            l_rate_polynomial_gamma=0.9,
            num_epochs=120,
            )
```
The `local_dataset` field is required if you want to run the InnerEye toolbox on your own VM, and you want to consume
the dataset from local storage. If you want to run the InnerEye toolbox inside of AzureML, you need to supply the
`azure_dataset_id`, pointing to a folder in Azure blob storage. This folder should reside in the `datasets` container
in the storage account that you designated for storing your datasets, see [the setup instructions](setting_up_aml.md).


#### Analyzing segmentation datasets

Once you have created your Azure dataset, either by the process described here or with the CreateDataset tool, 
you may want to analyze it in order to detect images and structures that are outliers
with respect to a number of statistics, and which therefore may be erroneous or unsuitable for your application. 
This can be done using the analyze command provided by 
[InnerEye-CreateDataset](https://github.com/microsoft/InnerEye-CreateDataset).


## Classification Datasets

Classification datasets should have a `dataset.csv` and a folder containing the image files. The `dataset.csv` should 
have at least the following fields:
 * subject: The subject ID, a unique positive integer assigned to every image
 * path: Path to the image file for this subject
 * value: For classification, a (binary) ground truth label. For regression, a scalar value.

These, and other fields which can be added to dataset.csv are described in the examples below.

For each entry (subject ID, label value, etc) needed to construct a single input sample, the entry value is read
from the channels and columns specified for that entry. 

#### A simple example

Let's look at how to construct a `dataset.csv` (and changes we will need to make to the model config file in parallel):

```
SubjectID, FilePath, Label 
1, images/image1.npy, True
2, images/image2.npy, False
```

This is the simplest `dataset.csv` possible. It has two images with subject IDs `1` and `2`, stored at `images/images1.npy`
and `images/images2.npy`. This dataset is a classification dataset, since the label values are binary. 

To use this `dataset.csv`, we need to make some additions to the model config. We will use the `GlaucomaPublicExt` config from 
the [sample tasks](sample_tasks.md#creating-the-classification-model-configuration)
in this example. The class should now resemble:

```python
class GlaucomaPublicExt(GlaucomaPublic):
    def __init__(self) -> None:
        super().__init__(azure_dataset_id="name_of_your_dataset_on_azure",
                         subject_column="SubjectID",
                         image_file_column="FilePath",
                         label_value_column="Label")
``` 

The parameters `subject_column`, `channel_column`, `image_file_column` and `label_value_column` tell InnerEye 
what columns in the csv contain the subject identifiers, channel names, image file paths and labels.
 
NOTE: If any of the `*_column` parameters are not specified, InnerEye will look for these entries under the default column names
if default names exist. See the CSV headers in [csv_util.py](/InnerEye/ML/utils/csv_util.py) for all the defaults. 

#### Using channels in dataset.csv
Channels are fields in `dataset.csv` which can be used to filter rows. They are typically used when there are multiple 
images or labels per subject (for example, if multiple images were taken across a period of time for each subject). 

A slightly more complex `dataset.csv` would be the following:

```
SubjectID, Channel, FilePath, Label
1, image_feature_1, images/image_1_feature_1.npy,
1, image_feature_2, images/image_1_feature_2.npy,
1, label, , True
2, image_feature_1, images/image_2_feature_1.npy
2, image_feature_2, images/image_2_feature_2.npy
2, label, , False
```

The config file would be

```python
class GlaucomaPublicExt(GlaucomaPublic):
    def __init__(self) -> None:
        super().__init__(azure_dataset_id="name_of_your_dataset_on_azure",
                         subject_column="SubjectID",
                         channel_column="Channel",
                         image_channels=["image_feature_1", "image_feature_2"],
                         image_file_column="FilePath",
                         label_channels=["label"],
                         label_value_column="Label")
``` 

The added parameters `image_channels` and `label_channels` tell InnerEye to search for image file paths for each subject
in rows labelled with `image_feature_1` or `image_feature_2` and for label values in the rows labelled with `label`. 
Thus, in this dataset, each sample will have 2 image features (read from rows with `Channel` set to `image_feature_1` 
and `image_feature_2`) and the associated label (read from the row with `Channel` set to `label`).

NOTE: There are no defaults for the `*_channels` parameters, so these must be set as parameters.

#### Recognized columns in dataset.csv and filtering based on channels
Other recognized fields, apart from subject, channel, file path and label are numerical features and categorical features. 
These are extra scalar and categorical values to be used as model input. 

Any *unrecognized* columns (any column which is both not described in the model config and has no default) 
will be converted to a dict of key-value pairs and stored in a object of type `GeneralSampleMetadata` in the sample.

```
SubjectID, Channel, FilePath, Label, Tag, weight, class
1, image_time_1, images/image_1_time_1.npy, True, , ,
1, image_time_2, images/image_1_time_2.npy, False, , ,
1, scalar, , , , 0.5,
1, categorical, , , , , 2
1, tags, , ,foo, ,
2, image_time_1, images/image_2_time_1.npy, True, , ,
2, image_time_2, images/image_2_time_2.npy, True, , ,
2, tags, , , bar, ,
1, scalar, , , , 0.3,
1, categorical, , , , , 4
```

```python
class GlaucomaPublicExt(GlaucomaPublic):
    def __init__(self) -> None:
        super().__init__(azure_dataset_id="name_of_your_dataset_on_azure",
                         subject_column="SubjectID",
                         channel_column="Channel",
                         image_channels=["image_time_1", "image_time_2"],
                         image_file_column="FilePath",
                         label_channels=["image_time_2"],
                         label_value_column="Label",
                         non_image_feature_channels=["scalar"],
                         numerical_columns=["weight"],
                         categorical_columns="class")
``` 

In this example, `weight` is a scalar feature read from the csv, and `class` is a categorical feature. The extra field 
`Tag` is not a recognized field, and so the dataloader will return the tags in the form of key:value pairs for each sample.

**Filtering on channels**: This example also shows why filtering values by channel is useful: In this example, each subject has 2 images taken at 
 different times with different label values. By using `label_channels=["image_time_2"]`, we can use the label associated with
 the second image for all subjects.
 