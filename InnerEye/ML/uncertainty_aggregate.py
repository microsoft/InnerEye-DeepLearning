import nibabel as nib
import numpy as np
from sklearn.ensemble import RandomForestRegressor 
import pandas as pd
from radiomics import featureextractor
from pathlib import Path
#from InnerEye.ML.utils.split_dataset import DatasetSplits
from sklearn.model_selection import GroupKFold
from sklearn.utils import shuffle
from sklearn import metrics
import scipy
import matplotlib.pyplot as plt

import SimpleITK as sitk
from InnerEye.Common.common_util import BEST_EPOCH_FOLDER_NAME, METRICS_AGGREGATES_FILE, ModelProcessing, \
    SUBJECT_METRICS_FILE_NAME, get_best_epoch_results_path, is_linux, logging_section
from InnerEye.ML.common import ModelExecutionMode
import pickle

# think about what the inputs will be:
# output trained random forest model 


# possibility to add extra functionality to simulate poor segmentations 

def trainRandomForest(x,y):
    # requires dataset: subjects, groups
    mdl = RandomForestRegressor(n_estimators=10,criterion='mse',max_depth=None)
    mdl.fit(x, y)
    return mdl

'''
def splitKfolds():
    # maybe we don't need this... 
    # we only need this if we want to report a value on how good the random forest is at predicting the dice score
    # i.e. how trustworthy is our uncertainty metric
    subject_ids = features['subjects'].values
    groups =  features['groupID'].values
    k_folds = GroupKFold(n_splits=5)
    return k_folds.split(subject_ids, groups=groups)
'''

def UncertaintyMapFeatures(dice,outputs_folder,subjects):
    '''extract PyRadiomics features from the uncertainty map'''
    features = pd.DataFrame()
    for i,subject in enumerate(subjects):
        subject_dir = outputs_folder / f"{int(patient_id):03d}"
        extractor = featureextractor.RadiomicsFeatureExtractor(imageType='Original',binWidth=25)
        result=pd.Series(extractor.execute(
            str(subject_dir / 'uncertainty.nii.gz'), 
            str(subject_dir / 'segmentation.nii.gz')))
        featureVector=df[subj_i]
        featureVector=featureVector.append(result)
        featureVector.name=subj

        features = features.join(featureVector, how='outer')

    # remove unused features
    return features

def PostProcessFeatures(features):
    '''drop unused pyRadiomics features
    TO DO: pyradiomics website, create .yml file which selects the desired features'''

    features = features.T

    features=features.drop(['diagnostics_Versions_PyRadiomics',
    'diagnostics_Versions_Numpy', 'diagnostics_Versions_SimpleITK',
    'diagnostics_Versions_PyWavelet', 'diagnostics_Versions_Python',
    'diagnostics_Configuration_Settings',
    'diagnostics_Configuration_EnabledImageTypes',
    'diagnostics_Image-original_Hash',
    'diagnostics_Image-original_Dimensionality',
    'diagnostics_Image-original_Spacing', 'diagnostics_Image-original_Size',
    'diagnostics_Image-original_Mean', 'diagnostics_Image-original_Minimum',
    'diagnostics_Image-original_Maximum', 'diagnostics_Mask-original_Hash',
    'diagnostics_Mask-original_Spacing', 'diagnostics_Mask-original_Size',
    'diagnostics_Mask-original_BoundingBox',
    'diagnostics_Mask-original_VoxelNum',
    'diagnostics_Mask-original_VolumeNum',
    'diagnostics_Mask-original_CenterOfMassIndex',
    'diagnostics_Mask-original_CenterOfMass','original_ngtdm_Busyness',
    'original_ngtdm_Coarseness', 'original_ngtdm_Complexity',
    'original_ngtdm_Contrast', 'original_ngtdm_Strength'],1)

    # shuffle rows - currently ordered by accending Dice score
    #features = shuffle(features,random_state=0)
    return features


def load_uncertainty_from_nifti(fname):
    img = sitk.ReadImage(fname)
    return sitk.GetArrayFromImage(img)

def average_dice_across_structures(subjects,metrics):
    avgDice=[]
    for subject_id in subjects:
        avgDice.append(np.mean(metrics.Dice[metrics.Patient==subject_id]))
    return avgDice

def UncertaintyAggregate(config):
    # iterate through uncertainty maps
    ###### need to correct this: file path is different for single and ensemble models ####
    outputs_folder = config.outputs_folder
    metrics_df = pd.read_csv(outputs_folder / 'best_validation_epoch/Test/metrics.csv')
    subjects = metrics_df.Patient.unique()

    # why are two dataframes needed???
    dice = pd.DataFrame(average_dice_across_structures(subjects,metrics_df))
    features = UncertaintyMapFeatures(dice,outputs_folder,subjects)
    features = PostProcessFeatures(features)
    # cross-validation to estimate model performance
    
    # train model with all data
    mdl = trainRandomForest(features.drop(['dice']),features.dice)

    # save model to disk - save in outputs file? Where will it be seen by the inference model?
    fname = 'uncertainty_metric_model'
    pickle.dump(mdl, open(fname, 'wb'))

    # could run cross-validation and save some metrics in the outputs file to report how well the model is about the predict dice
    return 