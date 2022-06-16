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

# think about what the inputs will be:
# output trained random forest model 


class uncertainty_aggregate():

    def __init__():


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

    def UncertaintyMapFeatures():
        '''extract PyRadiomics features from the uncertainty map'''
        for subject in subjects:
            # extract pyradiomics features

        # remove unused features
        return features

    def load_uncertainty_from_nifti(fname):
        img = sitk.ReadImage(fname)
        return sitk.GetArrayFromImage(img)

    def average_dice_across_structures(subjects,metrics):
        avgDice=[]
        for subj in subjects:
            avgDice.append(np.mean(metrics.Dice[metrics.Patient==subj]))
        return avgDice

    def run_and_save():
        # iterate through uncertainty maps
        outputdir = 
        metrics = 

        dice=[]
        for subject in subjects:
            uncertainty = load_uncertainty_from_nifti(subject)
            features = UncertaintyMapFeatures(testSet)
            dice.append(average_dice_across_structures())

        # cross-validation to estimate model performance
        
        # train model with all data
        mdl = trainRandomForest(features,dice)

        # save model to disk - save in outputs file? Where will it be seen by the inference model?
        fname = 'uncertainty_metric_model'
        pickle.dump(mdl, open(fname, 'wb'))

        # could run cross-validation and save some metrics in the outputs file to report how well the model is about the predict dice

        return 