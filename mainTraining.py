#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 12 20:44:38 2018

@author: michaelyin1994
"""
import numpy as np
import pandas as pd 
import gc
#warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
from matplotlib import rcParams
import seaborn as sns
from scipy.stats import ranksums
from WeaponLib import timefn
from WeaponLib import basic_feature_report
from WeaponLib import replace_inf_with_nan
from WeaponLib import drop_most_empty_features
from WeaponLib import FeatureSelection
from WeaponLib import RandomSearchCVLightGBM
from WeaponLib import RandomSearchCVXGBoost
from WeaponLib import load_data
from WeaponLib import Stacking
from main_table import run_data_main_app
np.random.seed(0)
sns.set(style="ticks", font_scale=1.2, palette='deep', color_codes=True)
###############################################################################
@timefn
def run_main_data(nrows=None, enable=False):
    if enable == True:
        run_data_main_app(nrows=nrows)
    else:
        pass

@timefn
def combine_data(nrows=None):
    appTrain = pd.read_csv("..//TrainTestData//appTrain.csv", nrows=nrows)
    appTest = pd.read_csv("..//TrainTestData//appTest.csv", nrows=nrows)
    appTrain = replace_inf_with_nan(appTrain)
    appTest = replace_inf_with_nan(appTest)
    return appTrain, appTest

if __name__ == "__main__":
    run_main_data()
    appTrain, appTest = combine_data(nrows=20000)
    app = pd.concat([appTrain, appTest], axis=0, ignore_index=True)
    
    # LightGBM based feature selection
    clf = FeatureSelection(appTrain.drop(["SK_ID_CURR", "TARGET"], axis=1), appTrain["TARGET"], verbose=False)
    clf.lgb_feature_importance(n_folds=5)
    featureImportance = clf._importance

    medianImportance = featureImportance["lgb"].median()
    dropList = list(featureImportance["featureName"][featureImportance["lgb"]<0.1*medianImportance])
    app.drop(dropList, axis=1, inplace=True)
    
    # New training data and testing data
    appTrain = app[~app["TARGET"].isnull()]
    appTest = app[app["TARGET"].isnull()]
    appTest.drop(["TARGET"], axis=1, inplace=True)
    del app
    gc.collect()
    '''
    baseModelPath = ["..//TrainedModel//xgboost.pkl", "..//TrainedModel//LightGBM.pkl"]
    clf = Stacking(dataTrain=appTrain.drop(["SK_ID_CURR", "TARGET"], axis=1).values, dataTrainLabel=appTrain["TARGET"].values,
                   dataToPredict=appTest.drop(["SK_ID_CURR"], axis=1).values, baseModelPath=baseModelPath, folds=5, randomState=2018)
    appTest["TARGET"] = clf.fit_predict()
    submission = appTest[['SK_ID_CURR', 'TARGET']]
    submission.to_csv("submission_stacking.csv", index=False)
    '''
    # Search XGBoost model
    paramGrid ={'n_estimators': list(range(7000, 10000, 10)),
            'learning_rate': np.arange(0.001, 0.2, 0.01),
            'num_leaves': list(range(15, 50, 1)),
            'subsample': np.arange(0.8, 1, 0.01),
            'max_depth':list(range(5, 15)),
            'colsample_bytree':np.arange(0.00001, 1, 0.01),
            'colsample_bylevel': np.arange(0.5, 0.8, 0.01),
            }
    
    clf = RandomSearchCVXGBoost(dataTrain=appTrain.drop(["SK_ID_CURR", "TARGET"], axis=1), dataTrainLabel=appTrain["TARGET"], 
                                 n_estimators=20, paramGrid=paramGrid)
    clf.random_search_cv(n_folds=5, random_state=35)
    importance, report = clf.get_score()
    bestParam = clf._bestParam
    appTest["TARGET"] = clf.refit(appTest.drop(["SK_ID_CURR"], axis=1).values)
    submission = appTest[['SK_ID_CURR', 'TARGET']]
    submission.to_csv("submission_lightgbm.csv", index=False)
#    
#    # Search LightGBM model
#    paramGrid ={'n_estimators': list(range(7000, 10000, 10)),
#                'learning_rate': np.arange(0.001, 0.2, 0.01),
#                'num_leaves': list(range(15, 50, 1)),
#                'subsample': np.arange(0.8, 1, 0.01),
#                'max_depth':list(range(5, 15)),
#                'colsample_bytree':np.arange(0.00001, 1, 0.01),
#                'reg_alpha': np.arange(0.00001, 0.6, 0.01),
#                'reg_lambda': np.arange(0.00001, 0.05, 0.001),
#                }
#    
#    clf = RandomSearchCVLightGBM(dataTrain=appTrain.drop(["SK_ID_CURR", "TARGET"], axis=1), dataTrainLabel=appTrain["TARGET"], 
#                                 n_estimators=100, paramGrid=paramGrid)
#    clf.random_search_cv(n_folds=5, random_state=165)
#    importance, report = clf.get_score()
#    bestParam = clf._bestParam
#    appTest["TARGET"] = clf.refit(appTest.drop(["SK_ID_CURR"], axis=1).values)
#    submission = appTest[['SK_ID_CURR', 'TARGET']]
#    submission.to_csv("submission_lightgbm.csv", index=False)
