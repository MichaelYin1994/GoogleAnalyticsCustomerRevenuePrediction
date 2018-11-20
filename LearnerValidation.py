#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  8 17:21:10 2018

@author: michaelyin1994
"""
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import time
import pickle
from functools import wraps
import pandas as pd
from pandas import DataFrame

from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
np.random.seed(2018)
sns.set(style="ticks", font_scale=1.2, palette='deep', color_codes=True)
###############################################################################
def timefn(fn):
    @wraps(fn)
    def measure_time(*args, **kwargs):
        start = time.time()
        result = fn(*args, **kwargs)
        end = time.time()
        print("@timefn: " + fn.__name__ + " took " + str(end-start) + " seconds")
        return result
    return measure_time

class ValidationMachine():
    def __init__(self, data=None, folds=2, stratified=False, debug=False, random_state=1):
        self._X = data.drop(["TARGET"], axis=1).values
        self._y = data["TARGET"].values
        self._featureName = list(data.drop(["TARGET"], axis=1).columns)
        self._folds = folds
        self._stratified = stratified #True or False
        self._debug = debug
        self._randomState = random_state
    
    def set_learner(self, learner=None):
        if learner not in ["gbdt", "xgboost", "erf", "rf", "lr"]:
            assert 0, "Wrong learner !"
        else:
            self.__set_learner(learner=learner)
    
    def __set_learner(self, learner=None):
        self.__learner = learner
    
    def train_learner(self):
        if self._stratified == True:
            folds = StratifiedKFold(n_splits=self._folds, shuffle=True, random_state=self._randomState)
        else:
            folds = KFold(n_splits=self._folds, shuffle=True, random_state=self._randomState)
        pass
    
    

