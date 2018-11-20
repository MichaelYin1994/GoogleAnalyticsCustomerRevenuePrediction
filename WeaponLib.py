# -*- coding: utf-8 -*-
"""
Created on Wed Jan 17 15:04:42 2018

@author: Administrator
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import time
import pickle
from functools import wraps
import pandas as pd
from pandas import DataFrame

from sklearn.metrics import roc_curve, auc, precision_recall_curve, recall_score, make_scorer, accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
import lightgbm as lgb
import xgboost as xgb
import random
sns.set(style="ticks", font_scale=1.2, palette='deep', color_codes=True)
###############################################################################
def save_data(data, fileName=None):
    assert fileName, "Invalid file name !"
    print("-------------------------------------")
    print("Save file to {}".format(fileName))
    f = open(fileName, 'wb')
    pickle.dump(data, f)
    f.close()
    print("Save successed !")
    print("-------------------------------------")
    
def load_data(fileName=None):
    assert fileName, "Invalid file name !"
    print("-------------------------------------")
    print("Load file from {}".format(fileName))
    f = open(fileName, 'rb')
    data = pickle.load(f)
    f.close()
    print("Load successed !")
    print("-------------------------------------")
    return data

def timefn(fn):
    @wraps(fn)
    def measure_time(*args, **kwargs):
        start = time.time()
        result = fn(*args, **kwargs)
        end = time.time()
        print("@timefn: " + fn.__name__ + " took " + str(end-start) + " seconds")
        return result
    return measure_time

@timefn
def replace_inf_with_nan(data):
    featureNameList = list(data.columns)
    for name in featureNameList:
        data[name].replace([np.inf, -np.inf], np.nan, inplace=True)
    return data

@timefn
def basic_feature_report(data):
    basicReport = data.isnull().sum()
    sampleNums = len(data)
    basicReport = pd.DataFrame(basicReport, columns=["missingNums"])
    basicReport["missingPrecent"] = basicReport["missingNums"]/sampleNums
    basicReport["nuniqueValues"] = data.nunique(dropna=False).values
    basicReport["types"] = data.dtypes.values
    basicReport.reset_index(inplace=True)
    basicReport.rename(columns={"index":"featureName"}, inplace=True)
    dataDescribe = data.describe([0.01, 0.5, 0.99]).transpose()
    dataDescribe.reset_index(inplace=True)
    dataDescribe.rename(columns={"index":"featureName"}, inplace=True)
    basicReport = pd.merge(basicReport, dataDescribe, on='featureName', how='left')
    return basicReport

def drop_most_empty_features(data=None, precent=None):
    assert precent, "@MichaelYin: Invalid missing precent !"
    dataReport = basic_feature_report(data)
    featureName = list(dataReport["featureName"][dataReport["missingPrecent"] >= precent].values)
    data.drop(featureName, axis=1, inplace=True)
    return data, featureName

###############################################################################
class FeatureSelection():
    def __init__(self, dataTrain=None, dataTrainLabel=None, stratified=True, verbose=False):
        self._dataTrain = dataTrain
        self._dataTrainLabel = dataTrainLabel
        self._verbose = verbose
        self._stratified = stratified
        self._importance = pd.DataFrame(None, columns=["featureName"])
        self._importance["featureName"] = list(self._dataTrain.columns)
        self._report = DataFrame(None, columns=["fold", "trainScore", "validScore"])
    
    @timefn    
    def lgb_feature_importance(self, n_folds=3, shuffle=True, random_state=0):
        self.__lgb_feature_importance(n_folds=n_folds, shuffle=shuffle, random_state=random_state)
    
    def __lgb_feature_importance(self, n_folds=None, shuffle=True, random_state=0):
        print("\nLightGBM cross validation feature importance:")
        print("-------------------------------------------")
        importances = 0
        if self._verbose == True:
            print("Training data shape is {}.".format(self._dataTrain.shape))
        else:
            pass
        if self._stratified == True:
            folds = StratifiedKFold(n_splits=n_folds, shuffle=shuffle, random_state=random_state)
        else:
            folds = KFold(n_splits=n_folds, shuffle=shuffle, random_state=random_state)
        reportTmp = np.zeros((n_folds, 3))
        for fold, (trainId, validationId) in enumerate(folds.split(self._dataTrain, self._dataTrainLabel)):
            print("\n-------------------------------------------")
            print("Start fold {}".format(fold))
            X_train, y_train = self._dataTrain.iloc[trainId], self._dataTrainLabel.iloc[trainId]
            X_valid, y_valid = self._dataTrain.iloc[validationId], self._dataTrainLabel.iloc[validationId]
            params = {
                    'n_estimators': 5000,
                    'objective': 'binary',
                    'boosting_type': 'gbdt',
                    'nthread': -1,
                    'learning_rate': 0.02,
                    'num_leaves': 20,
                    'subsample': 0.8715623,
                    'subsample_feq': 1,
                    'colsample_bytree':0.9497036,
                    'reg_alpha': 0.04154,
                    'reg_lambda': 0.0735294,
                    'silent': self._verbose
                    }
            self._params = params
            clf = lgb.LGBMClassifier(**params)
            clf.fit(X_train.values, y_train.values, eval_set=[(X_valid.values, y_valid.values)], early_stopping_rounds=20)
            score = clf.evals_result_
            scoreKeys = list(score.keys())
            y_pred = clf.predict_proba(X_valid.values)[:, 1]
            reportTmp[fold, 0] = fold
            reportTmp[fold, 1] = score[scoreKeys[0]]["binary_logloss"][-1]
            reportTmp[fold, 2] = roc_auc_score(y_valid.values, y_pred)
            importances += clf.feature_importances_
        self._importance["lgb"] = importances/n_folds
        save_data(clf, "..//TrainedModel//LightGBM.pkl")
        print("-------------------------------------------")
    
    @timefn    
    def xgb_feature_importance(self, n_folds=3, shuffle=True, random_state=0):
        self.__xgb_feature_importance(n_folds=n_folds, shuffle=shuffle, random_state=random_state)
    
    def __xgb_feature_importance(self, n_folds=None, shuffle=True, random_state=0):
        print("\nXgboost cross validation feature importance:")
        importances = 0
        print("-------------------------------------------")
        if self._verbose == True:
            print("Training data shape is {}.".format(self._dataTrain.shape))
        else:
            pass
        if self._stratified == True:
            folds = StratifiedKFold(n_splits=n_folds, shuffle=shuffle, random_state=random_state)
        else:
            folds = KFold(n_splits=n_folds, shuffle=shuffle, random_state=random_state)
        reportTmp = np.zeros((n_folds, 3))
        for fold, (trainId, validationId) in enumerate(folds.split(self._dataTrain, self._dataTrainLabel)):
            print("\n-------------------------------------------")
            print("Start fold {}".format(fold))
            X_train, y_train = self._dataTrain.iloc[trainId], self._dataTrainLabel.iloc[trainId]
            X_valid, y_valid = self._dataTrain.iloc[validationId], self._dataTrainLabel.iloc[validationId]
            params = {'max_depth':6,
                      "booster": "gbtree",
                      'n_estimators': 2000,
                      'learning_rate': 0.05,
                      'subsample': 0.85,
                      'colsample_bylevel': 0.632,
                      'colsample_bytree': 0.7,
                      'silent': self._verbose,
                      'objective':'binary:logistic',
                      'eval_metric':'auc',
                      'seed': random_state,
                      'nthread': -1,
                      'missing': np.nan}
            self._params = params
            clf = xgb.XGBClassifier(**params)
            clf.fit(X_train.values, y_train.values, eval_set=[(X_valid.values, y_valid.values)], early_stopping_rounds=20)
            score = clf.evals_result()
            scoreKeys = list(score.keys())
            y_pred = clf.predict_proba(X_valid.values)[:, 1]
            reportTmp[fold, 0] = fold
            reportTmp[fold, 1] = score[scoreKeys[0]]["auc"][-1]
            reportTmp[fold, 2] = roc_auc_score(y_valid.values, y_pred)
            importances += clf.feature_importances_
        self._importance["xgb"] = importances/n_folds
        print("-------------------------------------------")
###############################################################################
def lgb_cv(trainData, trainLabel, testData, testLabel):
    params = {'n_estimators': 5000,
              'objective': 'binary',
              'boosting_type': 'gbdt',
              'nthread': -1,
              'learning_rate': 0.02,
              'num_leaves': 20,
              'subsample': 0.8715623,
              'subsample_feq': 1,
              'colsample_bytree':0.9497036,
              'reg_alpha': 0.04154,
              'reg_lambda': 0.0735294,
              'verbose_eval': 1
              }
    trainData= lgb.Dataset(data=trainData, label=trainLabel)
    cv_results = lgb.cv(params, trainData, num_boost_round=10000,
                        early_stopping_rounds=80, metrics='auc',
                        nfold=5, seed=42)
    return cv_results
###############################################################################
# Reduce training data memory
class ReduceMemoryUsage():
    def __init__(self, data=None, verbose=True):
        self._data = data
        self._verbose = verbose
    
    def types_report(self, data):
        dataTypes = data.dtypes.values
        basicReport = pd.DataFrame(dataTypes, columns=["types"])
        basicReport["featureName"] = list(data.columns)
        return basicReport
    
    @timefn
    def reduce_memory_usage(self):
        self.__reduce_memory()
        return self._data
    
    def __reduce_memory(self):
        print("\nReduce memory process:")
        print("-------------------------------------------")
        memoryStart = self._data.memory_usage(deep=True).sum() / 1024**2
        if self._verbose == True:
            print("@Memory usage of data is {}".format(memoryStart))
        self._types = self.types_report(self._data)
        for ind, name in enumerate(self._types["featureName"].values):
            featureType = str(self._types[self._types["featureName"] == name]["types"])
            if featureType != "object":
                featureMin = self._data[name].min()
                featureMax = self._data[name].max()
                if "int" in featureType:
                    # np.iinfo for reference:
                    # https://docs.scipy.org/doc/numpy/reference/generated/numpy.iinfo.html
                    
                    # numpy data types reference:
                    # https://wizardforcel.gitbooks.io/ts-numpy-tut/content/3.html
                    if featureMin > np.iinfo(np.int8).min and featureMax < np.iinfo(np.int8).max:
                        self._data[name] = self._data[name].astype(np.int8)
                    elif featureMin > np.iinfo(np.int16).min and featureMax < np.iinfo(np.int16).max:
                        self._data[name] = self._data[name].astype(np.int16)
                    elif featureMin > np.iinfo(np.int32).min and featureMax < np.iinfo(np.int32).max:
                        self._data[name] = self._data[name].astype(np.int32)
                    elif featureMin > np.iinfo(np.int64).min and featureMax < np.iinfo(np.int64).max:
                        self._data[name] = self._data[name].astype(np.int64)
                else:
                    if featureMin > np.finfo(np.float16).min and featureMax < np.finfo(np.float16).max:
                        self._data[name] = self._data[name].astype(np.float16)
                    elif featureMin > np.finfo(np.float32).min and featureMax < np.finfo(np.float32).max:
                        self._data[name] = self._data[name].astype(np.float32)
                    else:
                        self._data[name] = self._data[name].astype(np.float64)
            if self._verbose == True:
                print("Processed {} feature, total is {}.".format(ind+1, len(self._types)))
        memoryEnd = self._data.memory_usage(deep=True).sum() / 1024**2
        if self._verbose == True:
            print("@Memory usage after optimization: {}".format(memoryEnd))
            print("@Decreased by {}".format(100 * (memoryStart - memoryEnd) / memoryStart))
        print("-------------------------------------------")
        
###############################################################################
# XGBoost Cross Validation
'''
'max_depth':6,
"booster": "gbtree",
'n_estimators': 500,
'learning_rate': 0.05,
'subsample': 0.85,
'colsample_bylevel': 0.632,
'colsample_bytree': 0.7,
'silent': self._verbose,
'objective':'binary:logistic',
'eval_metric':'auc',
'seed': randomState,
'n_jobs': 4,
'missing': np.nan
'''
class RandomSearchCVXGBoost():
    def __init__(self, dataTrain=None, dataTrainLabel=None, n_estimators=None, stratified=True, paramGrid=None, verbose=False, randomState=2018):
        self._dataTrain, self._dataTest, self._dataTrainLabel, self._dataTestLabel = train_test_split(dataTrain, dataTrainLabel,
                                                                                                      test_size=0.15)
        self._stratified = stratified
        self._verbose = verbose
        self._importance = pd.DataFrame(None, columns=["featureName"])
        self._importance["featureName"] = list(self._dataTrain.columns)
        self._paramGrid = paramGrid
        self._report = {}
        self._estimators = n_estimators
        
    def get_score(self):
        return self._importance, self._report
    
    def random_search_cv(self, n_folds=3, random_state=0):
        print("-------------------------------------------")
        self._bestParam = {}
        bestAucMean = 0
        
        for paramInd in range(self._estimators):
            print("\nNow is {}:".format(paramInd))
            cvRes = {}
            cvRes["param"] = {k: random.sample(list(v), 1)[0] for k, v in self._paramGrid.items()}
            importance, cvReport = self.__cross_validation(n_folds=n_folds, random_state=random_state, params=cvRes["param"])
            cvRes["cvReport"] = cvReport
            cvRes["validAucMean"] = cvReport["validAuc"].mean()
            cvRes["validAucStd"] = cvReport["validAuc"].std()
            cvRes["testAucMean"] = cvReport["testAuc"].mean()
            cvRes["testAucStd"] = cvReport["testAuc"].std()
            self._report[paramInd] = cvRes
            self._importance["param_" + str(paramInd)] = importance
            if cvRes["validAucMean"] > bestAucMean:
                self._bestParam["param"] = cvRes["param"]
                self._bestParam["paramInd"] = paramInd
                self._bestParam["cvReport"] = cvReport
                self._bestParam["validAucMean"] = cvRes["validAucMean"]
                self._bestParam["validAucStd"] = cvRes["validAucStd"]
                self._bestParam["testAucMean"] = cvRes["testAucMean"]
                self._bestParam["testAucStd"] = cvRes["testAucStd"]
                bestAucMean = cvRes["testAucMean"]
                
    def __cross_validation(self, n_folds=None, shuffle=True, random_state=0, params=None):
        print("XGBoost cross validation:")
        importance = 0
        params['objective'] = 'binary:logistic'
        params["booster"] = 'gbtree'
        params["nthread"] = -1
        params['eval_metric'] = 'auc'
        params["silent"] = self._verbose
        print("-------------------------------------------")
        if self._verbose == True:
            print("Training data shape is {}.".format(self._dataTrain.shape))
        else:
            pass
        if self._stratified == True:
            folds = StratifiedKFold(n_splits=n_folds, shuffle=shuffle, random_state=random_state)
        else:
            folds = KFold(n_splits=n_folds, shuffle=shuffle, random_state=random_state)
        reportTmp = np.zeros((n_folds, 5))
        for fold, (trainId, validationId) in enumerate(folds.split(self._dataTrain, self._dataTrainLabel)):
            print("-------------------------------------------")
            print("Start fold {}:".format(fold))
            X_train, y_train = self._dataTrain.iloc[trainId], self._dataTrainLabel.iloc[trainId]
            X_valid, y_valid = self._dataTrain.iloc[validationId], self._dataTrainLabel.iloc[validationId]
            clf = xgb.XGBClassifier(**params)
            clf.fit(X_train.values, y_train.values, eval_set=[(X_valid.values, y_valid.values)], early_stopping_rounds=20, verbose=self._verbose)
            score = clf.evals_result()
            scoreKeys = list(score.keys())
            y_pred = clf.predict_proba(X_valid.values)[:, 1]
            y_test_pred = clf.predict_proba(self._dataTest.values)[:, 1]
            
            reportTmp[fold, 0] = fold
            reportTmp[fold, 1] = score[scoreKeys[0]]["auc"][clf.best_ntree_limit-1]
            reportTmp[fold, 2] = roc_auc_score(y_valid.values, y_pred)
            reportTmp[fold, 3] = roc_auc_score(self._dataTestLabel.values, y_test_pred)
            reportTmp[fold, 4] = clf.best_ntree_limit-1
            importance += clf.feature_importances_
            print("Test score:{}, valid score:{}".format(reportTmp[fold, 3], reportTmp[fold, 2]))
        report = DataFrame(reportTmp, columns=["fold", "trainAuc", "validAuc", "testAuc", 'bestIteration'])
        return importance/n_folds, report
        print("-------------------------------------------\n")
        
    def refit(self, test):
        bestAuc = self._bestParam["cvReport"]["validAuc"].max()
        bestIterations = int(self._bestParam["cvReport"]["bestIteration"][self._bestParam["cvReport"]["validAuc"] == bestAuc].values)
        params = self._bestParam["param"]
        params['objective'] = 'binary:logistic'
        params["booster"] = 'gbtree'
        params["nthread"] = -1
        params['eval_metric'] = 'auc'
        params["silent"] = self._verbose
        params["n_estimators"] = bestIterations
        clf =  xgb.XGBClassifier(**params)
        clf.fit(self._dataTrain.values, self._dataTrainLabel.values, verbose=False)
        y_pred = clf.predict_proba(test, ntree_limit=bestIterations)[:, 1]
        self.save_refit_model(clf, "..//TrainedModel//xgboost.pkl")
        return y_pred
    
    def save_refit_model(self, data, fileName=None):
        self.__save_model(data, fileName)
        
    def __save_model(self, data, fileName=None):
        assert fileName, "Invalid file name !"
        print("-------------------------------------")
        print("Save file to {}".format(fileName))
        f = open(fileName, 'wb')
        pickle.dump(data, f)
        f.close()
        print("Save successed !")
        print("-------------------------------------")

###############################################################################
class RandomSearchCVLightGBM():
    def __init__(self, dataTrain=None, dataTrainLabel=None, n_estimators=None, stratified=True, paramGrid=None, verbose=False, randomState=2018):
        self._dataTrain, self._dataTest, self._dataTrainLabel, self._dataTestLabel = train_test_split(dataTrain, dataTrainLabel,
                                                                                                      test_size=0.15)
        self._stratified = stratified
        self._verbose = verbose
        self._importance = pd.DataFrame(None, columns=["featureName"])
        self._importance["featureName"] = list(self._dataTrain.columns)
        self._paramGrid = paramGrid
        self._report = {}
        self._estimators = n_estimators
        
    def get_score(self):
        return self._importance, self._report
    
    def random_search_cv(self, n_folds=3, random_state=0):
        print("-------------------------------------------")
        self._bestParam = {}
        bestAucMean = 0
        
        for paramInd in range(self._estimators):
            print("\nNow is {}:".format(paramInd))
            cvRes = {}
            cvRes["param"] = {k: random.sample(list(v), 1)[0] for k, v in self._paramGrid.items()}
            importance, cvReport = self.__cross_validation(n_folds=n_folds, random_state=random_state, params=cvRes["param"])
            cvRes["cvReport"] = cvReport
            cvRes["validAucMean"] = cvReport["validAuc"].mean()
            cvRes["validAucStd"] = cvReport["validAuc"].std()
            cvRes["testAucMean"] = cvReport["testAuc"].mean()
            cvRes["testAucStd"] = cvReport["testAuc"].std()
            self._report[paramInd] = cvRes
            self._importance["param_" + str(paramInd)] = importance
            if cvRes["validAucMean"] > bestAucMean:
                self._bestParam["param"] = cvRes["param"]
                self._bestParam["paramInd"] = paramInd
                self._bestParam["cvReport"] = cvReport
                self._bestParam["validAucMean"] = cvRes["validAucMean"]
                self._bestParam["validAucStd"] = cvRes["validAucStd"]
                self._bestParam["testAucMean"] = cvRes["testAucMean"]
                self._bestParam["testAucStd"] = cvRes["testAucStd"]
                bestAucMean = cvRes["testAucMean"]
                
    def __cross_validation(self, n_folds=None, shuffle=True, random_state=0, params=None):
        print("LightGBM cross validation:")
        importance = 0
        params['objective'] = 'binary'
        params["boosting_type"] = 'gbdt'
        params["nthread"] = -1
        params['subsample_feq'] = 1
        print("-------------------------------------------")
        if self._verbose == True:
            print("Training data shape is {}.".format(self._dataTrain.shape))
        else:
            pass
        if self._stratified == True:
            folds = StratifiedKFold(n_splits=n_folds, shuffle=shuffle, random_state=random_state)
        else:
            folds = KFold(n_splits=n_folds, shuffle=shuffle, random_state=random_state)
        reportTmp = np.zeros((n_folds, 5))
        for fold, (trainId, validationId) in enumerate(folds.split(self._dataTrain, self._dataTrainLabel)):
            print("-------------------------------------------")
            print("Start fold {}:".format(fold))
            X_train, y_train = self._dataTrain.iloc[trainId], self._dataTrainLabel.iloc[trainId]
            X_valid, y_valid = self._dataTrain.iloc[validationId], self._dataTrainLabel.iloc[validationId]
            clf = lgb.LGBMClassifier(**params)
            clf.fit(X_train.values, y_train.values, eval_set=[(X_valid.values, y_valid.values)], early_stopping_rounds=200, verbose=self._verbose)
            score = clf.evals_result_
            scoreKeys = list(score.keys())
            y_pred = clf.predict_proba(X_valid.values)[:, 1]
            y_test_pred = clf.predict_proba(self._dataTest.values)[:, 1]
            
            reportTmp[fold, 0] = fold
            reportTmp[fold, 1] = score[scoreKeys[0]]["binary_logloss"][-1]
            reportTmp[fold, 2] = roc_auc_score(y_valid.values, y_pred)
            reportTmp[fold, 3] = roc_auc_score(self._dataTestLabel.values, y_test_pred)
            reportTmp[fold, 4] = clf.best_iteration_
            importance += clf.feature_importances_
            print("Test score:{}, valid score:{}".format(reportTmp[fold, 3], reportTmp[fold, 2]))
        report = DataFrame(reportTmp, columns=["fold", "binaryLogloss", "validAuc", "testAuc", 'bestIteration'])
        return importance/n_folds, report
        print("-------------------------------------------\n")
        
    def refit(self, test):
        bestAuc = self._bestParam["cvReport"]["validAuc"].max()
        bestIterations = int(self._bestParam["cvReport"]["bestIteration"][self._bestParam["cvReport"]["validAuc"] == bestAuc].values)
        params = self._bestParam["param"]
        params['objective'] = 'binary'
        params["boosting_type"] = 'gbdt'
        params["nthread"] = -1
        params['subsample_feq'] = 1
        clf = lgb.LGBMClassifier(**params)
        clf.fit(self._dataTrain.values, self._dataTrainLabel.values)
        y_pred = clf.predict_proba(test, num_iteration=bestIterations)[:, 1]
        return y_pred
    
    def save_refit_model(self, data, fileName=None):
        self.__save_model(data, fileName)
        
    def __save_model(self, data, fileName=None):
        assert fileName, "Invalid file name !"
        print("-------------------------------------")
        print("Save LightGBM to {}".format(fileName))
        f = open(fileName, 'wb')
        pickle.dump(data, f)
        f.close()
        print("Save successed !")
        print("-------------------------------------")
###############################################################################
class Stacking():
    def __init__(self, dataTrain=None, dataTrainLabel=None, dataToPredict=None, baseModelPath=None, folds=3, randomState=2018):
        '''
        dataTrain: numpy array
        dataTrainLabels: numpy array
        dataToPredict: Pandas DataFrame, kaggle submission file
        baseModelPath: List, it contains baseModel source path, and the base model is saved in .pkl file
        '''
        self._dataTrain = dataTrain
        self._dataTrainLabel = dataTrainLabel
        self._dataToPredict = dataToPredict
        self._randomState = randomState
        self._baseModelPath = baseModelPath
        self._folds = folds    
    
    def fit_predict(self):
        X = self._dataTrain
        y = self._dataTrainLabel
        X_test = self._dataToPredict
        
        folds = StratifiedKFold(n_splits=self._folds, shuffle=True, random_state=self._randomState)
        self._baseModel = self.load_base_model()
        
        S_train = np.zeros((X.shape[0], len(self._baseModel)))
        S_test = np.zeros((X_test.shape[0], len(self._baseModel)))
        
        for ind, clf in enumerate(self._baseModel):
            print(("\n-------------------------------------"))
            print("Learner {}, path {}:".format(ind, self._baseModelPath[ind]))
            S_test_tmp = np.zeros((X_test.shape[0], self._folds))
            for fold, (trainId, validationId) in enumerate(folds.split(X, y)):
                print("Learner fold {}".format(fold))
                X_train, y_train = X[trainId], y[trainId]
                X_valid, _ = X[validationId], y[validationId]
                
                clf.fit(X_train, y_train)
                S_train[validationId, ind] = clf.predict_proba(X_valid)[:, 1]
                S_test_tmp[:, fold] = clf.predict_proba(X_test)[:, 1]
            S_test[:, ind] = S_test_tmp.mean(1)
        print(("-------------------------------------"))
        lr_results = self.logistic_regression(S_train, y, n_iters=200)
        y_pred = lr_results["bestEstimator"].predict_proba(S_test)[:, 1]
        return y_pred
    
    def logistic_regression(self, X_train, y_train, n_iters=10):
        # Random Search for 2nd level model
        lr = LogisticRegression(fit_intercept=True, max_iter=500, penalty='l2', solver='sag')
        C = np.arange(0.0001, 100, 0.1)
        random_state = [1, 2, 3, 4, 5]
        param = {
                "C":C,
                "random_state":random_state
                }
        
        print("===================Training Logistic Regression===================")
        clf = RandomizedSearchCV(estimator=lr,
                                 param_distributions=param,
                                 n_iter=n_iters,
                                 cv=self._folds,
                                 verbose=1,
                                 n_jobs=-1)
        clf.fit(X_train, y_train)
        print("==================================================================")
        lr_results = {}
        lr_results["bestEstimator"] = clf.best_estimator_
        lr_results["bestCvScore"] = clf.best_score_
        lr_results["bestParam"] = clf.best_params_
        return lr_results
    
    def load_base_model(self):
        return self.__load_base_model()
    
    def __load_base_model(self):
        assert len(self._baseModelPath), "Invalid file path !"
        baseModel = []
        print("-------------------------------------")
        for ind in range(len(self._baseModelPath)):
            pathTmp = self._baseModelPath[ind]
            print("Load file from {}".format(pathTmp))
            f = open(pathTmp, 'rb')
            baseModel.append(pickle.load(f))
            f.close()
        print("Load successed !")
        print("-------------------------------------")
        return baseModel

###############################################################################
def PR_curve(y_real, y_prob, colorCode='b', name=None):
    precision, recall, _ = precision_recall_curve(y_real, y_prob)
    plt.step(recall, precision, color='k', alpha=0.2, where='post')
    plt.fill_between(recall, precision, step='post', alpha=0.2, color=colorCode)
    prAuc = auc(recall, precision)
    
    plt.xlim(0, 1)
    plt.ylim(0, 1.05)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("PR curve and auc is {:4f}".format(prAuc))
    
def ROC_curve(y_real, y_prob, colorCode='b'):
    fpr, tpr, _ = roc_curve(y_real, y_prob)
    rocAuc = auc(fpr, tpr)
    plt.plot(fpr, tpr)
    
    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.xlim(0, 1)
    plt.ylim(0, 1.05)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC curve and auc = {:4f}".format(rocAuc))

def printInfo(data):
	print("============================================")
	print(data.info(memory_usage='deep'))
	print("============================================")

def GBDT(X_train, y_train):
    gbdt = GradientBoostingClassifier()
    n_estimators = [i for i in range(100, 1000)]
    learning_rate = np.arange(0.01, 1, 0.1)
    max_depth = [i for i in range(3, 50)]
    min_samples_split = [i for i in range(2, 30)]
    min_samples_leaf = [i for i in range(1, 30)]
    subsample = np.arange(0.8, 1, 0.02)
    max_features = ['sqrt', 'log2', None]
    random_state = [1, 20, 300, 400, 500]
    
    param = {
            "n_estimators":n_estimators,
            'learning_rate':learning_rate,
            "max_depth":max_depth,
            'min_samples_split':min_samples_split,
            'min_samples_leaf':min_samples_leaf,
            'subsample':subsample,
            'max_features':max_features,
            "random_state":random_state
            }
    print("===================Training GBDT Classifier===================")
    clf = RandomizedSearchCV(estimator=gbdt,
                             param_distributions=param,
                             n_iter=1,
                             cv=4,
                             verbose=1,
                             n_jobs=-1)
    clf.fit(X_train, y_train)
    print("==================================================================")
    gbdt_results = {}
    gbdt_results["bestEstimator"] = clf.best_estimator_
    gbdt_results["bestCvScore"] = clf.best_score_
    gbdt_results["bestParam"] = clf.best_params_
    
    return gbdt_results

def random_forest(X, y, searchMethod='RandomSearch'): 
    rf = RandomForestClassifier()
    n_estimators = [i for i in range(100, 1000)]
    max_depth = [int(x) for x in range(5, 200, 5)]
    max_features = ('auto', 'sqrt', 'log2', None)
    min_samples_split = [int(x) for x in range(2, 20)]
    min_samples_leaf = [int(x) for x in range(1, 20)]
    param = {
            "n_estimators": n_estimators,
            "max_features": max_features,
            "min_samples_split": min_samples_split,
            "min_samples_leaf": min_samples_leaf,
            "max_depth": max_depth,
            }
    print("------------Training Random Forest--------------")
    clf = RandomizedSearchCV(estimator=rf,
                             param_distributions=param,
                             n_iter=30,
                             cv=5,
                             verbose=1,
                             n_jobs=-1,
                             )
    clf.fit(X, y)
    print("------------------------------------------------")
    rf_results = {}
    rf_results["bestEstimator"] = clf.best_estimator_
    rf_results["bestCvScore"] = clf.best_score_
    rf_results["bestParam"] = clf.best_params_
    return rf_results