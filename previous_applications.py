#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug  4 00:43:48 2018

@author: michaelyin1994
"""
import numpy as np
import pandas as pd 
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
from matplotlib import rcParams
import seaborn as sns
from WeaponLib import basic_feature_report
from WeaponLib import replace_inf_with_nan
import gc
rcParams['patch.force_edgecolor'] = True
rcParams['patch.facecolor'] = 'b'

np.random.seed(0)
sns.set(style="ticks", font_scale=1.2, palette='deep', color_codes=True)
###############################################################################
def load_previous_applications(nrows=None):
    appPrev = pd.read_csv("..//Data//previous_application.csv", nrows=nrows)
    return appPrev

def transfer_time_feature(data):
    dataColumns = list(data.columns)
    for name in dataColumns:
        if "DAY" in name:
            data[name] = data[name]/-365
            data[name].replace(data[name].min(), np.nan, axis=1, inplace=True)
    return data

def encoding_features(data):
    dataReport = basic_feature_report(data)
    print("------------------------------------------------------")
    print("Orignial data shape :{}".format(data.shape))
    binaryStrFeatureName = list(dataReport["featureName"][(dataReport["types"]=='object')&(dataReport["nuniqueValues"]<=2)])
    # First, encoding binary str features.
    bTransfer = 0
    lbl = LabelEncoder()
    for name in binaryStrFeatureName:
        if data[name].isnull().sum() == 0:
            lbl.fit(data[name])
            data[name] = lbl.transform(data[name])
            bTransfer += 1
    
    # Second, encoding other str features.
    data = pd.get_dummies(data, dummy_na=True)
    print("Data shape after transfering :{}".format(data.shape))
    print("binary str feature transform :{}".format(bTransfer))
    dataReport = basic_feature_report(data)
    dropList = list(dataReport["featureName"][dataReport["nuniqueValues"]==1].values)
    data.drop(dropList, axis=1, inplace=True)
    print("Dropped features :{}".format(dropList))
    print("Data shape after dropping :{}".format(data.shape))
    print("------------------------------------------------------\n")
    return data

def feature_engineering(appPrev):
    appPrev['APP_CREDIT_PERC'] = appPrev['AMT_APPLICATION'] / appPrev['AMT_CREDIT']
    appPrev["APP_CREDIT_PERC"][appPrev["APP_CREDIT_PERC"]>1000] = 0
    numericList = ["AMT_ANNUITY", "APP_CREDIT_PERC", "RATE_DOWN_PAYMENT",
                   "AMT_GOODS_PRICE", "AMT_APPLICATION", "AMT_CREDIT", 
                   "AMT_DOWN_PAYMENT", "DAYS_LAST_DUE_1ST_VERSION", "DAYS_DECISION",
                   "DAYS_FIRST_DUE", "DAYS_LAST_DUE", "DAYS_FIRST_DRAWING", 
                   "DAYS_TERMINATION", "SELLERPLACE_AREA", "RATE_INTEREST_PRIMARY",
                   "CNT_PAYMENT", "RATE_INTEREST_PRIVILEGED", "HOUR_APPR_PROCESS_START"]
    categoryList = list(set(appPrev.columns).difference(set(numericList)))
    
    aggNumeric = {}
    aggCategory = {}
    for featureName in numericList:
        aggNumeric[featureName] = ["min", "max", "mean", "std"]
    for featureName in categoryList:
        aggCategory[featureName] = ["sum", "mean"]
    appPrev = appPrev.groupby(["SK_ID_CURR"]).agg({**aggNumeric, **aggCategory})
    featureName = []
    for name in list(appPrev.columns):
        featureName.append(name[0] + "_" + name[1])
    appPrev.columns = featureName
    appPrev.reset_index(inplace=True)
    appPrev.rename(columns={"index":"SK_ID_CURR"}, inplace=True)
    return appPrev

def run_data_previous_applications(nrows=None):
    appPrev = load_previous_applications(nrows=nrows)
    appPrev = encoding_features(appPrev)
    appPrev = transfer_time_feature(appPrev)
    
    appPrev = feature_engineering(appPrev)
    appPrev = replace_inf_with_nan(appPrev)
    appPrev.to_csv("..//TrainTestData//appPrev.csv", index=False)
    gc.collect()
    #return appPrev
    
if __name__ == "__main__":
    appPrev = run_data_previous_applications(nrows=None)
    appPrevReport = basic_feature_report(appPrev)
    