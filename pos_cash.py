#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug  5 23:17:26 2018

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
def load_pos_cash(nrows=None):
    posCash = pd.read_csv("..//Data//POS_CASH_balance.csv", nrows=nrows)
    return posCash

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

def feature_engineering(posCash):
    posCash.drop(["SK_ID_PREV"], axis=1, inplace=True)
    numericList = ['MONTHS_BALANCE', 'SK_DPD', 'SK_DPD_DEF']
    categoryList = list(set(posCash.columns).difference(set(numericList)))
    categoryList.remove("SK_ID_CURR")
    
    agg = {}
    for name in numericList:
        agg[name] = ["min", "max", "mean", "sum"]
    for name in categoryList:
        agg[name] = ["size", "sum"]
        
    posCashGroupby = posCash.groupby(["SK_ID_CURR"])
    posCash = posCashGroupby.agg({**agg})
    featureName = []
    for name in list(posCash.columns):
        featureName.append(name[0] + "_" + name[1])
    posCash.columns = featureName
    posCash.reset_index(inplace=True)
    posCash.rename(columns={"index":"SK_ID_CURR"}, inplace=True)
    
    return posCash
    
def run_data_pos_cash(nrows=None):
    posCash = load_pos_cash(nrows=nrows)
    posCash = encoding_features(posCash)
    posCash = feature_engineering(posCash)
    posCash = replace_inf_with_nan(posCash)
    posCash.to_csv("..//TrainTestData//posCash.csv", index=False)
    gc.collect()
    #return posCash

if __name__ == "__main__":
    posCash = run_data_pos_cash(nrows=None)
    posCashReport = basic_feature_report(posCash)
    