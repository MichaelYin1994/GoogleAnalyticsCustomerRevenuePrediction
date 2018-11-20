#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  6 17:16:17 2018

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
def load_credit_card_balance(nrows=None):
    insPay = pd.read_csv('..//Data//credit_card_balance.csv', nrows=nrows)
    return insPay

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

def feature_engineering(cc):
    cc.drop(["SK_ID_PREV"], axis=1, inplace=True)
    cc = cc.groupby('SK_ID_CURR').agg(['min', 'max', 'mean', 'sum', 'var'])
    featureName = []
    for name in list(cc.columns):
        featureName.append(name[0] + "_" + name[1])
    cc.columns = featureName
    cc.reset_index(inplace=True)
    cc.rename(columns={"index":"SK_ID_CURR"}, inplace=True)
    
    gc.collect()
    return cc
    
def run_data_cc(nrows=None):
    cc = load_credit_card_balance(nrows=nrows)
    cc = encoding_features(cc)
    cc = feature_engineering(cc)
    cc = replace_inf_with_nan(cc)
    cc.to_csv("..//TrainTestData//cc.csv", index=False)
    gc.collect()
    #return cc

if __name__ == "__main__":
    cc = run_data_cc(nrows=None)
    ccReport = basic_feature_report(cc)
    
