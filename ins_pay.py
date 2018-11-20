#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  6 00:34:48 2018

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
def load_ins_pay(nrows=None):
    insPay = pd.read_csv("..//Data//installments_payments.csv", nrows=nrows)
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

def feature_engineering(insPay):
    insPay.drop(["SK_ID_PREV"], axis=1, inplace=True)
    for name in list(insPay.columns):
        insPay[name].fillna(insPay[name].mean(), inplace=True)
    insPay['PAYMENT_PERC'] = insPay['AMT_PAYMENT'] / insPay['AMT_INSTALMENT']
    insPay['PAYMENT_DIFF'] = insPay['AMT_INSTALMENT'] - insPay['AMT_PAYMENT']
    insPay['DPD'] = insPay['DAYS_ENTRY_PAYMENT'] - insPay['DAYS_INSTALMENT']
    agg = {
        'NUM_INSTALMENT_VERSION': ['nunique'],
        'DPD': ['max', 'mean', 'sum'],
        'PAYMENT_PERC': ['max', 'mean', 'sum', 'var'],
        'PAYMENT_DIFF': ['max', 'mean', 'sum', 'var'],
        'AMT_INSTALMENT': ['max', 'mean', 'sum'],
        'AMT_PAYMENT': ['min', 'max', 'mean', 'sum'],
        'DAYS_ENTRY_PAYMENT': ['max', 'mean', 'sum']
    }
    insPayGroupby = insPay.groupby(["SK_ID_CURR"])
    insPay = insPayGroupby.agg({**agg})
    featureName = []
    for name in list(insPay.columns):
        featureName.append(name[0] + "_" + name[1])
    insPay.columns = featureName
    insPay.reset_index(inplace=True)
    insPay.rename(columns={"index":"SK_ID_CURR"}, inplace=True)
    gc.collect()
    return insPay
    
def run_data_ins_pay(nrows=None):
    insPay = load_ins_pay(nrows=nrows)
    insPay = encoding_features(insPay)
    insPay = transfer_time_feature(insPay)
    insPay = feature_engineering(insPay)
    insPay = replace_inf_with_nan(insPay)
    insPay.to_csv("..//TrainTestData//insPay.csv", index=False)
    gc.collect()
    #return insPay

if __name__ == "__main__":
    insPay = run_data(nrows=None)
    insPayReport = basic_feature_report(insPay)