#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  1 20:53:29 2018

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
from WeaponLib import ReduceMemoryUsage
import gc
rcParams['patch.force_edgecolor'] = True
rcParams['patch.facecolor'] = 'b'

np.random.seed(0)
sns.set(style="ticks", font_scale=1.2, palette='deep', color_codes=True)
###############################################################################
def load_bureau_and_balance(nrows=None):
    bureau = pd.read_csv("..//Data//bureau.csv", nrows=nrows)
    balance = pd.read_csv("..//Data//bureau_balance.csv", nrows=nrows)
    return bureau, balance

def transfer_time_feature(data):
    dataColumns = list(data.columns)
    for name in dataColumns:
        if "DAY" in name:
            data[name] = data[name]/-365
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

def feature_engineering(bureau, balance):
    # Feature engineering on balance data
    bureau = transfer_time_feature(bureau)
    balanceGroupby = balance.groupby(["SK_ID_BUREAU"])
    balanceMonthBalance = balanceGroupby["MONTHS_BALANCE"].agg(["min", "max", "mean", "size", "std", "sum"])
    for name in list(balanceMonthBalance.columns):
        balanceMonthBalance.rename(columns={name:"MONTHS_BALANCE_"+name}, inplace=True)
    balanceStatus = balance.drop(["MONTHS_BALANCE"], axis=1).groupby(["SK_ID_BUREAU"]).agg(["sum", "mean", "size"])
    balanceStatusCol = list(balanceStatus.columns)
    for ind, name in enumerate(list(balanceStatus.columns)):
        balanceStatusCol[ind] = balanceStatusCol[ind][0] + "_" + balanceStatusCol[ind][1]
    balanceStatus.columns = balanceStatusCol
    balanceStatus.reset_index(inplace=True)
    balanceStatus.rename(columns={"index":"SK_ID_BUREAU"}, inplace=True)
    balanceMonthBalance.reset_index(inplace=True)
    balanceMonthBalance.rename(columns={"index":"SK_ID_BUREAU"}, inplace=True)
    balanceConcat = pd.merge(balanceStatus, balanceMonthBalance, on="SK_ID_BUREAU", how='left')
    
    # Merging Balance and Bureau data
    print("Merging Bureau with Balance data:")
    print("------------------------------------------------------")
    print("BalanceConcat merged shape :{}".format(balanceConcat.shape))
    bureau = pd.merge(bureau, balanceConcat, on="SK_ID_BUREAU", how='left')
    bureau.drop(['SK_ID_BUREAU'], axis=1, inplace= True)
    print("Bureau and balance merged shape :{}".format(bureau.shape))
    del balanceConcat,  balanceStatus, balanceMonthBalance
    gc.collect()
    print("------------------------------------------------------\n")
    
    # New features of bureau data:
    # https://www.kaggle.com/aantonova/797-lgbm-and-bayesian-optimization
    bureau['bureau AMT_CREDIT_SUM - AMT_CREDIT_SUM_DEBT'] = bureau['AMT_CREDIT_SUM'] - bureau['AMT_CREDIT_SUM_DEBT']
    bureau['bureau AMT_CREDIT_SUM - AMT_CREDIT_SUM_LIMIT'] = bureau['AMT_CREDIT_SUM'] - bureau['AMT_CREDIT_SUM_LIMIT']
    bureau['bureau AMT_CREDIT_SUM - AMT_CREDIT_SUM_OVERDUE'] = bureau['AMT_CREDIT_SUM'] - bureau['AMT_CREDIT_SUM_OVERDUE']
    bureau['bureau DAYS_CREDIT - CREDIT_DAY_OVERDUE'] = bureau['DAYS_CREDIT'] - bureau['CREDIT_DAY_OVERDUE']
    bureau['bureau DAYS_CREDIT - DAYS_CREDIT_ENDDATE'] = bureau['DAYS_CREDIT'] - bureau['DAYS_CREDIT_ENDDATE']
    bureau['bureau DAYS_CREDIT - DAYS_ENDDATE_FACT'] = bureau['DAYS_CREDIT'] - bureau['DAYS_ENDDATE_FACT']
    bureau['bureau DAYS_CREDIT_ENDDATE - DAYS_ENDDATE_FACT'] = bureau['DAYS_CREDIT_ENDDATE'] - bureau['DAYS_ENDDATE_FACT']
    bureau['bureau DAYS_CREDIT_UPDATE - DAYS_CREDIT_ENDDATE'] = bureau['DAYS_CREDIT_UPDATE'] - bureau['DAYS_CREDIT_ENDDATE']
    
    bureauNumericList = ["AMT_CREDIT_SUM", "AMT_CREDIT_SUM_DEBT", "AMT_CREDIT_MAX_OVERDUE",
                         "AMT_ANNUITY", "DAYS_CREDIT_ENDDATE", "AMT_CREDIT_SUM_LIMIT", "CREDIT_DAY_OVERDUE",
                         "DAYS_CREDIT", "DAYS_ENDDATE_FACT", "DAYS_CREDIT_UPDATE", "AMT_CREDIT_SUM_OVERDUE"]
    bureauCategoryList = list(set(bureau.columns).difference(set(bureauNumericList)))
    print("Groupby Bureau features:")
    print("------------------------------------------------------")
    print("Bureau data before transfering:{}".format(bureau.shape))
    bureauGroupby = bureau.groupby(["SK_ID_CURR"])
    aggData = pd.DataFrame(bureau["SK_ID_CURR"].unique(), columns=["SK_ID_CURR"])
    for name in bureauNumericList:
        tmp = bureauGroupby[name].agg(["min", "max", "mean", "sum", "std", "size"])
        for tmpName in list(tmp.columns):
            tmp.rename(columns={tmpName:name+"_"+tmpName}, inplace=True)
        tmp.reset_index(inplace=True)    
        tmp.rename(columns={"index":"SK_ID_CURR"})
        aggData = pd.merge(aggData, tmp, on="SK_ID_CURR", how="left")
    
    for name in bureauCategoryList:
        tmp = bureauGroupby[name].agg(["size", "mean"])
        for tmpName in list(tmp.columns):
            tmp.rename(columns={tmpName:name+"_"+tmpName}, inplace=True)
        tmp.reset_index(inplace=True)    
        tmp.rename(columns={"index":"SK_ID_CURR"})
        aggData = pd.merge(aggData, tmp, on="SK_ID_CURR", how="left")
    print("AggData shape is {}".format(aggData.shape))
    gc.collect()
    print("------------------------------------------------------\n")
    return aggData

def debug_data(nrows=None):
    bureau, balance = load_bureau_and_balance(nrows=nrows)
    print("Encoding bureau and balance data:")
    bureau = encoding_features(bureau)
    balance = encoding_features(balance)
    aggData = feature_engineering(bureau, balance)
    
    return bureau, balance, aggData

def run_data_bureau_balance(nrows=None):
    bureau, balance = load_bureau_and_balance(nrows=nrows)
    print("Encoding bureau and balance data:")
    bureau = encoding_features(bureau)
    balance = encoding_features(balance)
    
    aggData = feature_engineering(bureau, balance)
    aggData = replace_inf_with_nan(aggData)
    
    clf = ReduceMemoryUsage(data=aggData, verbose=False)
    aggData = clf.reduce_memory_usage()
    aggData.to_csv("..//TrainTestData//BureauBalance.csv", index=False)
    gc.collect()

if __name__ == "__main__":
    bureau, balance, aggData = debug_data(nrows=50000)
    
    