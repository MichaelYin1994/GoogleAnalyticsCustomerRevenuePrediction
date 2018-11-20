#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 28 21:56:46 2018

@author: michaelyin1994
"""
import numpy as np
import pandas as pd 
from sklearn.preprocessing import LabelEncoder
# 关掉可能出现的warings
import warnings
warnings.filterwarnings('ignore')
import gc
import matplotlib.pyplot as plt
import seaborn as sns
from WeaponLib import basic_feature_report
from WeaponLib import replace_inf_with_nan

# 调用ReduceMemoryUsage，用来节约内存，参见WeaponLib模块
from WeaponLib import ReduceMemoryUsage
from matplotlib import rcParams
rcParams['patch.force_edgecolor'] = True
rcParams['patch.facecolor'] = 'b'

np.random.seed(0)
sns.set(style="ticks", font_scale=1.2, palette='deep', color_codes=True)
###############################################################################
# 导入训练集测试集
def load_train(nrows=None):
    appTrain = pd.read_csv("..//Data//application_train.csv", nrows=nrows)
    print("Traing data shape is ", appTrain.shape)
    return appTrain

def load_test(nrows=None):
    appTest = pd.read_csv("..//Data//application_test.csv", nrows=nrows)
    print("Testing data shape is ", appTest.shape)
    return appTest

def transfer_time_feature(appTrain, appTest):
    appTrainColumns = list(appTrain.columns)
    for name in appTrainColumns:
        if "DAY" in name:
            appTrain[name] = appTrain[name]/-365
            appTest[name] = appTest[name]/-365
    return appTrain, appTest

# 对训练集与测试集的特征进行编码（One-hot endcoding），返回编码之后的数据
def encoding_features(dataTrain, dataTest):
    # encoding binary features
    print("\nEncoding features:")
    print("------------------------------------------------------")
    print("Orignial dataTrain shape :{}".format(dataTrain.shape))
    print("Orignial dataTest shape :{}".format(dataTest.shape))
    lbl = LabelEncoder()
    
    # 获取数据的基本描述
    basicReport = basic_feature_report(dataTrain)
    bTransfer = 0
    # 对字符串类型的Binary features（类型为object，不同值小于等于2的特征），使用lbl进行编码
    binaryStrFeatureName = list(basicReport["featureName"][(basicReport["nuniqueValues"] <= 2) & (basicReport["types"] == 'object')])
    for name in binaryStrFeatureName:
        if dataTrain[name].isnull().sum() == 0 and dataTest[name].isnull().sum() == 0:
            dataTrain[name] = lbl.fit_transform(dataTrain[name])
            dataTest[name] = lbl.fit_transform(dataTest[name])
            bTransfer += 1
    print("Binary str feature encoding :{}\n".format(bTransfer))
    
    # 对其他的类别特征进行编码
    dataTrain = pd.get_dummies(dataTrain)
    dataTest = pd.get_dummies(dataTest)
    print("Training data shape now is {}".format(dataTrain.shape))
    print("Testing data shape now is {}\n".format(dataTest.shape))
    
    # 获取数据的基本描述，并且drop掉没有变化的特征（nunique==1）
    dataTrainReport = basic_feature_report(dataTrain)
    dropList = list(dataTrainReport["featureName"][dataTrainReport["nuniqueValues"]==1].values)
    dataTrain.drop(dropList, axis=1, inplace=True)
    print("Train dropped useless features :{}".format(dropList))
    print("Train data shape after dropping :{}\n".format(dataTrain.shape))
    
    # 训练集与测试集数据进行对其，保留双方都有的特征
    print("Train shape {}, test shape {} before aligned.".format(dataTrain.shape, dataTest.shape))
    dataTrainLabels = dataTrain["TARGET"]
    dataTrain.drop(["TARGET"], axis=1, inplace=True)
    dataTrain, dataTest = dataTrain.align(dataTest, join='inner', axis=1)
    dataTrain["TARGET"] = dataTrainLabels
    print("Train shape {}, test shape {} after aligned.".format(dataTrain.shape, dataTest.shape))
    print("------------------------------------------------------\n")
    return dataTrain, dataTest

# 部分特征的可视化，用来做分箱特征的依据并且可以发现一些异常值
def anomalous_features_visualizing(appTrain, appTest):
    
    plt.figure()
    sns.distplot(appTrain["TARGET"].astype(int), kde=False, bins=3)
    plt.title("Target distribution")
    '''
    # Feature days employed, -1000 means value lost
    plt.figure()
    plt.title("Training data before removing the -1000")
    sns.distplot(appTrain["DAYS_EMPLOYED"], bins=50, kde=False)
    plt.savefig("..//Plots//trainDaysEmployedAbnormal.png", dpi=500, bbox_inches='tight')
    
    plt.figure()
    plt.title("Training data after removing -1000")
    sns.distplot(appTrain["DAYS_EMPLOYED"][appTrain["DAYS_EMPLOYED"]>-700], bins=50, kde=False)
    plt.savefig("..//Plots//trainDaysEmployedNormal.png", dpi=500, bbox_inches='tight')
    plt.close('all')
    
    plt.figure()
    plt.title("Testing data before removing -1000")
    sns.distplot(appTrain["DAYS_EMPLOYED"], bins=50, kde=False)
    plt.figure()
    
    plt.title("Testing data after removing -1000")
    sns.distplot(appTrain["DAYS_EMPLOYED"][appTrain["DAYS_EMPLOYED"]>-700], bins=50, kde=False)
    plt.close("all")
    
    plt.figure()
    plt.title("Training data before removing the -1000")
    sns.distplot(appTrain["DAYS_EMPLOYED"][appTrain["DAYS_EMPLOYED"]>-700][appTrain["TARGET"] == 1], bins=50, kde=False)
    sns.distplot(appTrain["DAYS_EMPLOYED"][appTrain["DAYS_EMPLOYED"]>-700][appTrain["TARGET"] == 0], bins=50, kde=False)
    plt.close("all")
    
    plt.figure()
    sns.kdeplot(appTrain[appTrain["DAYS_EMPLOYED"]>-700].loc[appTrain["TARGET"]==0, "DAYS_EMPLOYED"], label="target == 0")
    sns.kdeplot(appTrain[appTrain["DAYS_EMPLOYED"]>-700].loc[appTrain["TARGET"]==1, "DAYS_EMPLOYED"], label="target == 1")
    plt.ylim(0, 0.2)
    plt.title("Distribution of DAYS_EMPLOYED")
    plt.savefig("..//Plots//targetVSdaysEmployed.png", dpi=500, bbox_inches='tight')
    
    # Feature days birth, -1000 means value lost
    plt.figure()
    sns.kdeplot(appTrain.loc[appTrain["TARGET"]==0, "DAYS_BIRTH"], label="target == 0")
    sns.kdeplot(appTrain.loc[appTrain["TARGET"]==1, "DAYS_BIRTH"], label="target == 1")
    plt.title("Distribution of DAYS_BIRTH")
    plt.ylim(0, 0.04)
    plt.savefig("..//Plots//targetVSdaysBrith.png", dpi=500, bbox_inches='tight')
    plt.close("all")
    
    # Correlations between features and TARGET, select top 20.
    appTrainCorr = appTrain.corr()["TARGET"].sort_values(ascending=False).reset_index().rename(columns={"index":"featureName"})
    appTrainCorr = appTrainCorr.iloc[1:]
    fig, ax = plt.subplots()
    plt.title("Top 20 positive features")
    ax.set_xticklabels(appTrainCorr.iloc[:20]["featureName"], rotation=90, fontsize=7)
    sns.barplot(appTrainCorr["featureName"].iloc[:20], appTrainCorr["TARGET"].iloc[:20], palette='Blues_d')
    plt.savefig("..//Plots//targetCorrPositive.png", dpi=500, bbox_inches='tight')
    
    fig, ax = plt.subplots()
    plt.title("Top 20 negitive features")
    ax.set_xticklabels(appTrainCorr.iloc[-20:]["featureName"], rotation=90, fontsize=7)
    sns.barplot(appTrainCorr["featureName"].iloc[-20:], appTrainCorr["TARGET"].iloc[-20:], palette='Blues_d')
    plt.savefig("..//Plots//targetCorrNegitive.png", dpi=500, bbox_inches='tight')
    plt.close("all")
    
    # Own car age distributions
    plt.figure()
    sns.kdeplot(appTrain.loc[appTrain["TARGET"]==0, "OWN_CAR_AGE"], label="target == 0")
    sns.kdeplot(appTrain.loc[appTrain["TARGET"]==1, "OWN_CAR_AGE"], label="target == 1")
    plt.title("Distribution of OWN_CAR_AGE")
    plt.savefig("..//Plots//targetVSownCarAge.png", dpi=500, bbox_inches='tight')
    plt.close("all")
    '''
    return
    
def feature_engineering(appTrain, appTest):
    print("\nProcessing with features:")
    print("--------------------------------------------------------------")
    # Feature days employed, replace -1000 with mean value
    print("processing DAYS_EMPLOYED:")
    appTrain["DAYS_EMPLOYED"].replace(appTrain["DAYS_EMPLOYED"].min(), np.nan, axis=1, inplace=True)
    appTest["DAYS_EMPLOYED"].replace(appTest["DAYS_EMPLOYED"].min(), np.nan, axis=1, inplace=True)
    appTrain["DAYS_EMPLOYED_3_7_BINNED"] = (appTrain["DAYS_EMPLOYED"] >= 3) & (appTrain["DAYS_EMPLOYED"] <= 7) 
    appTest["DAYS_EMPLOYED_3_7_BINNED"] = (appTest["DAYS_EMPLOYED"] >= 3) & (appTest["DAYS_EMPLOYED"] <= 7) 
    print("Train data shape is {}, test data shape is {}.".format(appTrain.shape, appTest.shape))
    
    print("processing DAYS_BIRTH_BINNED:")
    # Feature days birth, create new feature, Birth_year_bins
    appTrain["DAYS_BIRTH_BINNED"] = pd.cut(appTrain["DAYS_BIRTH"], bins=np.linspace(appTrain["DAYS_BIRTH"].min(), appTrain["DAYS_BIRTH"].max(), num=10))
    appTest["DAYS_BIRTH_BINNED"] = pd.cut(appTest["DAYS_BIRTH"], bins=np.linspace(appTest["DAYS_BIRTH"].min(), appTest["DAYS_BIRTH"].max(), num=10))
    appTrain["DAYS_BIRTH_BINNED"] = appTrain["DAYS_BIRTH_BINNED"].astype(str)
    appTest["DAYS_BIRTH_BINNED"] = appTest["DAYS_BIRTH_BINNED"].astype(str)
    lbl = LabelEncoder()
    appTrain["DAYS_BIRTH_BINNED"] = lbl.fit_transform(appTrain["DAYS_BIRTH_BINNED"])
    appTest["DAYS_BIRTH_BINNED"] = lbl.fit_transform(appTest["DAYS_BIRTH_BINNED"])
    print("Train data shape is {}, test data shape is {}.".format(appTrain.shape, appTest.shape))
    
    print("processing OWN_CAR_AGE:")
    # Own car age features
    appTrain["OWN_CAR_AGE"].replace(appTrain["OWN_CAR_AGE"].min(), appTrain["OWN_CAR_AGE"].mean(), axis=1, inplace=True)
    appTest["OWN_CAR_AGE"].replace(appTest["OWN_CAR_AGE"].min(), appTest["OWN_CAR_AGE"].mean(), axis=1, inplace=True)
    appTrain["OWN_CAR_AGE_4_20_BINNED"] = (appTrain["OWN_CAR_AGE"] >= 4) & (appTrain["OWN_CAR_AGE"] <= 20) 
    appTest["OWN_CAR_AGE_4_20_BINNED"] = (appTest["OWN_CAR_AGE"] >= 4) & (appTest["OWN_CAR_AGE"] <= 20) 
    print("Train data shape is {}, test data shape is {}.".format(appTrain.shape, appTest.shape))
    
    print("Cross features:")
    # 构建交叉特征
    # kaggle: https://www.kaggle.com/willkoehrsen/start-here-a-gentle-introduction
    # kaggle: https://www.kaggle.com/aantonova/797-lgbm-and-bayesian-optimization
    appTrain['CREDIT_INCOME_PERCENT'] = appTrain['AMT_CREDIT'] / appTrain['AMT_INCOME_TOTAL']
    appTrain['ANNUITY_INCOME_PERCENT'] = appTrain['AMT_ANNUITY'] / appTrain['AMT_INCOME_TOTAL']
    appTrain['CREDIT_TERM'] = appTrain['AMT_ANNUITY'] / appTrain['AMT_CREDIT']
    appTrain['DAYS_EMPLOYED_PERCENT'] = appTrain['DAYS_EMPLOYED'] / appTrain['DAYS_BIRTH']
    appTrain['AMT_INCOME_TOTAL/CNT_CHILDREN'] = appTrain['AMT_INCOME_TOTAL'] / (1 + appTrain['CNT_CHILDREN'])
    appTrain['MISSING_FLAG'] = appTrain.isnull().sum(axis=1).values
    appTrain['EXT_SOURCE mean'] = appTrain[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].mean(axis = 1)
    appTrain['EXT_SOURCE std'] = appTrain[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].std(axis = 1)
    appTrain['EXT_SOURCE prod'] = appTrain['EXT_SOURCE_1']* appTrain['EXT_SOURCE_2'] * appTrain['EXT_SOURCE_3']
    appTrain['EXT_SOURCE_1 * EXT_SOURCE_2'] = appTrain['EXT_SOURCE_1']* appTrain['EXT_SOURCE_2']
    appTrain['EXT_SOURCE_1 * EXT_SOURCE_3'] = appTrain['EXT_SOURCE_1']* appTrain['EXT_SOURCE_3']
    appTrain['EXT_SOURCE_2 * EXT_SOURCE_3'] = appTrain['EXT_SOURCE_2']* appTrain['EXT_SOURCE_3']
    appTrain['EXT_SOURCE_1 * DAYS_EMPLOYED'] = appTrain['EXT_SOURCE_1']* appTrain['DAYS_EMPLOYED']
    appTrain['EXT_SOURCE_2 * DAYS_EMPLOYED'] = appTrain['EXT_SOURCE_2']* appTrain['DAYS_EMPLOYED']
    appTrain['EXT_SOURCE_3 * DAYS_EMPLOYED'] = appTrain['EXT_SOURCE_3']* appTrain['DAYS_EMPLOYED']
    appTrain['EXT_SOURCE_1 / DAYS_BIRTH'] = appTrain['EXT_SOURCE_1']/ appTrain['DAYS_BIRTH']
    appTrain['EXT_SOURCE_2 / DAYS_BIRTH'] = appTrain['EXT_SOURCE_2']/ appTrain['DAYS_BIRTH']
    appTrain['EXT_SOURCE_3 / DAYS_BIRTH'] = appTrain['EXT_SOURCE_3']/ appTrain['DAYS_BIRTH']
    
    appTest['CREDIT_INCOME_PERCENT'] = appTest['AMT_CREDIT'] / appTest['AMT_INCOME_TOTAL']
    appTest['ANNUITY_INCOME_PERCENT'] = appTest['AMT_ANNUITY'] / appTest['AMT_INCOME_TOTAL']
    appTest['CREDIT_TERM'] = appTest['AMT_ANNUITY'] / appTest['AMT_CREDIT']
    appTest['DAYS_EMPLOYED_PERCENT'] = appTest['DAYS_EMPLOYED'] / appTest['DAYS_BIRTH']
    appTest['AMT_INCOME_TOTAL/CNT_CHILDREN'] = appTest['AMT_INCOME_TOTAL'] / (1 + appTest['CNT_CHILDREN'])
    appTest['MISSING_FLAG'] = appTest.isnull().sum(axis=1).values
    appTest['EXT_SOURCE mean'] = appTest[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].mean(axis = 1)
    appTest['EXT_SOURCE std'] = appTest[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].std(axis = 1)
    appTest['EXT_SOURCE prod'] = appTest['EXT_SOURCE_1']* appTest['EXT_SOURCE_2']* appTest['EXT_SOURCE_3']
    appTest['EXT_SOURCE_1 * EXT_SOURCE_2'] = appTest['EXT_SOURCE_1']* appTest['EXT_SOURCE_2']
    appTest['EXT_SOURCE_1 * EXT_SOURCE_3'] = appTest['EXT_SOURCE_1']* appTest['EXT_SOURCE_3']
    appTest['EXT_SOURCE_2 * EXT_SOURCE_3'] = appTest['EXT_SOURCE_2']* appTest['EXT_SOURCE_3']
    appTest['EXT_SOURCE_1 * DAYS_EMPLOYED'] = appTest['EXT_SOURCE_1']* appTest['DAYS_EMPLOYED']
    appTest['EXT_SOURCE_2 * DAYS_EMPLOYED'] = appTest['EXT_SOURCE_2']* appTest['DAYS_EMPLOYED']
    appTest['EXT_SOURCE_3 * DAYS_EMPLOYED'] = appTest['EXT_SOURCE_3']* appTest['DAYS_EMPLOYED']
    appTest['EXT_SOURCE_1 / DAYS_BIRTH'] = appTest['EXT_SOURCE_1']/ appTest['DAYS_BIRTH']
    appTest['EXT_SOURCE_2 / DAYS_BIRTH'] = appTest['EXT_SOURCE_2']/ appTest['DAYS_BIRTH']
    appTest['EXT_SOURCE_3 / DAYS_BIRTH'] = appTest['EXT_SOURCE_3']/ appTest['DAYS_BIRTH']
    print("Train data shape is {}, test data shape is {}.\n".format(appTrain.shape, appTest.shape))
    print("Train data shape is {}, test data shape is {}.\n".format(appTrain.shape, appTest.shape))
    print("--------------------------------------------------------------\n")
    return appTrain, appTest

# 两种模式，一种是debug模式，一种是调用run模式
def debug_data(nrows=None):
    appTrain = load_train(nrows=nrows)
    appTest = load_test(nrows=nrows)
    appTrain["CODE_GENDER"].replace("XNA", "F", inplace=True, axis=1)
    appTrain.drop(appTrain[appTrain['NAME_INCOME_TYPE'] == 'Maternity leave'].index, inplace = True)
    appTrain.drop(appTrain[appTrain['NAME_FAMILY_STATUS'] == 'Unknown'].index, inplace = True)
    appTest.drop(appTest[appTrain['NAME_INCOME_TYPE'] == 'Maternity leave'].index, inplace = True)
    appTest.drop(appTest[appTrain['NAME_FAMILY_STATUS'] == 'Unknown'].index, inplace = True)
    
    appTrain, appTest = encoding_features(appTrain, appTest)
    appTrain, appTest = transfer_time_feature(appTrain, appTest)
    appTrain, appTest = feature_engineering(appTrain, appTest)
    
    appTrain = replace_inf_with_nan(appTrain)
    memorySaver = ReduceMemoryUsage(data=appTrain, verbose=False)
    appTrain = memorySaver.reduce_memory_usage()
    
    appTest = replace_inf_with_nan(appTest)
    memorySaver = ReduceMemoryUsage(data=appTest, verbose=False)
    appTest = memorySaver.reduce_memory_usage()
    return appTrain, appTest
    
def run_data_main_app(nrows=None):
    '''
    Step 1: Visualizing some basic features, encoding the str features, keep training
    and testing data aligned.
    '''
    appTrain = load_train(nrows=nrows)
    appTest = load_test(nrows=nrows)
    appTrain["CODE_GENDER"].replace("XNA", "F", inplace=True, axis=1)
    appTrain.drop(appTrain[appTrain['NAME_INCOME_TYPE'] == 'Maternity leave'].index, inplace = True)
    appTrain.drop(appTrain[appTrain['NAME_FAMILY_STATUS'] == 'Unknown'].index, inplace = True)
    appTest.drop(appTest[appTrain['NAME_INCOME_TYPE'] == 'Maternity leave'].index, inplace = True)
    appTest.drop(appTest[appTrain['NAME_FAMILY_STATUS'] == 'Unknown'].index, inplace = True)
    appTrain, appTest = encoding_features(appTrain, appTest)
    
    '''
    Step 2: feature visualizing and feature engineering.
    '''
    #anomalous_features_visualizing(appTrain, appTest)
    appTrain, appTest = transfer_time_feature(appTrain, appTest)
    appTrain, appTest = feature_engineering(appTrain, appTest)
    appTrain = replace_inf_with_nan(appTrain)
    
    # 调用memorySaver，节约内存
    memorySaver = ReduceMemoryUsage(data=appTrain, verbose=False)
    appTrain = memorySaver.reduce_memory_usage()
    appTrain.to_csv("..//TrainTestData//appTrain.csv", index=False)
    del appTrain
    gc.collect()
    
    appTest = replace_inf_with_nan(appTest)
    memorySaver = ReduceMemoryUsage(data=appTest, verbose=False)
    appTest = memorySaver.reduce_memory_usage()
    appTest.to_csv("..//TrainTestData//appTest.csv", index=False)
    del appTest
    gc.collect()
        
if __name__ == "__main__":
    #appTrain, appTest = debug_data(nrows=None)
    #trainReport = basic_feature_report(appTrain)
    #testReport = basic_feature_report(appTest)
    #anomalous_features_visualizing(appTrain, appTest)
    run_data_main_app()
    