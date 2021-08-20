# -*- coding: utf-8 -*-
############ module ###################################
import os
import numpy as np
import random 
from pathlib import Path
from glob import glob
import pickle
import warnings 
import json

import pandas as pd
from sklearn.preprocessing import StandardScaler
from scipy import integrate

warnings.simplefilter("ignore")
############# function ###############################33
def seed_everything(seed=42):
#乱数値を設定するための関数
  random.seed(seed)
  os.environ["PYTHONHASHSEED"] = str(seed)
  np.random.seed(seed)

seed_everything()

def make_setup(train_path,test_path,sample_path):
#データを読み込んでデータフレームに帰るための関数
#trainは各サイクル別々のcsvになっているのでそれを一つのデータフレームにまとめている
    temps = [25,10,0,-10,-20]
    drivecycle_csvs = glob(train_path)
    train_dfs = [pd.read_csv(path) for path in drivecycle_csvs]
    sample_submission = pd.read_csv(sample_path)
    test_df = pd.read_csv(test_path)
    return train_dfs,test_df,sample_submission

def feature_engineering(data,graph=True):
#Current(電流)のTime(時間)積分の値を求めるための関数
  data["Time_h"] = data["Time"]/3600
  data["integral_I"] = integrate.cumtrapz(data["Current"],data["Time_h"],initial=0)
  return data

def make_data(train_df,feature,graph=False,test=False):
#(重要)
#各サイクルの中に含まれる小サイクルを抽出している
#小サイクルを抽出し、小サイクル内での特徴量の変化値を新たな特徴量としている
  copy_feature = feature.copy()
  name = train_df["Drive Cycle"].iloc[0] + "_" + train_df["Chamber_Temp_degC"].iloc[0].astype(str)
  if test:
    copy_feature.remove("Ah")
  try_data = train_df[copy_feature].copy()
  try_data["Time_diff"] = try_data["Time"].diff().fillna(0)
  condition = (try_data["Time_diff"]>0.2) & (try_data["integral_I"]!=0)
  
  index_list = list(try_data[condition].index)
  dataset = []
  for i,index in enumerate(index_list):
    if i == 0:
      start_index = index
      data = try_data[:start_index].copy()
      dataset.append(data)
      continue
    data = try_data[start_index:index].copy()
    dataset.append(data)
    start_index = index
    if i == len(index_list)-1:
      data = try_data[start_index:].copy()
      dataset.append(data)

  integral_I = float(try_data["integral_I"].iloc[0])
  for i,data in enumerate(dataset):
    data["number"] = i
    data["Temp_diff"] = data["Battery_Temp_degC"].diff().fillna(0)
    data["Time_h"] = data["Time"]/3600
    data["Time_diff"] = data["Time"].diff().fillna(0)
    data["new_Time"] = data["Time_diff"].cumsum()
    data["new_Time_h"] = data["new_Time"]/3600
    data["integral_I"] = integrate.cumtrapz(data["Current"],data["new_Time_h"],initial=0)+integral_I
    integral_I = data["integral_I"].iloc[-1]

  new_data = pd.concat(dataset,axis=0)
  return new_data

def get_name_feature(dfs):
#これはモデル構築にはかかわらないものの、大きなヒントになったので掲載
#EDAの結果から,抽出した小サイクルのサンプル数はDrive Cycleに固有な値を示すとわかった
#例えば,NNの小サイクルなら5951,HWFETなら13701など
#さらに,訓練データの複合サイクルCycle_1内部には抽出されたDrive Cycle固有のサンプル数の小サイクルで多くが構成されていることがわかった
#なお,EDAの結果は下記のリンクのgoogle colabのbookに記載している.
#https://colab.research.google.com/drive/11cCYs0UAepH5UIbCWbDvxYPvQ1v1eaf8
  new_dfs = dfs.copy()
  for try_data in new_dfs:
    seq_dict = try_data["number"].value_counts()
    try_data["type"] = " "
    try_data["seq_length"] = " "
    for number in range(len(seq_dict)):
      condition = try_data["number"] == number
      try_data["seq_length"][condition] = seq_dict[number]
      if seq_dict[number] == 5951:
        try_data["type"][condition] = "NN"
      elif seq_dict[number] == 7661:
        try_data["type"][condition] = "HWFET"
      elif seq_dict[number] == 13701:
        try_data["type"][condition] = "UDDS"
      elif seq_dict[number] == 14361:
        try_data["type"][condition] = "LA92"
      elif seq_dict[number] == 6021:
        try_data["type"][condition] = "US06_1"
      elif seq_dict[number] == 6011:
        try_data["type"][condition] = "US06_2"
      else:
        try_data["type"][condition] = "Unknown"
  return new_dfs

def get_cycle_feature(new_dfs,train=True):
#上のmake_dataと同じく小サイクル由来の特徴量を作成するための関数
  cycle_datas_dict = {}
  replace = True
  for new_train_df in new_dfs:
    try_data = new_train_df.copy()
    if replace:
      try_data = new_train_df

    try_data["name"] = try_data["Chamber_Temp_degC"].astype(str)+"_"+try_data["Drive Cycle"]
    if train:
      try_data["residual"] = try_data["Ah"] - try_data["integral_I"]
    target_datas = []
    for number in try_data["number"].unique():
      condition = try_data["number"] == number
      target_data = try_data[condition].copy()
      if replace:
        target_data = try_data[condition]
      if train:
        target_data["residual_cycle"] = target_data["residual"] - float(target_data["residual"].iloc[0])
        target_data["Ah_cycle"] = target_data["Ah"] - target_data["Ah"].iloc[0]
        target_data["recovery_residual"] = target_data["residual"].iloc[0]
      target_data["integral_cycle"] = target_data["integral_I"] - target_data["integral_I"].iloc[0] 
      target_data["Time_cycle"] = target_data["Time"] - target_data["Time"].iloc[0]
      target_data["Current_cycle"] = target_data["Current"] - target_data["Current"].iloc[0]
      target_data["Voltage_cycle"] = target_data["Voltage"] - target_data["Voltage"].iloc[0]
      target_data["Power_cycle"] = target_data["Power"] - target_data["Power"].iloc[0] 
      target_data["Temp_cycle"] = target_data["Battery_Temp_degC"] - target_data["Battery_Temp_degC"].iloc[0]
      target_datas.append(target_data)
    cycle_datas_dict[try_data["name"].iloc[0]] = target_datas

  temp_datas = []
  for cycle,cycle_datas in cycle_datas_dict.items():
    temp_data = pd.concat(cycle_datas,axis=0)
    temp_datas.append(temp_data)
  temp_datas = pd.concat(temp_datas,axis=0)
  return temp_datas


######## main.py ##############################3
with open("./setting.json","r") as f:
    setting = json.load(f)

train_path = "{}/train/*.csv".format(setting["RAW_DATA_DIR"])
test_path = "{}/test.csv".format(setting["RAW_DATA_DIR"])
sample_path = setting["PRE_SUBMISSION_DATA_PATH"]
train_prepreocess_path = setting["TRAIN_PROCESSED_DATA_PATH"]
test_preprocess_path = setting["TEST_PROCESSED_DATA_PATH"]


electro_feature = ["Current","Voltage","Power","Time"]
temp_feature = ["Battery_Temp_degC"]
basic_feature = ["Chamber_Temp_degC","Drive Cycle"]
eng_feature = ["Ah","integral_I","ID"]
feature = electro_feature+temp_feature+basic_feature+eng_feature

print("1.データの読み込み")
train_dfs,test_df,sample_submission = make_setup(train_path,test_path,sample_path)
for train_df in train_dfs:
  train_df["ID"] = train_df.index
test_dfs = []
test_df["name"] = test_df["Chamber_Temp_degC"].astype(str)+test_df["Drive Cycle"]
name_list = test_df["name"].unique()
for name in name_list:
  test_dfs.append(test_df[test_df["name"]==name])
print("2.特徴量生成")
train_dfs = [feature_engineering(train_df,False) for train_df in train_dfs]
test_dfs = [feature_engineering(test_df,False) for test_df in test_dfs]
new_train_dfs = [make_data(train_df,feature,graph=False) for train_df in train_dfs]
new_test_dfs = [make_data(test_df.reset_index(),feature,graph=False,test=True) for test_df in test_dfs]

print("3.小サイクル生成")
new_train_dfs = get_name_feature(new_train_dfs)
new_train_dfs = get_cycle_feature(new_train_dfs)
new_test_dfs = get_name_feature(new_test_dfs)
new_test_dfs = get_cycle_feature(new_test_dfs,False)

######## 保存部分 #################################
print("4.binaryfile変換")
# new_train_dfs.to_csv(train_prepreocess_path,index=False)
# new_test_dfs.to_csv(test_preprocess_path,index=False)

with open(train_prepreocess_path,"wb") as f:
  pickle.dump(new_train_dfs,f)

with open(test_preprocess_path,"wb") as f:
  pickle.dump(new_test_dfs,f)

