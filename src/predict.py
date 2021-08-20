# -*- coding: utf-8 -*-
import random
import os
import numpy as np
import pickle
import warnings
import json

import pandas as pd
from sklearn.metrics import mean_squared_error as mse
from sklearn.linear_model import Ridge

warnings.simplefilter("ignore")

############### function ##############################
def seed_everything(seed=42):
  random.seed(seed)
  os.environ["PYTHONHASHSEED"] = str(seed)
  np.random.seed(seed)

seed_everything()

def to_SOC(try_datas):
#小サイクルごとのAh_cycleを予測させてた後,そこからSOCに変換するための関数
#Ah_cycleからAhに変換,SOCに戻している.
  name_dfs = []
  for name in try_datas["name"].unique():
      name_df = try_datas[try_datas["name"]==name]
      start_integral = 0
      number_dfs = []
      for number in name_df["number"].unique():
        number_df = name_df[name_df["number"]==number]
        number_df["y_pred_actual"] = number_df["y_pred"]+start_integral
        start_integral = number_df["y_pred_actual"].iloc[-1]
        number_dfs.append(number_df)
      number_dfs = pd.concat(number_dfs)
      number_dfs["SOC_pred"] = (2.9+number_dfs["y_pred_actual"])/2.9*100
      name_dfs.append(number_dfs["SOC_pred"])
  return pd.concat(name_dfs)

def cycle_to_result(X_test,test,model):
    pred = model.predict(X_test)
    try_datas = test.copy()
    try_datas["y_pred"] = pred
    try_datas["SOC_pred"] = to_SOC(try_datas)
    return try_datas[["ID","SOC_pred"]]

########3 main ###############################
with open("./setting.json","r") as f:
    setting = json.load(f)

train_prepreocess_path = setting["TRAIN_PROCESSED_DATA_PATH"]
test_preprocess_path = setting["TEST_PROCESSED_DATA_PATH"]
models_path = setting["MODEL_CHECKPOINT_DIR"]

old_submission_path = setting["PRE_SUBMISSION_DATA_PATH"]
new_submission_path = setting["SUBMISSION_DATA_PATH"]


with open(train_prepreocess_path,"rb") as f:
  new_train_dfs = pickle.load(f)

with open(test_preprocess_path,"rb") as f:
  new_test_dfs = pickle.load(f)

submission = pd.read_csv(old_submission_path)

basic_feature = ["integral_I","Battery_Temp_degC"]
cycle_feature = ["integral_cycle","Temp_cycle"]
feature = cycle_feature+basic_feature
cycle_target = "Ah_cycle"
target = cycle_target

##### prediction #############
#基本的な変数はBattery_Temp_degC,integral_I,
#小サイクルの特徴量であるintegral_cycle,temp_cycleを採用している
#20度,10度の予測は以下のgoogle colabのbooのEDAを元に
#構築するモデルと予測するサイクルを変えている.
#https://colab.research.google.com/drive/11cCYs0UAepH5UIbCWbDvxYPvQ1v1eaf8#scrollTo=xZ-JI2dC9XkD

##### -20度 ###################
try_test = new_test_dfs.copy()
try_test = try_test[(try_test["Chamber_Temp_degC"]==-20)&(try_test["integral_I"]!=0)]
condition = (try_test["name"].str.contains("HWFET"))
try_test = try_test[~condition]
X_test = try_test[feature]

with open("{}/model_min20.binaryfile".format(models_path),"rb") as f:
  model_min20 = pickle.load(f)

result_min20 = cycle_to_result(X_test,try_test,model_min20)

df_min20 = new_test_dfs[(new_test_dfs["Chamber_Temp_degC"]==-20)&(new_test_dfs["integral_I"]!=0)]
result_min20_HWFET = df_min20[df_min20["name"].str.contains("HWFET")][["ID","integral_I"]]
result_min20_HWFET["SOC_pred"] = (2.9+result_min20_HWFET["integral_I"])/2.9*100
result_min20_HWFET = result_min20_HWFET.drop("integral_I",axis=1)
result_min20_zero = new_test_dfs[(new_test_dfs["integral_I"]==0)&(new_test_dfs["Chamber_Temp_degC"]==-20)][["ID","integral_I"]]
result_min20_zero["SOC_pred"] = (2.9+result_min20_zero["integral_I"])/2.9*100
result_min20_zero = result_min20_zero.drop("integral_I",axis=1)
total_result_min20 = pd.concat([result_min20,result_min20_HWFET,result_min20_zero],axis=0)
total_result_min20 = total_result_min20.sort_values("ID").reset_index(drop=True)

#### -10度 #####################

try_test = new_test_dfs.copy()
try_test = try_test[(try_test["Chamber_Temp_degC"]==-10)&(try_test["integral_I"]!=0)]
condition = (try_test["name"].str.contains("Cycle"))
try_test = try_test[condition]
X_test = try_test[feature]

with open("{}/model_min10_Cycle.binaryfile".format(models_path),"rb") as f:
  model_min10_Cycle = pickle.load(f)

result_min10_Cycle = cycle_to_result(X_test,try_test,model_min10_Cycle)

try_test = new_test_dfs.copy()
try_test = try_test[(try_test["Chamber_Temp_degC"]==-10)&(try_test["integral_I"]!=0)]
condition = (try_test["name"].str.contains("HWFET|Cycle"))
try_test = try_test[~condition]
X_test = try_test[feature]

with open("{}/model_min10_nonCycle.binaryfile".format(models_path),"rb") as f:
  model_min10_nonCycle = pickle.load(f)

result_min10_nonCycle = cycle_to_result(X_test,try_test,model_min10_nonCycle)

df_min10 = new_test_dfs[(new_test_dfs["Chamber_Temp_degC"]==-10)&(new_test_dfs["integral_I"]!=0)]
result_min10_HWFET = df_min10[df_min10["name"].str.contains("HWFET")][["ID","integral_I"]]
result_min10_HWFET["SOC_pred"] = (2.9+result_min10_HWFET["integral_I"])/2.9*100
result_min10_HWFET = result_min10_HWFET.drop("integral_I",axis=1)
result_min10_zero = new_test_dfs[(new_test_dfs["integral_I"]==0)&(new_test_dfs["Chamber_Temp_degC"]==-10)][["ID","integral_I"]]
result_min10_zero["SOC_pred"] = (2.9+result_min10_zero["integral_I"])/2.9*100
result_min10_zero = result_min10_zero.drop("integral_I",axis=1)
total_result_min10 = pd.concat([result_min10_nonCycle,result_min10_Cycle,result_min10_HWFET,result_min10_zero],axis=0)
total_result_min10 = total_result_min10.sort_values("ID").reset_index(drop=True)

#### 0度  ######################

try_test = new_test_dfs.copy()
try_test = try_test[(try_test["Chamber_Temp_degC"]==0)&(try_test["integral_I"]!=0)]
condition = (try_test["name"].str.contains("HWFET"))
try_test = try_test[~condition]
X_test = try_test[feature]

with open("{}/model_zero.binaryfile".format(models_path),"rb") as f:
  model_zero = pickle.load(f)

result_min0 = cycle_to_result(X_test,try_test,model_zero)

df_min0 = new_test_dfs[(new_test_dfs["Chamber_Temp_degC"]==0)&(new_test_dfs["integral_I"]!=0)]
result_min0_HWFET = df_min0[df_min0["name"].str.contains("HWFET")][["ID","integral_I"]]
result_min0_HWFET["SOC_pred"] = (2.9+result_min0_HWFET["integral_I"])/2.9*100
result_min0_HWFET = result_min0_HWFET.drop("integral_I",axis=1)
result_min0_zero = new_test_dfs[(new_test_dfs["integral_I"]==0)&(new_test_dfs["Chamber_Temp_degC"]==0)][["ID","integral_I"]]
result_min0_zero["SOC_pred"] = (2.9+result_min0_zero["integral_I"])/2.9*100
result_min0_zero = result_min0_zero.drop("integral_I",axis=1)
total_result_min0 = pd.concat([result_min0,result_min0_HWFET,result_min0_zero],axis=0)
total_result_min0 = total_result_min0.sort_values("ID").reset_index(drop=True)

## 追加分
copy_0 =new_test_dfs[new_test_dfs["Chamber_Temp_degC"]==0].sort_values("ID").reset_index()
total_result_min0["SOC_pred"] = (2.9+copy_0["integral_I"])/2.9*100
#### 10度 #####################

try_test = new_test_dfs.copy()
try_test = try_test[(try_test["Chamber_Temp_degC"]==10)&(try_test["integral_I"]!=0)]
condition = (try_test["name"].str.contains("UDDS"))
try_test = try_test[condition]
X_test = try_test[feature]

with open("{}/model_10_nonNN.binaryfile".format(models_path),"rb") as f:
  model_10_nonNN =  pickle.load(f)

result_10_UDDS = cycle_to_result(X_test,try_test,model_10_nonNN)

try_test = new_test_dfs.copy()
try_test = try_test[(try_test["Chamber_Temp_degC"]==10)&(try_test["integral_I"]!=0)]
condition = (try_test["name"].str.contains("LA92"))
try_test = try_test[condition]
X_test = try_test[feature]

with open("{}/model_10_NN.binaryfile".format(models_path),"rb") as f:
  model_10_NN = pickle.load(f)

result_10_LA92 = cycle_to_result(X_test,try_test,model_10_NN)


try_test = new_test_dfs.copy()
try_test = try_test[(try_test["Chamber_Temp_degC"]==10)&(try_test["integral_I"]!=0)]
condition = (try_test["name"].str.contains("UDDS|HWFET|LA92"))
try_test = try_test[~condition]
X_test = try_test[feature]


with open("{}/model_10_normal.binaryfile".format(models_path),"rb") as f:
  model_10_normal = pickle.load(f)

result_10_normal = cycle_to_result(X_test,try_test,model_10_normal)

df_10 = new_test_dfs[(new_test_dfs["Chamber_Temp_degC"]==10)&(new_test_dfs["integral_I"]!=0)]
result_10_HWFET = df_10[df_10["name"].str.contains("HWFET")][["ID","integral_I"]]
result_10_HWFET["SOC_pred"] = (2.9+result_10_HWFET["integral_I"])/2.9*100
result_10_HWFET = result_10_HWFET.drop("integral_I",axis=1)
result_10_zero = new_test_dfs[(new_test_dfs["integral_I"]==0)&(new_test_dfs["Chamber_Temp_degC"]==10)][["ID","integral_I"]]
result_10_zero["SOC_pred"] = (2.9+result_10_zero["integral_I"])/2.9*100
result_10_zero = result_10_zero.drop("integral_I",axis=1)
total_result_10 = pd.concat([result_10_normal,result_10_HWFET,result_10_zero,result_10_LA92,result_10_UDDS],axis=0)
total_result_10 = total_result_10.sort_values("ID").reset_index(drop=True)

#### 25度 #####################
try_test = new_test_dfs.copy()
try_test = try_test[(try_test["Chamber_Temp_degC"]==25)&(try_test["integral_I"]!=0)]
condition = (try_test["name"].str.contains("LA92"))
try_test = try_test[condition]
X_test = try_test[feature]

with open("{}/model_25_Cycle_1.binaryfile".format(models_path),"rb") as f:
  model_25_Cycle_1 = pickle.load(f)

result_25_LA92 = cycle_to_result(X_test,try_test,model_25_Cycle_1)

try_test = new_test_dfs.copy()
try_test = try_test[(try_test["Chamber_Temp_degC"]==25)&(try_test["integral_I"]!=0)]
condition = (try_test["name"].str.contains("UDDS"))
try_test = try_test[condition]
X_test = try_test[feature]

with open("{}/model_25_NN.binaryfile".format(models_path),"rb") as f:
  model_25_NN = pickle.load(f)

result_25_UDDS = cycle_to_result(X_test,try_test,model_25_NN)

df_25 = new_test_dfs[(new_test_dfs["Chamber_Temp_degC"]==25)&(new_test_dfs["integral_I"]!=0)]
result_25_normal = df_25[~df_25["name"].str.contains("LA92|UDDS")][["ID","integral_I"]]
result_25_normal["SOC_pred"] = (2.9+result_25_normal["integral_I"])/2.9*100
result_25_normal = result_25_normal.drop("integral_I",axis=1)
result_25_zero = new_test_dfs[(new_test_dfs["integral_I"]==0)&(new_test_dfs["Chamber_Temp_degC"]==25)][["ID","integral_I"]]
result_25_zero["SOC_pred"] = (2.9+result_25_zero["integral_I"])/2.9*100
result_25_zero = result_25_zero.drop("integral_I",axis=1)
total_result_25 = pd.concat([result_25_normal,result_25_zero,result_25_LA92,result_25_UDDS],axis=0)
total_result_25 = total_result_25.sort_values("ID").reset_index(drop=True)

################################
total_result = pd.concat([total_result_min20,total_result_min10,total_result_min0,total_result_10,total_result_25]).sort_values("ID").reset_index(drop=True)
total_result[total_result["SOC_pred"]>100]["SOC_pred"] = 100
submission["SOC"] = total_result["SOC_pred"]
submission.to_csv(new_submission_path,index=False)