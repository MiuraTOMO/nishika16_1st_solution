# -*- coding: utf-8 -*-
import os
import numpy as np
import random 
import pickle
import warnings
import json

import pandas as pd
from sklearn.metrics import mean_squared_error as mse
from sklearn.linear_model import Ridge
warnings.simplefilter("ignore")

####################### function #######################
def seed_everything(seed=42):
  random.seed(seed)
  os.environ["PYTHONHASHSEED"] = str(seed)
  np.random.seed(seed)

seed_everything()
############################ main ######################
with open("./setting.json","r") as f:
    setting = json.load(f)


train_prepreocess_path = setting["TRAIN_PROCESSED_DATA_PATH"]
test_preprocess_path = setting["TEST_PROCESSED_DATA_PATH"]
models_path = setting["MODEL_CHECKPOINT_DIR"]

basic_feature = ["integral_I","Battery_Temp_degC"]
cycle_feature = ["integral_cycle","Temp_cycle"]
feature = cycle_feature+basic_feature
target = "Ah_cycle"

with open(train_prepreocess_path,"rb") as f:
  new_train_dfs = pickle.load(f)

with open(test_preprocess_path,"rb") as f:
  new_test_dfs = pickle.load(f)

new_train_dfs["SOC"] = (2.9+new_train_dfs["Ah"])/2.9*100
new_train_dfs_type = pd.get_dummies(new_train_dfs["type"])
new_train_dfs = pd.concat([new_train_dfs,new_train_dfs_type])

##### -20度 ###################
try_data = new_train_dfs[(new_train_dfs["Chamber_Temp_degC"]==-20)&(new_train_dfs["integral_I"]!=0)]
X_train,y_train = try_data[feature],try_data[target]
model_min20 = Ridge(alpha=1e-5,normalize=True)
model_min20.fit(X_train,y_train)

with open("{}/model_min20.binaryfile".format(models_path),"wb") as f:
  pickle.dump(model_min20,f)

#### -10度 #####################
try_data = new_train_dfs[(new_train_dfs["Chamber_Temp_degC"]==-10)&(new_train_dfs["integral_I"]!=0)]
try_data = try_data[try_data["name"].str.contains("Cycle")]
X_train,y_train = try_data[feature],try_data[target]
model_min10_Cycle = Ridge(alpha=1e-3,normalize=True)
model_min10_Cycle.fit(X_train,y_train)

with open("{}/model_min10_Cycle.binaryfile".format(models_path),"wb") as f:
  pickle.dump(model_min10_Cycle,f)

try_data = new_train_dfs[(new_train_dfs["Chamber_Temp_degC"]==-10)&(new_train_dfs["integral_I"]!=0)]
try_data = try_data[~try_data["name"].str.contains("Cycle")]
X_train,y_train = try_data[feature],try_data[target]
model_min10_nonCycle = Ridge(alpha=3e-4,normalize=True)
model_min10_nonCycle.fit(X_train,y_train)

with open("{}/model_min10_nonCycle.binaryfile".format(models_path),"wb") as f:
  pickle.dump(model_min10_nonCycle,f)

#### 0度  ######################
try_data = new_train_dfs[(new_train_dfs["Chamber_Temp_degC"]==0)&(new_train_dfs["integral_I"]!=0)]
X_train,y_train = try_data[feature],try_data[target]
model_zero = Ridge(alpha=1e-5,normalize=True)
model_zero.fit(X_train,y_train)

with open("{}/model_zero.binaryfile".format(models_path),"wb") as f:
  pickle.dump(model_zero,f)

#### 10度 #####################
try_data = new_train_dfs[(new_train_dfs["Chamber_Temp_degC"]==10)&(new_train_dfs["integral_I"]!=0)]
condition = try_data["name"].str.contains("10_NN")
try_data = try_data[~condition]
X_train,y_train = try_data[feature],try_data[target]
model_10_nonNN = Ridge(alpha=1e-5,normalize=True)
model_10_nonNN.fit(X_train,y_train)

with open("{}/model_10_nonNN.binaryfile".format(models_path),"wb") as f:
  pickle.dump(model_10_nonNN,f)

try_data = new_train_dfs[(new_train_dfs["Chamber_Temp_degC"]==10)&(new_train_dfs["integral_I"]!=0)]
condition = try_data["name"].str.contains("10_NN")
try_data = try_data[condition]
X_train,y_train = try_data[feature],try_data[target]
model_10_NN = Ridge(alpha=8e-4,normalize=True)
model_10_NN.fit(X_train,y_train)

with open("{}/model_10_NN.binaryfile".format(models_path),"wb") as f:
  pickle.dump(model_10_NN,f)

try_data = new_train_dfs[(new_train_dfs["Chamber_Temp_degC"]==10)&(new_train_dfs["integral_I"]!=0)]
condition = try_data["name"].str.contains("10_NN")
try_data = try_data[~condition]
X_train,y_train = try_data[feature],try_data[target]
model_10_normal = Ridge(alpha=1e-5,normalize=True)
model_10_normal.fit(X_train,y_train)

with open("{}/model_10_normal.binaryfile".format(models_path),"wb") as f:
  pickle.dump(model_10_normal,f)

#### 25度 #####################
try_data = new_train_dfs[(new_train_dfs["Chamber_Temp_degC"]==25)&(new_train_dfs["integral_I"]!=0)]
condition = try_data["name"].str.contains("NN")
try_data = try_data[condition]
X_train,y_train = try_data[feature],try_data[target]
model_25_NN = Ridge(alpha=1e-5,normalize=True)
model_25_NN.fit(X_train,y_train)

with open("{}/model_25_NN.binaryfile".format(models_path),"wb") as f:
  pickle.dump(model_25_NN,f)

try_data = new_train_dfs[(new_train_dfs["Chamber_Temp_degC"]==25)&(new_train_dfs["integral_I"]!=0)]
condition = try_data["name"].str.contains("Cycle")
try_data = try_data[condition]
X_train,y_train = try_data[feature],try_data[target]
model_25_Cycle_1 = Ridge(alpha=6e-4,normalize=True)
model_25_Cycle_1.fit(X_train,y_train)

with open("{}/model_25_Cycle_1.binaryfile".format(models_path),"wb") as f:
  pickle.dump(model_25_Cycle_1,f)


















