#!/usr/bin/env python
# coding: utf-8

import scipy.io
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score,precision_score,recall_score
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
import warnings
np.random.seed(1210)

## Shared
print("load patient 1")
# Load the mtx file as a sparse matrix
df = scipy.io.mmread('/cluster/projects/gaitigroup/Users/Yiyan/ParseBio_data/LMP1210/all/RNA_train.mtx')

# Import dataset
cell_region = pd.read_csv("/cluster/projects/gaitigroup/Users/Yiyan/ParseBio_data/LMP1210/all/region_train.csv")
feature = pd.read_csv("/cluster/projects/gaitigroup/Users/Yiyan/ParseBio_data/LMP1210/all/feature.csv")

cell_region['region'] = cell_region['region'].replace('Necrotic_core',0)
cell_region['region'] = cell_region['region'].replace('Solid_core',1)
cell_region['region'] = cell_region['region'].replace('Tumor_edge',2)
cell_region['region'] = cell_region['region'].replace('Normal_edge_cortex',3)

region = cell_region['region'].values

df.columns = cell_region['cellname'].values
df.index = feature['feature'].values
df = df.T
df = scipy.sparse.coo_matrix.tocsc(df)

features = feature['feature'].values

# Split dataset
X_training, X_temp, Y_training, Y_temp = train_test_split(df, region, test_size=0.3, random_state=42, stratify=region)
X_validation, X_test, Y_validation, Y_test = train_test_split(X_temp, Y_temp, test_size=0.66, random_state=42, stratify= Y_temp)

# Oversampling - SMOTE
from imblearn.over_sampling import SMOTE
# transform the dataset
oversample = SMOTE()
X_training, Y_training = oversample.fit_resample(X_training, Y_training)

### External validation
print("load exteral test")
# Load the mtx file as a sparse matrix
df_ex = scipy.io.mmread('/cluster/projects/gaitigroup/Users/Yiyan/ParseBio_data/LMP1210/all/RNA_ex_test.mtx')

# region
cell_region_ex = pd.read_csv("/cluster/projects/gaitigroup/Users/Yiyan/ParseBio_data/LMP1210/all/region_ex_test.csv")
cell_region_ex['region'] = cell_region_ex['region'].replace('Necrotic_core',0)
cell_region_ex['region'] = cell_region_ex['region'].replace('Solid_core',1)
cell_region_ex['region'] = cell_region_ex['region'].replace('Tumor_edge',2)
cell_region_ex['region'] = cell_region_ex['region'].replace('Normal_edge_cortex',3)
region_ex = cell_region_ex['region'].values

df_ex.columns = cell_region_ex['cellname'].values
df_ex.index = feature['feature'].values
df_ex = df_ex.T
df_ex = scipy.sparse.coo_matrix.tocsc(df_ex)


# ### Model 1: Linear regression
# print("SVM")
# 
# from sklearn.preprocessing import StandardScaler
# from sklearn.svm import SVC
# 
# def train_svm(X_train, Y_train, X_validation, Y_validation):
#     for gamma in [0.001, 0.01, 0.1, 1, 10, 100]:
#     	for C in [0.001, 0.01, 0.1, 1, 10, 100]:
#         	# for each combination of parameters, train an SVC
#         	svm = SVC(gamma=gamma, C=C)
#         	svm.fit(X_train, Y_train)
#         	# evaluate the SVC on the test set
#         	score = svm.score(X_validation, Y_validation)
#         	# if we got a better score, store the score and parameters
#         	if score > best_score:
#         		best_score = score
#         		best_parameters = {'C': C, 'gamma': gamma}
#         
#     print("Best score: {:.2f}".format(best_score))
#     print("Best parameters: {}".format(best_parameters))
#     return best_parameters
#     
# 
# best_parameters = train_svm(X_training, Y_training, X_validation, Y_validation)
# 
# svm_model = SVC(C=best_parameters['C'], gamma=best_parameters['gamma'])
# svm_model.fit(X_training, Y_training)
# 
# # Test
# test_accuracy = clf.score(X_test, Y_test)
# print("Its accuracy on the test data is " + str(test_accuracy))
#     
# # External test
# ex_test_accuracy = clf.score(df_ex, region_ex)
# print("Its accuracy on the external test data is " + str(ex_test_accuracy))

### Model 2: XGBoost
print("XGBoost")

# Define function
def train_and_score(X_train,X_validation,y_train,y_validation,param_dict):
  xg_cl = XGBClassifier(**param_dict,random_state=1210)
  xg_cl.fit(X_train,y_train)
  train_score = xg_cl.score(X_train,y_train)
  validation_score = xg_cl.score(X_validation,y_validation)
  y_pred = xg_cl.predict(X_validation)
  recall = recall_score(y_validation,y_pred,average = 'weighted')
  f1 = f1_score(y_validation,y_pred,average = 'weighted')
  precision = precision_score(y_validation,y_pred,average = 'weighted')
  return train_score,validation_score,recall,f1,precision

# All parameters for tuning
param_dict = dict(
  learning_rate=0.3,
  reg_alpha=0.0
)
n_estimators = [100,200,500]
max_depth = [3,6,9]
reg_lambda = [0.5,1.0,2.0]
best_score = 0
best_params = [None,None,None]
report = pd.DataFrame(columns=('n_estimators','max_depth','reg_lambda','train_score','validation_score','recall','f1','precision'))

# Tuning
for x in n_estimators:
  for y in max_depth:
    for z in reg_lambda:
      new_param_dict = param_dict.copy()
      new_param_dict["n_estimators"] = x
      new_param_dict["max_depth"] = y
      new_param_dict["reg_lambda"] = z
      train_score,validation_score,recall,f1,precision = train_and_score(X_training,X_validation,Y_training,Y_validation,new_param_dict)
      report = report.append({'n_estimators':x,'max_depth': y,'reg_lambda': z,'train_score':train_score,'validation_score':validation_score,'recall':recall,'f1':f1,'precision':precision},ignore_index=True)
      if validation_score > best_score:
        best_score = validation_score
        best_params = [x,y,z]
print(f"Best accuracy = {best_score}")
print(f"Best parameters: n_estimators={best_params[0]}, max_depth={best_params[1]}, reg_lambda={best_params[2]}")
report.head()
report.to_csv('XGBoost_tuning.csv', index=False)



def best_xgb(best_params,train,train_label,test,test_label,ex_test,ex_test_label):
	# Testing
	testing = pd.DataFrame(columns=('dataset','accuracy','recall','f1','precision'))
	
	# Training
	xg_cl = XGBClassifier(n_estimators=best_params[0], max_depth=best_params[1], reg_lambda=best_params[2],random_state=1210)
	xg_cl.fit(train,train_label)
	
	# Test
	test_accuracy = xg_cl.score(test,test_label)
	y_pred = xg_cl.predict(test)
	recall = recall_score(test_label,y_pred,average = 'weighted')
	f1 = f1_score(test_label,y_pred,average = 'weighted')
	precision = precision_score(test_label,y_pred,average = 'weighted')
	testing = testing.append({'dataset':"Test",'accuracy':test_accuracy,'recall':recall,'f1':f1,'precision':precision},ignore_index=True)

	# External test
	ex_test_accuracy = xg_cl.score(ex_test,ex_test_label)
	y_pred = xg_cl.predict(ex_test)
	recall = recall_score(ex_test_label,y_pred,average = 'weighted')
	f1 = f1_score(ex_test_label,y_pred,average = 'weighted')
	precision = precision_score(ex_test_label,y_pred,average = 'weighted')
	testing = testing.append({'dataset':"Ex_test",'accuracy':ex_test_accuracy,'recall':recall,'f1':f1,'precision':precision},ignore_index=True)
	
	# Save testing results
	testing.to_csv('XGBoost_testing.csv', index=False)
	
	sorted_idx = xg_cl.feature_importances_.argsort()[::-1]
	xg_cl_Importance = pd.DataFrame({"Feature":features[sorted_idx],"Importance":xg_cl.feature_importances_[sorted_idx]})
	xg_cl_Importance.to_csv('XGBoost_Importance.csv', index=False)
	
best_xgb(best_params, X_training, Y_training, X_test, Y_test, df_ex, region_ex)


### Model 3: Random forest
print("Random forest")

# Select model
def select_RandomForestClassifier_model(n_estimator,train,train_label,validation,validation_label):
    training_accuracy_list = []
    validation_accuracy_list = []
    validation_f1_list = []
    Estimate_Number = []
    ModelName = []
    validation_f1_max = 0
    for n_estimator in n_estimators:
    	ModelName.append('Random Forest')
    	Estimate_Number.append('Estimate Number ='+str(n_estimator))
    	clf = RandomForestClassifier(n_estimators=n_estimator)
    	clf.fit(train,train_label)
    	
    	# Training
    	training_accuracy = clf.score(train,train_label)
    	training_accuracy_list.append(training_accuracy)
    	
    	# Validation
    	validation_accuracy = clf.score(validation,validation_label)
    	validation_accuracy_list.append(validation_accuracy)
    	
    	# F1
    	y_pred = clf.predict(validation)
    	f1 = f1_score(validation_label,y_pred, average = 'weighted')
    	validation_f1_list.append(f1)
    	    	
    	# Pick n_estimator with max validation accuracy
    	if f1 > validation_f1_max:
    		optimum_n_estimator = n_estimator
    		validation_accuracy_max = validation_accuracy
    		training_accuracy_best_estimator = training_accuracy
    		validation_f1_max = f1
    		
    plt.figure()
    plt.plot(n_estimators, training_accuracy_list, linestyle='-', marker='o')
    plt.plot(n_estimators, validation_accuracy_list, linestyle='-', marker='o')
    plt.xticks(n_estimators)
    plt.legend(labels = ["training", "validation"])
    plt.xlabel('n_estimators')
    plt.ylabel('Accuracy')
    plt.show()
    plt.savefig("RF_select.png")
    plt.close()
    
    print("The model with the best validation accuracy is n_estimator=" + str(optimum_n_estimator))
    
    RF_Results = pd.DataFrame({"Model":ModelName,"Feature":Estimate_Number,"Training Accuracy":training_accuracy_list,"Validation Accuracy":validation_accuracy_list,"Validation f1":validation_f1_list})
    return RF_Results,optimum_n_estimator

# Parameters
n_estimators = [10,20,50,100,500]
RF_select, optimum_n_estimator = select_RandomForestClassifier_model(n_estimators, X_training, Y_training, X_validation, Y_validation)
print(RF_select)
RF_select.to_csv('RF_select.csv', index=False)


def best_RandomForestClassifier(optimum_n_estimator,train,train_label,test,test_label,ex_test,ex_test_label):
	# Testing
	testing = pd.DataFrame(columns=('dataset','accuracy','recall','f1','precision'))
	
	# Training
	clf = RandomForestClassifier(n_estimators=optimum_n_estimator)
	clf.fit(train,train_label)
	
	# Test
	test_accuracy = clf.score(test,test_label)
	y_pred = clf.predict(test)
	recall = recall_score(test_label,y_pred,average = 'weighted')
	f1 = f1_score(test_label,y_pred,average = 'weighted')
	precision = precision_score(test_label,y_pred,average = 'weighted')
	testing = testing.append({'dataset':"Test",'accuracy':test_accuracy,'recall':recall,'f1':f1,'precision':precision},ignore_index=True)
	
	# External test
	ex_test_accuracy = clf.score(ex_test,ex_test_label)
	y_pred = clf.predict(ex_test)
	recall = recall_score(ex_test_label,y_pred,average = 'weighted')
	f1 = f1_score(ex_test_label,y_pred,average = 'weighted')
	precision = precision_score(ex_test_label,y_pred,average = 'weighted')
	testing = testing.append({'dataset':"Ex_test",'accuracy':ex_test_accuracy,'recall':recall,'f1':f1,'precision':precision},ignore_index=True)
	testing.to_csv('RF_testing.csv', index=False)
	
	sorted_idx = clf.feature_importances_.argsort()[::-1]
	RF_Importance = pd.DataFrame({"Feature":features[sorted_idx],"Importance":clf.feature_importances_[sorted_idx]})
	RF_Importance.to_csv('RF_Importance.csv', index=False)
	
best_RandomForestClassifier(optimum_n_estimator,X_training, Y_training, X_test, Y_test, df_ex, region_ex)

### Model 4:MLP
print("MLP")

# Define function
def train_and_score(X_train,X_validation,y_train,y_validation,param_dict):
  mlp=MLPClassifier(**param_dict,random_state=1210)
  mlp.fit(X_train,y_train)
  train_score = mlp.score(X_train,y_train)
  validation_score = mlp.score(X_validation,y_validation)
  y_pred = mlp.predict(X_validation)
  recall = recall_score(y_validation,y_pred,average = 'weighted')
  f1 = f1_score(y_validation,y_pred,average = 'weighted')
  precision = precision_score(y_validation,y_pred,average = 'weighted')
  return train_score,validation_score,recall,f1,precision

# All parameters for tuning
param_dict = dict(
  activation="relu",
  max_iter=100,
)
early_stopping = [True]
hidden_layer_sizes = [(10,),(100,),(10,10),(10,100),(100,10),(100,100)]
alpha = [0.001,0.01,0.1]
# hidden_layer_sizes = [(10,10)]
# alpha = [0.01]
# note: running this loop may take a few seconds
best_score = 0
best_params = [None,None,None]
report = pd.DataFrame(columns=('early_stopping','hidden_layer_sizes','alpha','train_score','validation_score','recall','f1','precision'))

# Tuning
for x in early_stopping:
  for y in hidden_layer_sizes:
    for z in alpha:
      new_param_dict = param_dict.copy()
      new_param_dict["early_stopping"] = x
      new_param_dict["hidden_layer_sizes"] = y
      new_param_dict["alpha"] = z
      train_score,validation_score,recall,f1,precision = train_and_score(X_training,X_validation,Y_training,Y_validation,new_param_dict)
      report = report.append({'early_stopping':x,'hidden_layer_sizes': y,'alpha': z,'train_score':train_score,'validation_score':validation_score,'recall':recall,'f1':f1,'precision':precision},ignore_index=True)
      if validation_score > best_score:
        best_score = validation_score
        best_params = [x,y,z]
print(f"Best accuracy = {best_score}")
print(f"Best parameters: early_stopping={best_params[0]}, hidden_layer_sizes={best_params[1]}, alpha={best_params[2]}")
report.head()
report.to_csv('MLP_tuning.csv', index=False)


def best_MLP(best_params,train,train_label,test,test_label,ex_test,ex_test_label):
	# Testing
	testing = pd.DataFrame(columns=('dataset','accuracy','recall','f1','precision'))
	
	# Training
	mlp = MLPClassifier(early_stopping=best_params[0], hidden_layer_sizes=best_params[1], alpha=best_params[2],random_state=1210)
	mlp.fit(train,train_label)
	
	# Test
	test_accuracy = mlp.score(test,test_label)
	y_pred = mlp.predict(test)
	recall = recall_score(test_label,y_pred,average = 'weighted')
	f1 = f1_score(test_label,y_pred,average = 'weighted')
	precision = precision_score(test_label,y_pred,average = 'weighted')
	testing = testing.append({'dataset':"Test",'accuracy':test_accuracy,'recall':recall,'f1':f1,'precision':precision},ignore_index=True)

	# External test
	ex_test_accuracy = mlp.score(ex_test,ex_test_label)
	y_pred = mlp.predict(ex_test)
	recall = recall_score(ex_test_label,y_pred,average = 'weighted')
	f1 = f1_score(ex_test_label,y_pred,average = 'weighted')
	precision = precision_score(ex_test_label,y_pred,average = 'weighted')
	testing = testing.append({'dataset':"Ex_test",'accuracy':ex_test_accuracy,'recall':recall,'f1':f1,'precision':precision},ignore_index=True)
	testing.to_csv('MLP_testing.csv', index=False)
	
best_MLP(best_params, X_training, Y_training, X_test, Y_test, df_ex, region_ex)