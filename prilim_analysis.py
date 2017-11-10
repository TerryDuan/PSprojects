# -*- coding: utf-8 -*-
"""
Created on Thu Oct 19 22:58:32 2017

@author: Duan
"""
import numpy as np
import pandas as pd
from sklearn import *
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt 

#load the data
raw_data = pd.read_csv('data/train.csv')

list(raw_data.target).count(1)
#21,694
list(raw_data.target).count(0)
#573,518

plt.matshow(raw_data.corr())
raw_corr = raw_data.corr()

#seperate 1 and 0 for analysis
raw_data_set_1 = raw_data.ix[raw_data.target == 1]
raw_data_set_0 = raw_data.ix[raw_data.target == 0]

#data analysis
#completeness
column_list = list(raw_data.columns)
for i, cols in enumerate(column_list):
    print(cols,' ',list(raw_data[cols]).count(-1)/595212)

#should we exclude these three?    
#ps_reg_03
#ps_car_03_cat
#ps_car_05_cat


#distribution
for i in range(len(column_list)-2):
    print(column_list[i+2])
    plt.bar(raw_data[column_list[i+2]].unique(), raw_data.groupby(column_list[i+2])[column_list[0]].nunique() )
    plt.show()
#replace missing value?
#any cluster
#colinearity?


#initial split
raw_data_set_test_idx = np.random.choice(len(raw_data), size = 50000, replace = False)
raw_data_set_test_tf = raw_data.index.isin(raw_data_set_test_idx)
dev_set = raw_data[raw_data_set_test_tf]
train_set = raw_data[~raw_data_set_test_tf]


#first run with logistic regression with L1, on all columns
C_list = [0.00001,0.0001, 0.001, 0.01, 0.1, 0.25, 0.5, 1]
class_weights = [{0:0.5, 1:0.5}, {0:0.3, 1:0.7},{0:0.1, 1:0.9},{0:0.05,1:0.95}]
grids = {'C':[],'weights_on_1':[],'train_set_score':[], 'train_set_false_true':[], 'dev_set_score':[], 'dev_set_false_true':[]}

for i, c in enumerate(C_list):
    for j, w in enumerate(class_weights):
        grids['C'].append(c)
        grids['weights_on_1'].append(w)
        clf_l1_LogitReg = LogisticRegression(C=c,class_weight = w, penalty='l1', tol=0.0001, max_iter = 1000)
        logit_l1 = clf_l1_LogitReg.fit(train_set.iloc[:,2:], train_set.iloc[:,1])
        grids['train_set_score'].append(logit_l1.score(train_set.iloc[:,2:], train_set.iloc[:,1]))
        compare_df = pd.DataFrame({'y':train_set.target, 'y_hat':logit_l1.predict(train_set.iloc[:,2:])})
        grids['train_set_false_true'].append(len(compare_df.ix[(compare_df.y_hat == compare_df.y) & (compare_df.y == 1)])/len(compare_df.ix[(compare_df.y == 1)]))
        
        grids['dev_set_score'].append(logit_l1.score(dev_set.iloc[:,2:], dev_set.iloc[:,1]))
        compare_df = pd.DataFrame({'y':dev_set.target, 'y_hat':logit_l1.predict(dev_set.iloc[:,2:])})
        grids['dev_set_false_true'].append(len(compare_df.ix[(compare_df.y_hat == compare_df.y) & (compare_df.y == 1)])/len(compare_df.ix[(compare_df.y == 1)]))

        
clf_l1_LogitReg = LogisticRegression(C=1, penalty='l1', tol=0.0001, max_iter = 1000)
logit_l1 = clf_l1_LogitReg.fit(train_set.iloc[:,2:], train_set.iloc[:,1])



from NaiveNN import *
