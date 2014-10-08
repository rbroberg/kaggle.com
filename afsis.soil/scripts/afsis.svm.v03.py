
# coding: utf-8
# @author: Abhishek Thakur
# Beating the benchmark in Kaggle AFSIS Challenge.

# v02: take top 15 and bottom 15 and retrain

import pandas as pd
import numpy as np
from sklearn import svm, cross_validation
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier

train = pd.read_csv('../data/training.csv')
test = pd.read_csv('../data/sorted_test.csv')
labels = train[['Ca','P','pH','SOC','Sand']].values

train.drop(['Ca', 'P', 'pH', 'SOC', 'Sand', 'PIDN'], axis=1, inplace=True)
test.drop('PIDN', axis=1, inplace=True)

# xtrain, xtest = np.array(train)[:,:3578], np.array(test)[:,:3578]

# The reflectance spectra were reduced to 410-2450 nm in order to eliminate the noise at the edges of each spectrum
# weakened overall score
# xtrain, xtest = np.array(train)[:,2600:3578], np.array(test)[:,2600:3578]

#co2_cols = ['m2379.76', 'm2377.83', 'm2375.9', 'm2373.97',
#'m2372.04', 'm2370.11', 'm2368.18', 'm2366.26',
#'m2364.33', 'm2362.4', 'm2360.47', 'm2358.54',
#'m2356.61', 'm2354.68', 'm2352.76']
#b=train.iloc[1,:]==train['m2379.76'][1]
#list(b).index(max(b)) # 2654
#train.columns[2654:2669]

# truncate
idx=[i for i in range(1800,3578)]
[idx.pop(2654-1800) for i in range(2654,2669)]
xtrain, xtest = np.array(train)[:,idx], np.array(test)[:,idx]

#center
for i in range(xtrain.shape[0]):
    xtrain[i,:]=xtrain[i,:]-xtrain[i,0]

for i in range(xtest.shape[0]):
    xtest[i,:]=xtest[i,:]-xtest[i,0]

# filter outliers
nouts=np.zeros(labels.shape)
for i in range(5):
    nouts[:,i] = (labels[:,i] < labels[:,i].std() + 3*labels[:,i].std()) & (labels[:,i] > labels[:,i].std() - 3*labels[:,i].std())


clf = svm.SVR(C=10000.0, verbose = 1)
preds = np.zeros((xtest.shape[0], 5))
for i in range(5):
    nidx = [j for j in range(xtest.shape[0]) if nouts[j,i]]
    clf.fit(xtrain[nidx,:], labels[nidx,i])
    preds[:,i] = clf.predict(xtest).astype(float)


sample = pd.read_csv('../download/sample_submission.csv')
sample['Ca'] = preds[:,0]
sample['P'] = preds[:,1]
sample['pH'] = preds[:,2]
sample['SOC'] = preds[:,3]
sample['Sand'] = preds[:,4]

sample.to_csv('../submissions/submit.svm.trunc1800.center0.norm.noco2.v03.csv', index = False)


