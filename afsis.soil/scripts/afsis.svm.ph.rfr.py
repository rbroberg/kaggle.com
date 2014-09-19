
# coding: utf-8
# @author: Abhishek Thakur
# Beating the benchmark in Kaggle AFSIS Challenge.

import pandas as pd
import numpy as np
from sklearn import svm, cross_validation
from sklearn.ensemble import RandomForestRegressor

train = pd.read_csv('../data/training.csv')
test = pd.read_csv('../data/sorted_test.csv')
labels = train[['Ca','P','pH','SOC','Sand']].values

train.drop(['Ca', 'P', 'pH', 'SOC', 'Sand', 'PIDN'], axis=1, inplace=True)
test.drop('PIDN', axis=1, inplace=True)

xtrain, xtest = np.array(train)[:,:3578], np.array(test)[:,:3578]
phtrain, phtest = np.array(train)[:,3578:3595], np.array(test)[:,3578:3595]
phtrain[:,15]=(phtrain[:,15]=='Topsoil')*1
phtest[:,15]=(phtest[:,15]=='Topsoil')*1

sup_vec = svm.SVR(C=10000.0, verbose = 2)
clf_rfr = RandomForestRegressor()

preds = np.zeros((xtest.shape[0], 5))
for i in range(5):
    sup_vec.fit(xtrain, labels[:,i])
    preds[:,i] = sup_vec.predict(xtest).astype(float)

# second model for pH
phtrain[:,0]=labels[:,0] # add Ca to list of pH features
phtest[:,0]=preds[:,0] # add Ca to list of pH features
clf_rfr.fit(phtrain,labels[:,2])
preds[:,2] = clf_rfr.predict(phtest).astype(float)

sample = pd.read_csv('../download/sample_submission.csv')
sample['Ca'] = preds[:,0]
sample['P'] = preds[:,1]
sample['pH'] = preds[:,2]
sample['SOC'] = preds[:,3]
sample['Sand'] = preds[:,4]

sample.to_csv('../submissions/all.svm.ph.rfr.csv', index = False)


