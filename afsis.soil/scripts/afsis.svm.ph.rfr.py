
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

xtrain, xtest = np.array(train)[:,1800:3578], np.array(test)[:,1800:3578]
phtrain, phtest = np.array(train)[:,1800:3595], np.array(test)[:,1800:3595]
phtrain[:,-1]=(phtrain[:,-1]=='Topsoil')*1
phtest[:,-1]=(phtest[:,-1]=='Topsoil')*1

#center
for i in range(xtrain.shape[0]):
    xtrain[i,:]=xtrain[i,:]-xtrain[i,0]
    phtrain[i,:-1]=phtrain[i,:-1]-phtrain[i,0]

for i in range(xtest.shape[0]):
    xtest[i,:]=xtest[i,:]-xtest[i,0]
    phtest[i,:-1]=phtest[i,:-1]-phtest[i,0]

sup_vec = svm.SVR(C=10000.0, verbose = 2)
sup_vec2 = svm.SVR(C=10000.0, verbose = 2)
#clf_rfr = RandomForestRegressor()

preds = np.zeros((xtest.shape[0], 5))
for i in range(5):
    sup_vec.fit(xtrain, labels[:,i])
    preds[:,i] = sup_vec.predict(xtest).astype(float)

# second model for pH
phtrain[:,0]=labels[:,0] # add Ca to list of pH features
phtest[:,0]=preds[:,0] # add Ca to list of pH features
sup_vec2.fit(phtrain,labels[:,1])
preds[:,1] = sup_vec2.predict(phtest).astype(float)

sample = pd.read_csv('../download/sample_submission.csv')
sample['Ca'] = preds[:,0]
sample['P'] = preds[:,1]
sample['pH'] = preds[:,2]
sample['SOC'] = preds[:,3]
sample['Sand'] = preds[:,4]

sample.to_csv('../submissions/all.svm.p.svm.csv', index = False)


