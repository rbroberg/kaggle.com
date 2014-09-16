
# coding: utf-8
# @author: Abhishek Thakur
# Beating the benchmark in Kaggle AFSIS Challenge.

import pandas as pd
import numpy as np
from sklearn import svm, cross_validation

train = pd.read_csv('../data/training.csv')
test = pd.read_csv('../data/sorted_test.csv')
labels = train[['Ca','P','pH','SOC','Sand']].values

train.drop(['Ca', 'P', 'pH', 'SOC', 'Sand', 'PIDN'], axis=1, inplace=True)
test.drop('PIDN', axis=1, inplace=True)

xtrain, xtest = np.array(train)[:,:3578], np.array(test)[:,:3578]
# truncate
#xtrain, xtest = np.array(train)[:,1800:3578], np.array(test)[:,1800:3578]

# filter outliers

#center
#for i in range(xtrain.shape[0]):
#    xtrain[i,:]=xtrain[i,:]-xtrain[i,0]

#for i in range(xtest.shape[0]):
#    xtest[i,:]=xtest[i,:]-xtest[i,0]

#sup_vec = svm.SVR(C=10000.0, verbose = 2)
sup_vec = svm.SVR(kernel='rbf', C=10000.0, gamma=0.1)
#
#ncv=7 # number of cv runs
#scores = np.zeros((5, ncv))
preds = np.zeros((xtest.shape[0], 5))

#cv = cross_validation.ShuffleSplit(xtrain.shape[0], n_iter=ncv, test_size=0.3, random_state=0)

import random
mcrmse=[]
for j in range(10):
	nidx=int(xtrain.shape[0]*0.75)
	ridx=np.random.choice(xtrain.shape[0], nidx, replace=False)
	train_data = xtrain[nidx,:]
	ridx_inv=[r for r in range(xtrain.shape[0])]
	dummy=[ridx_inv.remove(r) for r in ridx]
	#train_set = xtrain[ridx,:]
	#test_set = xtrain[ridx_inv,:]
	yobs=labels[ridx_inv,:]
	yhat = np.zeros((xtrain[ridx_inv,:].shape[0], 5))
	for i in range(5):
		#scores[i,:]=cross_validation.cross_val_score(sup_vec, xtrain, labels[:,i], cv=cv, n_jobs=-1)
		sup_vec.fit(xtrain[ridx,:], labels[ridx,i])
		yhat[:,i] = sup_vec.predict(xtrain[ridx_inv,:]).astype(float)
		#sup_vec.fit(xtrain, labels[:,i])
		#preds[:,i] = sup_vec.predict(xtest).astype(float)
	mcrmse.append(sum([(sum([(yobs[i,j]-yhat[i,j]) ** 2 for i in range(yobs.shape[0])])/yobs.shape[0])**0.5 for j in range(5)])/5)

print("mcrmse: ",np.mean(mcrmse), np.std(mcrmse))

sample = pd.read_csv('../download/sample_submission.csv')
sample['Ca'] = preds[:,0]
sample['P'] = preds[:,1]
sample['pH'] = preds[:,2]
sample['SOC'] = preds[:,3]
sample['Sand'] = preds[:,4]

sample.to_csv('../submissions/submit.svm.trunc1800.center0.csv', index = False)


