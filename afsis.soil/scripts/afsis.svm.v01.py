
# coding: utf-8
# @author: Abhishek Thakur
# Beating the benchmark in Kaggle AFSIS Challenge.

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

# truncate
xtrain, xtest = np.array(train)[:,1800:3578], np.array(test)[:,1800:3578]


#center
for i in range(xtrain.shape[0]):
    xtrain[i,:]=xtrain[i,:]-xtrain[i,0]

for i in range(xtest.shape[0]):
    xtest[i,:]=xtest[i,:]-xtest[i,0]

# filter outliers
nouts=np.zeros(labels.shape)
for i in range(5):
    nouts[:,i] = (labels[:,i] < labels[:,i].std() + 3*labels[:,i].std()) & (labels[:,i] > labels[:,i].std() - 3*labels[:,i].std())

sums=np.sum(nouts,axis=1)==5
idx=[x for x in range(1157) if sums[x] ]

#sup_vec = svm.SVR(C=100.0, verbose = 1, degree=3) # very bad
#sup_vec = svm.SVR(C=100000.0, verbose = 1, degree=3) # runs much longer
#clf = Ridge(alpha=1.0)
#clf = Lasso(alpha=0.1,max_iter=1e7, tol=1e-6)
#clf = AdaBoostClassifier(n_estimators=100)
sup_vec = svm.SVR(C=10000.0, verbose = 1, degree=3, gamma=.1) # .70 split; .5493, .0715
sup_vec = svm.SVR(C=10000.0, verbose = 1, degree=3, gamma=.05) # .70 split; .5248, .0303
sup_vec = svm.SVR(C=10000.0, verbose = 1, degree=4, gamma=.05) # .70 split; .5622, .0724
sup_vec = svm.SVR(C=10000.0, verbose = 1, degree=3, gamma=.05, tol=.0001) # .70 split; .5561, .0523
sup_vec = svm.SVR(C=100000.0, verbose = 1, degree=3, gamma=.05) # .70 split; .5295, 0.0281
sup_vec = svm.SVR(C=10000.0, verbose = 1, degree=3, gamma=.05, epsilon=.01) # .70 split; .5271, .0384
sup_vec = svm.SVR(C=10000.0, verbose = 1, degree=3, gamma=.01) # .70 split; .6067, .407

#ncv=7 # number of cv runs
#scores = np.zeros((5, ncv))
preds = np.zeros((xtest.shape[0], 5))

#cv = cross_validation.ShuffleSplit(xtrain.shape[0], n_iter=ncv, test_size=0.3, random_state=0)

'''
# best cv model to date: svm.SVR(C=100000.0, verbose = 1, degree=3), 1800:3578, zeroed, 3*std filter
import random
mcrmse=[]
for j in range(10):
	nidx=int(xtrain.shape[0]*0.70)
	ridx=np.random.choice(xtrain.shape[0], nidx, replace=False)
	train_data = xtrain[nidx,:]
	ridx_inv=[r for r in range(xtrain.shape[0])]
	dummy=[ridx_inv.remove(r) for r in ridx]
	yobs=labels[ridx_inv,:]
	yhat = np.zeros((xtrain[ridx_inv,:].shape[0], 5))
	for i in range(5):
		sup_vec.fit(xtrain[ridx,:], labels[ridx,i])
		yhat[:,i] = sup_vec.predict(xtrain[ridx_inv,:]).astype(float)
		#clf.fit(xtrain[ridx,:], labels[ridx,i])
		#yhat[:,i] = clf.predict(xtrain[ridx_inv,:]).astype(float)
	mcrmse.append(sum([(sum([(yobs[i,j]-yhat[i,j]) ** 2 for i in range(yobs.shape[0])])/yobs.shape[0])**0.5 for j in range(5)])/5)

print("mcrmse: ",np.mean(mcrmse), np.std(mcrmse))
'''
# for logreg
#mcrmse=[]
clf = LogisticRegression(C=10000)
for j in range(10):
	clf.fit(xtrain[ridx,:], labels[ridx,:])
	yhat = clf.predict(xtrain[ridx_inv,:]).astype(float)
	mcrmse.append(sum([(sum([(yobs[i,j]-yhat[i,j]) ** 2 for i in range(yobs.shape[0])])/yobs.shape[0])**0.5 for j in range(5)])/5)


for i in range(5):
    #sup_vec.fit(xtrain[idx,:], labels[idx,i])
    #preds[:,i] = sup_vec.predict(xtest).astype(float)
    clf.fit(xtrain[idx,:], labels[idx,i])
    preds[:,i] = clf.predict(xtest).astype(float)

sample = pd.read_csv('../download/sample_submission.csv')
sample['Ca'] = preds[:,0]
sample['P'] = preds[:,1]
sample['pH'] = preds[:,2]
sample['SOC'] = preds[:,3]
sample['Sand'] = preds[:,4]

sample.to_csv('../submissions/submit.svm.trunc1800.center0.norm.csv', index = False)


