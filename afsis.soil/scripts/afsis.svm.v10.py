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
idx=[i for i in range(1900,3578)]
[idx.pop(2654-1900) for i in range(2654,2669)]
xtrain, xtest = np.array(train)[:,idx], np.array(test)[:,idx]

# find lines highly correlated with labels
# find lines uncorrelated with each other
c=np.corrcoef(xtrain.transpose())
crossidx=[]
thresholds=[(.30,.30),(.10,.10),(.30,.30),(.45,.10),(.45,.10)]
for k in range(5):
	a=np.corrcoef(np.hstack((labels,xtrain)).transpose())[k,5:]
	colidx=[i for i in range(len(a)) if abs(a[i])>thresholds[k][0]]
	for i in range(len(colidx)):
		for j in range(i+1,len(colidx)):
			if abs(c[colidx[i]][colidx[j]])<thresholds[k][1]:
				crossidx.append(colidx[i])
				crossidx.append(colidx[j])

crossidx=list(set(crossidx))
crossidx.sort()
len(set(crossidx))

xtrain=xtrain[:,crossidx]
xtest=xtest[:,crossidx]

#center
for i in range(xtrain.shape[0]):
    xtrain[i,:]=xtrain[i,:]-xtrain[i,0]

for i in range(xtest.shape[0]):
    xtest[i,:]=xtest[i,:]-xtest[i,0]

# first pass
# spectra only
clf = svm.SVR(C=10000.0, verbose = 1)
preds1 = np.zeros((xtrain.shape[0], 5))
preds2 = np.zeros((xtest.shape[0], 5))
for i in range(5):
    clf.fit(xtrain, labels[:,i])
    preds1[:,i] = clf.predict(xtrain).astype(float)
    preds2[:,i] = clf.predict(xtest).astype(float)

# second pass
# spectra only + other preds
clf = svm.SVR(C=10000.0, verbose = 1)
for i in range(5):
    idx=[0,1,2,3,4]
    idx.pop(i)
    clf.fit( np.hstack((xtrain,preds1[:,idx])), labels[:,i])
    preds1[:,i] = clf.predict(np.hstack((xtrain,preds1[:,idx]))).astype(float)
    preds2[:,i] = clf.predict(np.hstack((xtest,preds2[:,idx]))).astype(float)
       
sample = pd.read_csv('../download/sample_submission.csv')
sample['Ca'] = preds2[:,0]
sample['P'] = preds2[:,1]
sample['pH'] = preds2[:,2]
sample['SOC'] = preds2[:,3]
sample['Sand'] = preds2[:,4]

sample.to_csv('../submissions/submit.svm.trunc1900.center0.noco2.pass2.corridx.v10.csv', index = False)



