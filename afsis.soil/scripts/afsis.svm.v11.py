import pandas as pd
import numpy as np

train = pd.read_csv('../data/training.csv')
test = pd.read_csv('../data/sorted_test.csv')
labels = train[['Ca','P','pH','SOC','Sand']].values

train.drop(['Ca', 'P', 'pH', 'SOC', 'Sand', 'PIDN'], axis=1, inplace=True)
test.drop('PIDN', axis=1, inplace=True)

# truncate
idx=[i for i in range(1900,3578)]
[idx.pop(2654-1900) for i in range(2654,2669)]
xtrain, xtest = np.array(train)[:,idx], np.array(test)[:,idx]

#center
for i in range(xtrain.shape[0]):
    xtrain[i,:]=xtrain[i,:]-xtrain[i,0]

for i in range(xtest.shape[0]):
    xtest[i,:]=xtest[i,:]-xtest[i,0]

x=xtrain[0,:]
y=labels[0,:]
x.shape=(1663,1)
y.shape=(1,5)
W=np.dot(x,y)
for i in range(1,1157):
    x=xtrain[i,:]
    y=labels[i,:]
    x.shape=(1663,1)
    y.shape=(1,5)
    W=W+np.dot(x,y)
    
x=xtrain[0,:]
x.shape=(1663,1)
W.T*x



sample = pd.read_csv('../download/sample_submission.csv')
sample['Ca'] = preds2[:,0]
sample['P'] = preds2[:,1]
sample['pH'] = preds2[:,2]
sample['SOC'] = preds2[:,3]
sample['Sand'] = preds2[:,4]

sample.to_csv('../submissions/submit.svm.trunc1900.center0.noco2.pass2.corridx.v10.csv', index = False)



