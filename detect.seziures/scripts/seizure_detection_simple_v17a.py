#to automatically generate plots, call python with ipython --pylab
import scipy.io
import scipy.signal
import os
import matplotlib
import pandas as pd
import numpy as np
from sklearn import cross_validation
import matplotlib.pyplot as plt
from math import log

version='v2.03' # .92585, .89338, .62158 total variance in file, svm
version='v2.04' # .90460, .86897, .xxxxx variance vector across channels, svm
version='v2.05' # .91517, .88274, .xxxxx variance slope and intercept, svm
version='v2.06' # .91783, .89700, .xxxxx variance slope only, svm
version='v2.07' # .92034, .88730, .xxxxx variance slope and first value, svm
version='v2.08' # .93118, .91263, .61620 total entropy in file, svm 
version='v2.09' # .92111, .88193, .xxxxx entropy slope and intercept along channels, svm
version='v2.10' # .91563, .88216, .xxxxx entropy slope only along channels, svm
version='v2.11' # .90545, .86982, .xxxxx entropy slope and intercept across channels, svm
version='v2.12' # .xxxxx, .xxxxx, .xxxxx entropy slope only across channels, svm
version='v2.13' # .91783, .88025, .xxxxx mean area under the curve, time under curve in file, svm
version='v2.14' # .92690, .90937, .xxxxx total variance in file, logit
version='v2.15' # .91895, .89139, .xxxxx total ent in file, logit
version='v2.16' # .92778, .91016, .xxxxx total var,ent in file, logit
version='v2.17' # .94090, .93326, .xxxxx total var,ent,auc in file, logit
version='v2.17a' # .94090, .93326, .xxxxx total var,ent,auc in file, logit


version='submit.'+version

ntrainfiles=[596,1320,5240,3047, 174,3141,1041,210, 2745,2997,3521,1890]
ntestfiles=[3181,2997,4450,3013, 2050,3894,1281,543, 2986,2997,3601,1922]


# deleted bunch of stuff


#run stashfiles to create pickle files with pandas data panels
def stashfiles():
    for p in ['Dog_1', 'Dog_2', 'Dog_3', 'Dog_4', 'Patient_1','Patient_2', 'Patient_3', 'Patient_4', 'Patient_5', 'Patient_6', 'Patient_7', 'Patient_8']: #,'Dog_1', 'Dog_2', 'Dog_3', 'Dog_4','Patient_1', ]:
        print p
        #load data with test data and with downsampling
        data = doload(p, True,True)
            
        data.to_pickle('../data/'+p+'_moredownsampled.pkl')


#simple plotting routine
def plot(data):
    item = 'interictal_5'
    #time = range(len(data[item]['1']))
    time = data[item]['time']
    #plt.ion()
    plt.plot(time, data[item][0], 'k.-')
    plt.plot(time, data[item][1], 'b.-')
    plt.plot(time, data[item][2], 'r.-')
    #plt.show()
    raw_input('press a key')
    plt.close()    

def getEntropy(S):
    [m,n]=np.histogram(S, bins=[i for i in range(min(S)-1,max(S)+1)])
    p=1.*m/sum(m)+1e-6
    entropy=-1.*sum([p[i]*log(p[i],2) for i in range(len(p))])
    return entropy

# your machine learning magic goes here....    
from sklearn.linear_model import *
from sklearn.cross_validation import *
from sklearn.metrics import *
def dofitSVMstd(X_train, Y_train, X_test):
    shape=X_train.shape
    a= [np.std(X_train[i]) for i in range(shape[0])]
    x=np.zeros([shape[0],4])
    x[:,0]=a
    b=[]
    for j in range(shape[0]):
        a=X_train[j,:,:].astype(int)
        a.shape=(a.size,)
        b.append(getEntropy(list(a)))
    
    x[:,1]=b
    c=[]
    for j in range(shape[0]):
        pts=[]
        for k in range(shape[2]):
            brks=[0]+[i+1 for i in range(shape[1]-1) if ((X_train[j,i,k]>0) <> (X_train[j,i+1,k]>0))]+[shape[1]]
            parts=[(X_train[j,brks[i]:brks[i+1],k]) for i in range(len(brks)-1)]
            pts=pts+[[abs(sum(parts[i])),len(parts[i])] for i in range(len(parts))]
        c.append(np.mean(np.array(pts),axis=0))
    
    x[:,2]=[ci[0] for ci in c]
    x[:,3]=[ci[1] for ci in c]
    
    clf = LogisticRegression()
    dummy=clf.fit(x, Y_train)
    scores = cross_validation.cross_val_score(clf, x, Y_train)
    
    p1=clf.predict(x)
    shape=X_test.shape
    a= [np.std(X_test[i]) for i in range(shape[0])]
    x=np.zeros([shape[0],4])
    x[:,0]=a
    b=[]
    for j in range(shape[0]):
        a=X_test[j,:,:].astype(int)
        a.shape=(a.size,)
        b.append(getEntropy(list(a)))
    
    x[:,1]=b
    c=[]
    for j in range(shape[0]):
        pts=[]
        for k in range(shape[2]):
            brks=[0]+[i+1 for i in range(shape[1]-1) if ((X_test[j,i,k]>0) <> (X_test[j,i+1,k]>0))]+[shape[1]]
            parts=[(X_test[j,brks[i]:brks[i+1],k]) for i in range(len(brks)-1)]
            pts=pts+[[abs(sum(parts[i])),len(parts[i])] for i in range(len(parts))]
        c.append(np.mean(np.array(pts),axis=0))
    
    x[:,2]=[ci[0] for ci in c]
    x[:,3]=[ci[1] for ci in c]
    p2=clf.predict(x)
    return [scores,np.concatenate((p1,p2))]

def make_input():
    
    patients = ['Dog_1','Dog_2', 'Dog_3', 'Dog_4', 'Patient_1', 'Patient_2', 'Patient_3', 'Patient_4','Patient_5','Patient_6','Patient_7','Patient_8',] #, 'Dog_2']
    sub = pd.read_csv('../download/sampleSubmission.csv', delimiter=',')
    c=0
    scores=[]
    outfile=file('../submissions/partial_'+version+'.csv', 'wb')
    
    for p in patients:
        print 'loading patient data: ',p
        data = pd.read_pickle('../data/'+p+'_moredownsampled.pkl')
        channels = data['ictal_1'].keys()[0:-1]
        #
        z1=len([i for i in data.items if not 'interictal' in i and not 'test' in i])
        X_train = data.loc[['ictal_'+str(i+1) for i in range(z1)],:,channels].values
        x1_labels=[p+'_ictal_'+str(i+1) for i in range(z1)]
        #Y_train = (data.loc[[i for i in data.items if not 'interictal' in i and not 'test' in i],:,'time'].mean().values > 15).astype(int).reshape(-1)
        Y_train =  np.ones(X_train.shape[0])
        z2=len([i for i in data.items if 'interictal' in i])
        x2_labels=[p+'_interictal_'+str(i+1) for i in range(z2)]
        X_train2 = data.loc[['interictal_'+str(i+1) for i in range(z2)],:,channels].values
        Y_train2 = np.zeros(X_train2.shape[0])
        #
        test_range = range(1, 1+np.max([int(i[i.find('_')+1:]) for i in data.items if 'test' in i]) )
        fnums = range(1, 1+len([i for i in data.items if 'test' in i]))
        x3_labels=[p+'_test_'+str(i) for i in fnums]
        X_test = data.loc[['test_'+str(i) for i in fnums],:,channels].values
        #
        Y_train = np.concatenate((Y_train, Y_train2)).astype(int)
        X_train = np.append(X_train, X_train2, axis=0)
        #        
        preds = dofitSVMstd(X_train, Y_train, X_test)    
        Y_pred=preds[1]
        scores.append(sum(preds[0])/preds[0].shape[0])
        labels=x1_labels+x2_labels+x3_labels
        #
        pdf=pd.DataFrame([labels,list(preds[1])]).transpose()
        pdf.to_csv(outfile,index=False,header=False)
        
    outfile.close()

make_input()
