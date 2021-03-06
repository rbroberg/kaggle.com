#to automatically generate plots, call python with ipython --pylab
import scipy.io
import scipy.signal
import os
import matplotlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from math import log
import pywt

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
version='v2.17' # .94090, .93326, .69687 total var,ent,auc in file, logit
version='v2.18' # .91933, .89850, .xxxxx total dwtefd in file, logit
version='v2.19' # .93851, .90663, .xxxxx variance vector along channels, logit
version='v2.20' # .93700, .91453, .xxxxx variance vector along 16 channels, logit
version='v2.21' # .94287, .91812, .xxxxx variance,ent vectors along 16 channels, logit
version='v2.22' # .91606, .87693, .xxxxx variance,ent vectors along 16 channels, SVC
version='v2.23' # .94287, .91812, .xxxxx variance,ent vectors along 16 channels, scaled, SVC
version='v2.24' # .92585, .89338, .xxxxx total variance in file, Lasso
    
version='submit.'+version+'.csv'

ntrainfiles=[596,1320,5240,3047, 174,3141,1041,210, 2745,2997,3521,1890]
ntestfiles=[3181,2997,4450,3013, 2050,3894,1281,543, 2986,2997,3601,1922]


#data = doload('Dog_1', False, False)
def doload(patient, incl_test, downsample):
    dir = '../data/'+ patient + '/'
    dict = {}
    
    #load files in numerical order
    files = os.listdir(dir)
    files2 =[]
    
    for i in range(len(files)):
        qp = files[i].rfind('_') +1
        files2.append( files[i][0:qp] + (10-len(files[i][files[i].rfind('_')+1:]) )*'0' + files[i][qp:] )
                
    print len(files), len(files2)
    t={}
    for key,value in zip(files2,files):
        t[key]=value
    #t = {key:value for key, value in zip(files2,files)}
    files2 = t.keys()
    files2.sort()
    f = [t[i] for i in files2]
    
    
    j = 0
    for i in f:
        
        if not 'test' in i or incl_test:
            seg = i[i.rfind('_')+1 : i.find('.mat')]
            segtype = i[i[0:i.find('_segment')].rfind('_')+1: i.find('_segment')]
            print i
            d = scipy.io.loadmat(dir+i)
            if j==0:
                cols = range(len(d['channels'][0,0]))
                cols = cols +['time']
            
            if  'inter' in i or 'test' in i:
                l = -3600.0#np.nan
            else:
                #print i
                l = d['latency'][0]
                
            df = pd.DataFrame(np.append(d['data'].T, l+np.array([range(len(d['data'][1]))]).T/d['freq'][0], 1 ), index=range(len(d['data'][1])), columns=cols)
            
            if downsample:
                if np.round(d['freq'][0]) == 5000:
                    df = df.groupby(lambda x: int(np.floor(x/20.0))).mean()
                if np.round(d['freq'][0]) == 500:
                    df = df.groupby(lambda x: int(np.floor(x/2.0))).mean()    
                if np.round(d['freq'][0]) == 400:
                    df = df.groupby(lambda x: int(np.floor(x/2.0))).mean()                    
                
                df['time'] = df['time'] - (df['time'][0]-np.floor(df['time'][0]))*(df['time'][0] > 0)
            
            dict.update({segtype+'_'+seg : df})
            
            j = j +1
            
    data = pd.Panel(dict)
    
    return data



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
from sklearn import cross_validation
from sklearn import svm
def dofitSVMstd(X_train, Y_train, X_test):
    from sklearn import svm
    X_train = [np.std(X_train[i]) for i in range(X_train.shape[0])]
    x=np.zeros([len(X_train),1])
    x[:,0]=X_train
    clf=Lasso(alpha=1.0,normalize=True)
    dummy=clf.fit(x, Y_train)
    scores = cross_validation.cross_val_score(clf, x, Y_train)
    X_test = [np.std(X_test[i]) for i in range(X_test.shape[0])]
    x=np.zeros([len(X_test),1])
    x[:,0]=X_test
    return [scores,clf.predict(x)]

def make_submission():
    
    patients = ['Dog_1','Dog_2', 'Dog_3', 'Dog_4', 'Patient_1', 'Patient_2', 'Patient_3', 'Patient_4','Patient_5','Patient_6','Patient_7','Patient_8',] #, 'Dog_2']
    sub = pd.read_csv('../download/sampleSubmission.csv', delimiter=',')
    c=0
    scores=[]
    for p in patients:
        print 'loading patient data: ',p
        data = pd.read_pickle('../data/'+p+'_moredownsampled.pkl')
        channels = data['ictal_1'].keys()[0:-1]
        #
        X_train = data.loc[[i for i in data.items if not 'interictal' in i and not 'test' in i],:,channels].values
        #Y_train = (data.loc[[i for i in data.items if not 'interictal' in i and not 'test' in i],:,'time'].mean().values > 15).astype(int).reshape(-1)
        Y_train =  np.ones(X_train.shape[0])
        X_train2 = data.loc[[i for i in data.items if 'interictal' in i],:,channels].values
        Y_train2 = np.zeros(X_train2.shape[0])
        #
        test_range = range(1, 1+np.max([int(i[i.find('_')+1:]) for i in data.items if 'test' in i]) )
        fnums = range(1, 1+len([i for i in data.items if 'test' in i]))
        X_test = data.loc[['test_'+str(i) for i in fnums],:,channels].values
        #
        Y_train = np.concatenate((Y_train, Y_train2)).astype(int)
        X_train = np.append(X_train, X_train2, axis=0)
        #        
        preds = dofitSVMstd(X_train, Y_train, X_test)    
        Y_pred=preds[1]
        scores.append(sum(preds[0])/preds[0].shape[0])
        print '  ',p,'prediction: '+str(scores[-1])
        
        #sub.ix[c:c+Y_pred.shape[0]-1,'seizure'] =  Y_pred[:,0] + Y_pred[:,1] 
        sub.ix[c:c+Y_pred.shape[0]-1,'seizure'] =  Y_pred
        sub.ix[c:c+Y_pred.shape[0]-1,'early'] =  Y_pred 
        c = c + Y_pred.shape[0]
    
    sub[['clip','seizure','early']].to_csv('../submissions/'+version,index=False, float_format='%.6f')
    print 'submission file created'
    
    print 'trained score: '+str(1.*sum([scores[i]*ntrainfiles[i] for i in range(len(scores))])/sum(ntrainfiles))
    print 'projected score: '+str(1.*sum([scores[i]*ntestfiles[i] for i in range(len(scores))])/sum(ntestfiles))

make_submission()
