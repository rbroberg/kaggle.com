import scipy.io
import scipy.signal
import os
import pandas as pd
import numpy as np
import pywt

datadir = "/data/www.kaggle.com/c/seizure-prediction/download/"

#         case       pre  inter  test
cases=[ ['Dog_1',     24,  480,  502],
        ['Dog_2',     42,  500, 1000],
        ['Dog_3',     72, 1440,  907],
        ['Dog_4',     97,  804,  990],
        ['Patient_1', 18,   50,  195],
        ['Patient_2', 18,   42,  150]]

def downsample(casedir, testdat, downsamplefreq):
    dir = datadir + casedir + '/'
    #print(dir)
    dict = {}
    
    #load files in numerical order
    files = os.listdir(dir)
    files=filter( lambda f: f.endswith('mat'), os.listdir(dir))
    if testdat:
        files=filter( lambda f: f.find('test') > -1, files)
    else:
        files=filter( lambda f: f.find('test') < 0, files)
    
    files2 =[]  
    for i in range(len(files)):
        qp = files[i].rfind('_') +1
        files2.append( files[i][0:qp] + (8-len(files[i][files[i].rfind('_')+1:]) )*'0' + files[i][qp:] )
    
    #print len(files), len(files2)
    t={}
    for key,value in zip(files2,files):
        t[key]=value
    
    files2 = t.keys()
    files2.sort()
    f = [t[i] for i in files2]
    
    dfiles={}
    for i in f:
        seg = i[i.rfind('_')+1 : i.find('.mat')]
        segtype = i[i[0:i.find('_segment')].rfind('_')+1: i.find('_segment')]
        print i, dir+i
        mat = scipy.io.loadmat(dir+i)
        matkey=segtype+"_segment_"+str(int(seg))
        data=mat[matkey]['data'][0,0]
        try:
            seq = int(mat[matkey]['sequence'][0,0])
        except:
            seq=-1
        freq = int(mat[matkey]['sampling_frequency'][0,0])
        ds = int(round(1.*freq/downsamplefreq))
        d2 = scipy.signal.decimate(data,ds)
        d2=d2.astype('int16')
        df1=pd.DataFrame([i,casedir,segtype,seg,seq,d2]).transpose()
        dict.update({i: df1})
    
    return pd.Panel(dict)
 
if __name__ == "__main__":
    for case in cases:
        downsamplefreq=200.
        mydf=downsample(case[0],False,downsamplefreq)
        mydf.to_pickle(datadir+case[0]+".train.ds."+str(int(round(downsamplefreq)))+".pkl")
        mydf=downsample(case[0],True,downsamplefreq)
        mydf.to_pickle(datadir+case[0]+".test.ds."+str(int(round(downsamplefreq)))+".pkl")

 
