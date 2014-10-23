import numpy as np
import pandas as pd
import math
import random
import pywt
import glob
import scipy.io

# all 25922
ntrainfiles=[596,1320,5240,3047, 174,3141,1041,210, 2745,2997,3521,1890]
ntestfiles=[3181,2997,4450,3013, 2050,3894,1281,543, 2986,2997,3601,1922]
datadir="../data/"

cases=[
	"Dog_1",
	"Dog_2",
	"Dog_3",
	"Dog_4",
	"Patient_1",
	"Patient_2",
	"Patient_3",
	"Patient_4",
	"Patient_5",
	"Patient_6",
	"Patient_7",
	"Patient_8"]

#for i in range(len(cases)):
i=5
c=cases[i]
eegno=[]
files=glob.glob(datadir+c+"/*_interictal_*mat")
for f in files:
    mat = scipy.io.loadmat(f)
    dat=mat['data'].astype(np.int64)
    for channel in range(dat.shape[0]):
        eegno.append(list(dat[channel,:]))

eegsz=[]
files=glob.glob(datadir+c+"/*_ictal_*mat")
for f in files:
    mat = scipy.io.loadmat(f)
    dat=mat['data'].astype(np.int64)
    for channel in range(dat.shape[0]):
        eegsz.append(list(dat[channel,:]))


#1. Join the data points of all EEG segments on the axis of
#amplitude.

joined_eeg=[x for x in e for e in eegno]


K=6.
#for CASE in range(len(cases)):
#2. Decompose the joined data into subbands according to
#given decomposition level.

cA1, cD1 = pywt.dwt(joined_eeg, 'db2')
cA2, cD2 = pywt.dwt(cA1, 'db2')

#3. In order to determine the boundaries of intervals, split
#each one of the obtained subbands into K intervals so
#that each interval has the same number of points.

Npts=len(cD2) # 102427 for A/Z
mx=max(cD2)+1 #  163.62055276601635 +1
mn=min(cD2)-1 # -212.52558883261165 -1
binpts=int(round(Npts/K))

# find bins
breakidx=[mn+k*binpts for k in range(int(K))]
cD2.sort()
brkidx=[k*int(round(binpts)) for k in range(int(K+1))]
brkidx[-1]=Npts-1
breaks=[cD2[x] for x in brkidx]

#4. For each EEG segment, compute the number of data
#points in each interval according to the determined
#boundaries.
            
#5. Compute the probability density of each EEG segment
#according to the computed number of data points in
#Step 4.

probseg=[]
probs=[]
for j in range(len(eegno)):
    e=eegno[j]
    eA1, eD1 = pywt.dwt(eegno[j], 'db2')
    eA2, eD2 = pywt.dwt(eA1, 'db2')
    ps=[]
    for k in range(int(K)):
        ps.append(1.*sum((eD2>breaks[k])&(eD2<breaks[k+1]))/len(eD2))
    
    probseg.append(ps)

probs.append(probseg)
probseg=[]
for j in range(len(eegsz)):
    e=eegsz[j]
    eA1, eD1 = pywt.dwt(eegsz[j], 'db2')
    eA2, eD2 = pywt.dwt(eA1, 'db2')
    ps=[]
    for k in range(int(K)):
        ps.append(1.*sum((eD2>breaks[k])&(eD2<breaks[k+1]))/len(eD2))
    
    probseg.append(ps)

probs.append(probseg)

1.*sum([sum(probs[0][i])<0.9 for i in range(len(probs[0]))])/len(probs[0])
1.*sum([sum(probs[1][i])>0.9 for i in range(len(probs[1]))])/len(probs[1])

lenno=len(eegno)/16
lensz=len(eegsz)/16
1.*sum([sum([sum(probs[0][16*j+i])>0.5 for i in range(16)])>0 for j in range(lenno)])/lenno
1.*sum([sum([sum(probs[0][16*j+i])>0.5 for i in range(16)])==0 for j in range(lenno)])/lenno
1.*sum([sum([sum(probs[1][16*j+i])<0.5 for i in range(16)])>0 for j in range(lensz)])/lensz
1.*sum([sum([sum(probs[1][16*j+i])<0.5 for i in range(16)])==0 for j in range(lensz)])/lensz

# Dog_3: 0.5
# Patient_2: 
