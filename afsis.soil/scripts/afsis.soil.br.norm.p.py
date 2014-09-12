import numpy as np
import sklearn.linear_model as linear
import sklearn.ensemble as ensemble
from sklearn import cross_validation
import pandas as pd
import sys
import sklearn.feature_selection as fs

dftest = pd.io.parsers.read_csv('../data/sorted_test.csv')
dftest.shape # 727, 3595
dftraining = pd.io.parsers.read_csv('../data/training.csv')
dftraining.shape # 1157, 3600

# extract labels (Y) and training data (X) (without sample id hashes)
dflabels = dftraining.iloc[:,3595:3600]
dftrain = dftraining.iloc[:,1:3595]
dftrain.shape # 1157, 3594

# separate hash ids from test data
dftestdata = dftest.iloc[:,1:3595]

# ------------------------------------------------------
# binary categories for Topsoil -v- Subsoil
# ------------------------------------------------------
# train
dftrain.iloc[0,3593] # Topsoil
dftrain['Depth'][dftrain['Depth']=='Topsoil']=1
dftrain['Depth'][dftrain['Depth']=='Subsoil']=0
dftrain.shape # 1157, 3594
# test
dftestdata['Depth'][dftestdata['Depth']=='Topsoil']=1
dftestdata['Depth'][dftestdata['Depth']=='Subsoil']=0
dftestdata.shape # 727, 3594

# ------------------------------------------------------
# remove CO2 bands 2654:2668 
# TODO: confirm the index are correct
# no change in LB score
# ------------------------------------------------------
# dftrain=dftrain.drop(dftrain.columns[2654:2669],1) # 1157, 3579
# dftestdata=dftestdata.drop(dftestdata.columns[2654:2669],1) # 727, 3579

#Dependent/Target variables
targets = ['Ca','P','pH','SOC','Sand']

# ------------------------------------------------------
# data exploration
# ------------------------------------------------------
'''
# find highest and lowest Ca values
dflabels.iloc[:,0].max() # 9.6458153543979712
dflabels.iloc[:,0].min() # -0.535827761069257
minmaxidx=[]
for i in range(5):
    minmaxidx.append(dflabels.iloc[:,i].idxmax())
    minmaxidx.append(dflabels.iloc[:,i].idxmin())
    
# subtract the mean
tgt=4
specdat=np.array(dftrain.iloc[:,1:3578])
specmean=specdat.mean(axis=1)
specmean.shape=(specmean.shape[0],1)
specdat2=specdat-specmean
# cut off first half of values
specdat2=specdat2[:,1800:]
# subtract the min Ca value
specmin=specdat2-specdat2[dflabels.iloc[:,tgt].idxmin(),:]
# center on first data point = 0
speczero=np.array([specmin[i,:]-specmin[i,0] for i in range(specmin.shape[0])])

lines = plt.plot(speczero[minmaxidx,:].transpose())
plt.setp(lines[tgt*2], color='b', linewidth=3.0)
plt.show()
'''

#Prepare empty result
df = pd.DataFrame({"PIDN": dftest['PIDN'], "Ca": dftest['PIDN'], "P": dftest['PIDN'], "pH": dftest['PIDN'], "SOC": dftest['PIDN'], "Sand": dftest['PIDN']})

tgt=0
for target in targets:
    # filter outliers (the logic is reversed, True is NOT OUTLIER)
    notoutlier = (dflabels[target] < dflabels[target].std() + 2*dflabels[target].std()) & (dflabels[target] > dflabels[target].std() - 2*dflabels[target].std())
    sum(notoutlier) # 1131 for 'Ca'; which is only 80%
    
    # drop "outliers"
    dfdata=dftrain[notoutlier]
    dftst=dftestdata[notoutlier]
    dfY = dflabels[notoutlier]
    
    # get spectra data, but drop the first half and non spectral fields)
    specdat = np.array(dfdata.iloc[:,1800:3578])
    # subtract the min target value
    specmin=specdat-specdat[dfY.iloc[:,tgt].idxmin(),:]
    # center on first data point = 0
    speczero=np.array([specmin[i,:]-specmin[i,0] for i in range(specmin.shape[0])])
    
    # replace spectrum data with scaled data
    X=np.concatenate((speczero,np.array(dfdata.iloc[:,3578:],dtype=float)),axis=1)
    
    clf = linear.BayesianRidge(normalize=True, verbose=True, tol=.01)
    
    #scores = np.array(cross_validation.cross_val_score(clf, dftrain[outliers], dflabels[target][outliers], cv=5))
    #print (-1 * scores), (-1  * scores.sum()/5)
    #continue
    Y=np.array(dfY[target],dtype=float)
    Y.shape=(Y.shape[0],1)
    clf.fit(X,dfY[target])
        
    # reshape test data
    testdat = np.array(dftestdata.iloc[:,1800:3578])
    testmin=testdat-specdat[dfY.iloc[:,tgt].idxmin(),:]
    testzero=np.array([testmin[i,:]-testmin[i,0] for i in range(testmin.shape[0])])
    Xtest=np.concatenate((testzero,np.array(dftestdata.iloc[:,3578:],dtype=float)),axis=1)
    
    #Get predictions
    pred = clf.predict(Xtest)
    
    # store predP using model data
    if target=="P":
        specdat = np.array(dftrain.iloc[:,1800:3578])
        specmin=specdat-specdat[dflabels.iloc[:,tgt].idxmin(),:]
        speczero=np.array([specmin[i,:]-specmin[i,0] for i in range(specmin.shape[0])])     
        Xp=np.concatenate((speczero,np.array(dftrain.iloc[:,3578:],dtype=float)),axis=1)
        predP=clf.predict(Xp)
    
    #Store results
    df[target] = pred
    
    # increment target counter
    tgt=tgt+1
    print("run: ",target)

# Wow! This was bad in an interesting way.
# I literally scored better by setting P=0
# did I miscode something?
# predict P based on Ca, pH, SOC, Sand ... and previous P prediction model
trueP=dflabels["P"]
dflabels["P"]=predP
clfP = linear.BayesianRidge(normalize=True, verbose=True, tol=.01)
clfP.fit(dflabels,trueP)
df["P"]=clfP.predict(df[["Ca","P","pH","SOC","Sand"]])

df.to_csv("../submissions/submit.br.norm.p.csv", index=False, cols=["PIDN","Ca","P","pH","SOC","Sand"])


