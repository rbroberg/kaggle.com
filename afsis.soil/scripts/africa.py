import numpy as np
import utilities as util
import sklearn.linear_model as linear
import sklearn.ensemble as ensemble
from sklearn import cross_validation
import pandas as pd
import sys
import sklearn.feature_selection as fs

headersTest = np.genfromtxt('../data/sorted_test.csv', delimiter=',', names=True)
dftest = pd.io.parsers.read_csv('../data/sorted_test.csv')
dftest.shape # 727, 3595
dftraining = pd.io.parsers.read_csv('../data/training.csv')
dftraining.shape # 1157, 3600

# extract labels (Y) and training data (X) (without sample id hashes)
dflabels = dftraining.iloc[:,3595:3600]
dftrain = dftraining.iloc[:,1:3595]
dftrain.shape # 1157, 3594

# binary categories for Topsoil -v- Subsoil
dftrain.iloc[0,3593] # Topsoil
dftrain['Depth'][dftrain['Depth']=='Topsoil']=1
dftrain['Depth'][dftrain['Depth']=='Subsoil']=0

# separate hash ids from test data
dftestdata = dftest.iloc[:,1:3595]
dftestdata['Depth'][dftestdata['Depth']=='Topsoil']=1
dftestdata['Depth'][dftestdata['Depth']=='Subsoil']=0
dftestdata.shape

#Dependent/Target variables
targets = ['Ca','P','pH','SOC','Sand']
#targets = ['P']

#Prepare empty result
df = pd.DataFrame({"PIDN": dftest['PIDN'], "Ca": dftest['PIDN'], "P": dftest['PIDN'], "pH": dftest['PIDN'], "SOC": dftest['PIDN'], "Sand": dftest['PIDN']})

for target in targets:
    # filter outliers (the logic is reversed, True is NOT OUTLIER)
    outliers = (dflabels[target] < dflabels[target].std() + 2*dflabels[target].std()) & (dflabels[target] > dflabels[target].std() - 2*dflabels[target].std())
    sum(outliers) # 1131 for 'Ca'
    
    clf = linear.BayesianRidge(normalize=True, verbose=True, tol=.01)
    
    scores = np.array(cross_validation.cross_val_score(clf, dftrain[outliers], dflabels[target][outliers], cv=5))
    #print (-1 * scores), (-1  * scores.sum()/5)
    #continue
    clf.fit(dftrain[outliers], dflabels[target][outliers])
    
    #Get predictions
    pred = clf.predict(dftestdata)
    
    #Store results
    df[target] = pred

df.to_csv("../submissions/submit.bayesianridge.params.csv", index=False, cols=["PIDN","Ca","P","pH","SOC","Sand"])

