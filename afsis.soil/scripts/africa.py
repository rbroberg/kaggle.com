import numpy as np
import utilities as util
import sklearn.linear_model as linear
import sklearn.ensemble as ensemble
#from sklearn.linear_model import BayesianRidge
from sklearn import cross_validation
import pandas as pd
import sys

#Columns to be picked from training file
pickTrain = ['BSAN','BSAS','BSAV','CTI','ELEV','EVI','LSTD','LSTN','REF1','REF2','REF3','REF7','RELI','TMAP','TMFI','Depth','Ca','P','pH','SOC','Sand']
data = np.genfromtxt("../data/training.csv", names=True, delimiter=',', usecols=(pickTrain))

#Column to be picked from test file
pickTest = ['PIDN', 'BSAN','BSAS','BSAV','CTI','ELEV','EVI','LSTD','LSTN','REF1','REF2','REF3','REF7','RELI','TMAP','TMFI']
test = np.genfromtxt("../data/sorted_test.csv", names=True, delimiter=',', usecols=(pickTest))

ids = np.genfromtxt("../data/sorted_test.csv", dtype=str, skip_header=1, delimiter=',', usecols=0)

#Features to train model on
featuresList = ['BSAN','BSAS','BSAV','CTI','ELEV','EVI','LSTD','LSTN','REF1','REF2','REF3','REF7','RELI','TMAP','TMFI']

#Keep a copy of train file for later use
data1 = np.copy(data)

#Dependent/Target variables
targets = ['Ca','P','pH','SOC','Sand']

#Prepare empty result
df = pd.DataFrame({"PIDN": ids, "Ca": test['PIDN'], "P": test['PIDN'], "pH": test['PIDN'], "SOC": test['PIDN'], "Sand": test['PIDN']})

for target in targets:
    #Prepare data for training
    data, testa, features, fillVal = util.prepDataTrain(data1, target, featuresList, False, 10, False, True, 'mean', False,'set')
    
    print 'Data prepped'
    
    #Use/tune your predictor
    clf = ensemble.GradientBoostingRegressor(n_estimators=20)
    #clf = BayesianRidge(compute_score=True)
    clf.fit(data[features].tolist(), data[target])
    
    #Prepare test data
    test = util.prepDataTest(test, featuresList, True, fillVal, False, 'set')
    
    #Get predictions
    pred = clf.predict(test[features].tolist())
    
    #Store results
    df[target] = pred

df.to_csv("../submissions/submit.africa.GradientBoostingRegressor.csv", index=False, cols=["PIDN","Ca","P","pH","SOC","Sand"])


