import numpy as np
import utilities as util
import sklearn.linear_model as linear
import sklearn.ensemble as ensemble
from sklearn import cross_validation
import pandas as pd
import sys
import sklearn.feature_selection as fs

headersTest = np.genfromtxt('../data/sorted_test.csv', delimiter=',', names=True)

#Columns to be picked from training file
pickTrain = ['BSAN','BSAS','BSAV','CTI','ELEV','EVI','LSTD','LSTN','REF1','REF2','REF3','REF7','RELI','TMAP','TMFI','Depth','Ca','P','pH','SOC','Sand']

data = np.genfromtxt('../data/training.csv', names=True, delimiter=',') #, usecols=(pickTrain))

#Column to be picked from test file
pickTest = ['PIDN', 'BSAN','BSAS','BSAV','CTI','ELEV','EVI','LSTD','LSTN','REF1','REF2','REF3','REF7','RELI','TMAP','TMFI']
test = np.genfromtxt('../data/sorted_test.csv', names=True, delimiter=',')#, usecols=(pickTest))

ids = np.genfromtxt('../data/sorted_test.csv', dtype=str, skip_header=1, delimiter=',', usecols=0)

#Features to train model on
#featuresList = ['BSAN','BSAS','BSAV','CTI','ELEV','EVI','LSTD','LSTN','REF1','REF2','REF3','REF7','RELI','TMAP','TMFI']
featuresList = np.array(headersTest.dtype.names)
featuresList = featuresList[2:-5]

#Keep a copy of train file for later use
data1 = np.copy(data)

#Dependent/Target variables
targets = ['Ca','P','pH','SOC','Sand']
#targets = ['P']

#Prepare empty result
df = pd.DataFrame({"PIDN": ids, "Ca": test['PIDN'], "P": test['PIDN'], "pH": test['PIDN'], "SOC": test['PIDN'], "Sand": test['PIDN']})

for target in targets:
    #Prepare data for training
    data = np.copy(data1)
    delList = np.array([])
    for i in range(len(data)):
        if data[target][i] > (data[target].mean() + 2*data[target].std()) or data[target][i] < (data[target].mean() - 2*data[target].std()):
            delList = np.append(delList, i)
    data = np.delete(data, delList, 0)
    data, testa, features, fillVal = util.prepDataTrain(data, target, featuresList, False, 10, False, True, 'mean', False, 'set')
    #sel = fs.SelectKBest(fs.f_regression, k=2000)
    print data.shape
    #data = np.array(sel.fit_transform(data[features].tolist(), data1[target]))
    #print data.shape
    #Use/tune your predictor
    clf = linear.BayesianRidge(normalize=True, verbose=True, tol=.01)
    
    #scores = cross_validation.cross_val_score(clf, data[features].tolist(), data[target], cv=5, scoring='mean_squared_error')
    scores = np.array(cross_validation.cross_val_score(clf, data[features].tolist(), data[target], cv=5, scoring='mean_squared_error'))
    print (-1 * scores), (-1  * scores.sum()/5)
    continue
    clf.fit(data[features].tolist(), data[target])

    #Prepare test data
    test = util.prepDataTest(test, featuresList, True, fillVal, False, 'set')
    
    #Get predictions
    pred = clf.predict(test[features].tolist())
    
    #Store results
    df[target] = pred

df.to_csv("../submissions.bayesianridge.params.csv", index=False, cols=["PIDN","Ca","P","pH","SOC","Sand"])



