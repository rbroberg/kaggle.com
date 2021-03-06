'''
           DO WHAT THE FUCK YOU WANT TO PUBLIC LICENSE
                   Version 2, December 2004

Copyright (C) 2004 Sam Hocevar <sam@hocevar.net>

Everyone is permitted to copy and distribute verbatim or modified
copies of this license document, and changing it is allowed as long
as the name is changed.

           DO WHAT THE FUCK YOU WANT TO PUBLIC LICENSE
  TERMS AND CONDITIONS FOR COPYING, DISTRIBUTION AND MODIFICATION

 0. You just DO WHAT THE FUCK YOU WANT TO.
'''


from datetime import datetime
from csv import DictReader
from math import exp, log, sqrt
from sys import stdout

datadir="/data/www.kaggle.com/c/criteo-display-ad-challenge/download/"
# parameters #################################################################

train = datadir+'train.csv'  # path to training file
test = datadir+'test.csv'  # path to testing file

dp = 32 
D = 2 ** dp   # number of weights use for learning
alpha = .11    # learning rate for sgd optimization

# create time index
days = [0,6564555,12684055,19359873,26369482,33156564,39832383,45840618]
d=[[str(int(24*(1.-1.*(i-days[j-1])/(days[j]-days[j-1])))) for i in range(days[j-1],days[j])] for j in range(1,len(days))]
hour=[ddd for dd in d for ddd in dd]

# function definitions #######################################################

# A. Bounded logloss
# INPUT:
#     p: our prediction
#     y: real answer
# OUTPUT
#     logarithmic loss of p given y
def logloss(p, y):
    p = max(min(p, 1. - 10e-12), 10e-12)
    return -log(p) if y == 1. else -log(1. - p)


# B. Apply hash trick of the original csv row
# for simplicity, we treat both integer and categorical features as categorical
# INPUT:
#     csv_row: a csv dictionary, ex: {'Lable': '1', 'I1': '357', 'I2': '', ...}
#     D: the max index that we can hash to
# OUTPUT:
#     x: a list of indices that its value is 1
def get_x(csv_row, D):
    x = [0]  # 0 is the index of the bias term
    for key, value in csv_row.items():
        index = int(value + key[1:], 16) % D  # weakest hash ever ;)
        x.append(index)
    # conjugates
    items=csv_row.items()
    l=len(items)
    for i in range(l-1):
        for j in range(i+1,l):
            index1 = int(items[i][1]+items[i][0][1:], 16)  # weakest hash ever ;)
            index2 = int(items[j][1]+items[j][0][1:], 16)  # weakest hash ever ;)
            x.append((index1+index2)% D)
    return x  # x contains indices of features that have a value of 1


# C. Get probability estimation on x
# INPUT:
#     x: features
#     w: weights
# OUTPUT:
#     probability of p(y = 1 | x; w)
def get_p(x, w):
    wTx = 0.
    for i in x:  # do wTx
        wTx += w[i] * 1.  # w[i] * x[i], but if i in x we got x[i] = 1.
    return 1. / (1. + exp(-max(min(wTx, 20.), -20.)))  # bounded sigmoid


# D. Update given model
# INPUT:
#     w: weights
#     n: a counter that counts the number of times we encounter a feature
#        this is used for adaptive learning rate
#     x: feature
#     p: prediction of our model
#     y: answer
# OUTPUT:
#     w: updated model
#     n: updated count
def update_w(w, n, x, p, y):
    for i in x:
        # alpha / (sqrt(n) + 1) is the adaptive learning rate heuristic
        # (p - y) * x[i] is the current gradient
        # note that in our case, if i in x then x[i] = 1
        w[i] -= (p - y) * alpha / (sqrt(n[i]) + 1.)
        n[i] += 1.
    
    return w, n


# training and testing #######################################################

# initialize our model
w = [0.] * D  # weights
n = [0.] * D  # number of times we've encountered a feature

# start training a logistic regression model using on pass sgd
loss = 0.
epoch=2
for e in range(epoch):
    for t, row in enumerate(DictReader(open(train))):
        y = 1. if row['Label'] == '1' else 0.
        
        del row['Label']  # can't let the model peek the answer
        del row['Id']  # we don't need the Id
        
        # add time
        row['h004'] = hour[t]
    
        # main training procedure
        # step 1, get the hashed features
        x = get_x(row, D)
        
        # step 2, get prediction
        p = get_p(x, w)
        
        # for progress validation, useless for learning our model
        loss += logloss(p, y)
        #if t % 1000000 == 0 and t > 1:
        if t % 1000000 == 0 and t > 1:
            it=int(e)+1
            itt=it*t
            print('%s D: %d alpha: %0.2f epoch: %d encountered: %d\tcurrent logloss: %f' % (
                datetime.now(), dp, alpha, it, t, loss/itt))
            stdout.flush()
        
        # step 3, update model with answer
        w, n = update_w(w, n, x, p, y)

# testing (build kaggle's submission file)
d=6042136
testhour=[str(int(24*(1.-1.*(i/d)))) for i in range(d)]
fn='.'.join(['submit.conj.hour','D'+str(dp),'alpha'+str(alpha),'epoch'+str(epoch),'csv'])
with open('../submissions/'+fn, 'w') as submission:
    submission.write('Id,Predicted\n')
    for t, row in enumerate(DictReader(open(test))):
        Id = row['Id']
        del row['Id']
	row['h004']=testhour[t]
        x = get_x(row, D)
        p = get_p(x, w)
        submission.write('%s,%f\n' % (Id, p))

