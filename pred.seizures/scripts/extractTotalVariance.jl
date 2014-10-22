using HDF5

datadir="/data/www.kaggle.com/c/seizure-prediction/download/"
case="Dog_1"
segfile="Dog_1_interictal_segment_0001.hdf5"

fn=datadir*case*"/"*segfile

data=h5open(fn,"r")
names(data)

nc=read(data["nchannels"])["values"][1];
#freq=read(data["freq"])["values"][1];
dat=read(data["data"])["block0_values"];
size(dat)
#=
'''
nsecs=int(size(dat)[1]/(200)) # there may be remaining trailing points ...
ex=size(dat)[1]
idx=[(ex-(i+1)*200+1,ex-i*200) for i in 0:(nsecs-1)] # from last point towards first point
ex=length(idx)
v=[var(dat[idx[ex-i][1]:idx[ex-i][2],j]) for i = 0:(ex-1), j = 1:nc] # from first offset sec to last second
'''
+=
v=[var(dat[:,j]) for j = 1:nc]

close(data)
