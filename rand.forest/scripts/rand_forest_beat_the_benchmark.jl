# predict cover type based on park (wilderness area) and soil type
# prediction is simply the cover type with highest frequency for those features
# score is 0.52050

# caveat: I am just learning Julia and this is an exercise.

# rbroberg: 20140820

using DataFrames

datadir="/projects/kaggle.com/rand.forest/data/"
submitdir="/projects/kaggle.com/rand.forest/submissions/"

df = readtable(datadir*"train.csv");

# look at data
size(df) # (15120,56)
df[1,:]

# collapse park identifier into column 12
df[df[:,12].==1,12]=1;
df[df[:,13].==1,12]=2;
df[df[:,14].==1,12]=3;
df[df[:,15].==1,12]=4;

# count the number of park instances
sum(df[:,12].==1) # 3597
sum(df[:,12].==2) # 499
sum(df[:,12].==3) # 6349
sum(df[:,12].==4) # 4675

# collapse park identifier into column 16
[df[df[:,i+15].==1,16]=i for i in 1:40];

# create a new, reduced df
df2=df[:,[1:12,16,56]];
size(df2)
df2[10000,:]

# don't need original
df=None

# count all trees by park and soil
# can I get an ascii table here?
pstree=zeros((4,40,7));
pstree=convert(Array{Int16,3},pstree); # bet there is a way to specify this in zeros

[[[pstree[i,j,k]=sum((df2[:,12].==i)
		&(df2[:,13].==j)
		&(df2[:,14].==k)) 
	for k in 1:7] for j in 1:40] for i in 1:4];
pstree[4,10,:] # 0, 13, 617, 170, 0, 682, 0]

# find max tree prob for each park and soil type
# takes first cover type by order in case of tie
findin(pstree[4,10,:],maximum(pstree[4,10,:]))
psdict=zeros((4,40));
psdict=convert(Array{Int16,2},psdict); # bet there is a way to specify this in zeros
[[psdict[i,j]=findin(pstree[i,j,:],maximum(pstree[i,j,:]))[1] 
	for j in 1:40] for i in 1:4];

dftest = readtable(datadir*"test.csv");

# reduce test data same as the train data
# col 56 'cover type' does not exist in test data
[dftest[dftest[:,i+11].==1,12]=i for i in 1:4];
[dftest[dftest[:,i+15].==1,16]=i for i in 1:40];
dftest2=dftest[:,[1,2,12,16]];

# don't need original
dftest=None

# store predictions in column two and rename
nrow=size(dftest2)[1] # 565892
[dftest2[i,2]=psdict[dftest2[i,3],dftest2[i,4]] for i in 1:nrow];
rename!(dftest2, "Elevation", "Cover_Type")

# find tree distribution
[sum(dftest2[:,2].==k) for k in 1:7]

# write submission file
writetable(submitdir*"submit.prob.park.soil.csv",
	dftest2[:,[1,2]],separator=',',header=true);
