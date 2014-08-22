# scored 4.14407

using DataFrames;

srand(3354);

datadir="/projects/kaggle.com/facial.key/data/";
submitdir="/projects/kaggle.com/facial.key/submissions/";
libdir="/projects/kaggle.com/facial.key/scripts/";
dldir="/projects/kaggle.com/facial.key/downloads/";

# data load
dftrain = readtable(datadir*"training.csv");
dftest = readtable(datadir*"test.csv");

# modified this table by adding 0.1 values in Location column to read
dfsubmit = readtable(dldir*"IdLookupTable.csv");

# get col names
dfnames=DataFrame([1:size(dftrain,2)]');
names!(dfnames, names(dftrain));

# get features
# col 1-30 is labeled data
# col 31 is imagedffeatures = dftrain[:,1:30];
dfmean=dffeatures[end,:];
dfmean=[mean(dropna(dftrain[:,i])) for i in 1:30]';
push!(dffeatures, dfmean);

# get image array
a=[int(split(dftrain[i,31])) for i in 1:7049];
imgs = Array(Int16,length(a),length(a[1]));
[imgs[i,:] = hcat(a[i]) for i in 1:length(a)];
m1=int(mean(imgs,1));
    
# select random images
# TO-DO: what if a number is duped?
nrandom=10
ridx = Array(Int16,nrandom); # empty random index 
rand!(1:7049,ridx); # get 10 random numbers in range
eximgs = Array(Int16,nrandom+1,9216); # example image array
eximgs[1:nrandom,:]=[imgs[ridx,:]]; # select 10 random
push!(ridx,7050); # add mean to index
eximgs[nrandom+1,:]=m1; # add mean image

# find least distance from this image to each of 11
# TO-DO: normalize images by edge detection? need to
function getImageIndex(img) 
	d=[dot(vec(eximgs[i,:]-img),vec(eximgs[i,:]-img))^0.5 for i in 1:(nrandom+1)];
	mn=minimum(d);
	findin(d,mn)[1];
end;

# imgmatches=[getImageIndex(imgs[i,:]) for i in 1:7049]; # image neighbors
# [sum(imgmatches.==i) for i in 1:(nrandom+1)] 
# 21% of images most closely match mean with n=10
# 23% of images most closely match mean with n=83

# run the test data
a=[int(split(dftest[i,2])) for i in 1:size(dftest,1)];
imgs = Array(Int16,length(a),length(a[1]));
[imgs[i,:] = hcat(a[i]) for i in 1:length(a)];
imgmatches=[getImageIndex(imgs[i,:]) for i in 1:size(dftest,1)]; # image neighbors
[sum(imgmatches.==i) for i in 1:(nrandom+1)] # varies on how many most closely match mean

testidx=[ridx[imgmatches[i]] for i in 1:size(imgmatches,1)];
results=dffeatures[int(testidx),:]; # these have NAs. Replace NAs with mean?
[[if isequal(results[i,j],NA) results[i,j]=dfmean[1,j] end for j in 1:30] for i in 1:size(results,1)];

# assign results to submission 
[dfsubmit[i,:Location]=results[dfsubmit[i,:ImageId],dfnames[dfsubmit[i,:FeatureName]][1]] for i in 1:size(dfsubmit,1)];

# hmmm ... max of 96?, see submission 1881-1886 ...
#[if dfsubmit[i,:Location]>96 dfsubmit[i,:Location]=96.0 end for i in 1:size(dfsubmit,1)];

# write submission
writetable(submitdir*"submit.templates."*string(nrandom)*".csv",
	dfsubmit[:,[1,4]],separator=',',header=true)