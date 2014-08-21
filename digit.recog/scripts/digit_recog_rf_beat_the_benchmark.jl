# out of the box Random Forest implementation
# the only 'tuning' is selecting the standard sqrt(features) for branching

# rf_benchmark in R is .96557

# caveat: I am just learning Julia and this is an exercise.

# rbroberg: 20140822

using DecisionTree
using DataFrames

datadir="/projects/kaggle.com/digit.recog/data/"
submitdir="/projects/kaggle.com/digit.recog/submissions/"
libdir="/projects/kaggle.com/digit.recog/scripts/"

# data load
dftrain = readtable(datadir*"train.csv");
dftest = readtable(datadir*"test.csv");

# data selection
labels = array(dftrain[:,1]);
features = array(dftrain[:,2:end]);

# model explortation
# nfeatures = 2
# 10 trees: Mean Accuracy: 0.74702
# 100 trees: Mean Accuracy: 0.85983
# 1000 trees: Mean Accuracy: 0.87345 (~45 min per 'fold' on my machine)

# nfeatures = 28
# 10 trees: Mean Accuracy: 0.92933, Submission Score: 0.93443
# 100 trees: Mean Accuracy: 0. , Submission Score:
# 1000 trees: Mean Accuracy: 0. , Submission Score:
nfeatures=int(size(features)[2]^0.5); # 28
ntrees=1000;
accuracy=nfoldCV_forest(labels, features, nfeatures, ntrees, 3);
println ("3 fold accuracy: $(mean(accuracy))")

# model
# n=28 splits, 100 trees
model = build_forest(labels, features, nfeatures, ntrees);

# prediction
features = array(dftest[:,1:end]);
preds=apply_forest(model, features);

# submission
sarr=zeros(length(preds),2);
sarr=convert(Array{Int16,2},sarr);
sarr[:,1]=1:length(preds);
sarr[:,2]=preds;
df=convert(DataFrame,sarr);
names!(df,[symbol("ImageID"),symbol("Label")]);
writetable(submitdir*"submit.rf."*string(nfeatures)*"."*string(ntrees)*".csv",
	df[:,[1,2]],separator=',',header=true);

# maybe a neural network?
# http://blog.yhathq.com/posts/julia-neural-networks.html