# it's been noted that the data set comes from at least two different sources
# the benchmark uses the mean of the entire data set.
# this uses the mean from each of the two inferred sources

# benchmark: 3.96244
# score: 3.96572

# caveat: I am just learning Julia and this is an exercise.

# rbroberg: 20140822

using DataFrames
using ImageView

datadir="/projects/kaggle.com/facial.key/data/"
submitdir="/projects/kaggle.com/facial.key/submissions/"
libdir="/projects/kaggle.com/facial.key/scripts/"
dldir="/projects/kaggle.com/facial.key/downloads/"

# data load
dftrain = readtable(datadir*"training.csv");
dftest = readtable(datadir*"test.csv");

# modified this table by adding 0.1 values in Location column to read
dfsubmit = readtable(dldir*"IdLookupTable.csv");

# get col names
dfnames=DataFrame([1:size(dftrain,2)]')
names!(dfnames, names(dftrain))

# data is composed of at least two distinct data sets A and B
# http://www.kaggle.com/c/facial-keypoints-detection/forums/t/5059/how-to-crack-this-dataset
A1 = 2284
B1 = A1+1
A2 = 591
B2 = A2+1

# examine patches 3*std for left and right eye center looking for darkest patch
left_eye_center_x_mean=int(round(mean(dropna(dftrain[:,:left_eye_center_x]))));
left_eye_center_y_mean=int(round(mean(dropna(dftrain[:,:left_eye_center_y]))));
left_eye_center_x_std=int(std(dropna(dftrain[:,:left_eye_center_x]))) # 4;
left_eye_center_y_std=int(std(dropna(dftrain[:,:left_eye_center_y]))) # 4;
x1=left_eye_center_x_mean-2*left_eye_center_x_std;
x2=left_eye_center_x_mean+2*left_eye_center_x_std;
y1=left_eye_center_y_mean-2*left_eye_center_y_std;
y2=left_eye_center_y_mean+2*left_eye_center_y_std;
xrng=4*left_eye_center_x_std+1;
yrng=4*left_eye_center_y_std+1;
function find_left_eye_center(img)
	img=reshape(img,(96,96));
	#view(img/256); # rotated
	img=img';
	#view(img/256);# hey! img#1
	#view(img[y1:y2,x1:x2]/256); # hey! its an eye img#1
	mn=findin(img[y1:y2,x1:x2],minimum(img[y1:y2,x1:x2]));
	feature_y=int(mn%yrng[1])[1];
	feature_x=int(mn/xrng[1])[1];
	#img[y1:y2,x1:x2][feature_y,feature_x]
	(feature_x,feature_y)
end

right_eye_center_x_mean=int(round(mean(dropna(dftrain[:,:right_eye_center_x]))));
right_eye_center_y_mean=int(round(mean(dropna(dftrain[:,:right_eye_center_y]))));
right_eye_center_x_std=int(std(dropna(dftrain[:,:right_eye_center_x]))) # 4;
right_eye_center_y_std=int(std(dropna(dftrain[:,:right_eye_center_y]))) # 4;
x1=right_eye_center_x_mean-2*right_eye_center_x_std;
x2=right_eye_center_x_mean+2*right_eye_center_x_std;
y1=right_eye_center_y_mean-2*right_eye_center_y_std;
y2=right_eye_center_y_mean+2*right_eye_center_y_std;
xrng=4*right_eye_center_x_std+1;
yrng=4*right_eye_center_y_std+1;
function find_right_eye_center(img)
	img=reshape(img,(96,96));
	#view(img/256); # rotated
	img=img';
	#view(img/256);# hey! img#1
	#view(img[y1:y2,x1:x2]/256); # hey! its an eye img#1
	mn=findin(img[y1:y2,x1:x2],minimum(img[y1:y2,x1:x2]));
	feature_y=int(mn%yrng[1])[1];
	feature_x=int(mn/xrng[1])[1];
	#img[y1:y2,x1:x2][feature_y,feature_x]
	(feature_x,feature_y)
end

# create image array
a=[int(split(dftrain[i,31])) for i in 1:7049];
imgs = Array(Int16,length(a),length(a[1]));
[imgs[i,:] = hcat(a[i]) for i in 1:length(a)];

# find the mean for each column data set
featuremean=[mean(dropna(dftrain[:,i])) for i in 1:30]';

# set results based on means
[dfsubmit[i,:Location]=featuremean[dfnames[dfsubmit[i,:FeatureName]][1]] for i in 1:size(dfsubmit,1)];

# run patch functions for left and right eye center
xeyeidx=dfsubmit[dfsubmit[:,:FeatureName].=="left_eye_center_x",:RowId];
[dfsubmit[i,:left_eye_center_x]=find_right_eye_center(imgs[dfsubmit[i,:ImageId]]) for i in xeyeidx]

xeyeidx=dfsubmit[:,:FeatureName].=="right_eye_center_x";
yeyeidx=dfsubmit[:,:FeatureName].=="right_eye_center_y";

# write submission
writetable(submitdir*"submit.mean.a.b.csv",
	dfsubmit[:,[1,4]],separator=',',header=true);
	
