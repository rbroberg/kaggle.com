# Julia refactor of the following
# http://www.kaggle.com/c/afsis-soil-properties/forums/t/10176/simple-linear-regression-starting-kit
using DataFrames
# addprocs(4)

datadir="/projects/kaggle.com/afsis.soil/data/"
submitdir="/projects/kaggle.com/afsis.soil/submissions/"

# read data
dftraining = readtable(datadir*"training.csv");
dftesting = readtable(datadir*"sorted_test.csv");

# convert char factors to integers
dftraining[:Depth][dftraining[:Depth].=="Topsoil"]="1";
dftraining[:Depth][dftraining[:Depth].=="Subsoil"]="0";
dftraining[:Depth]=int(dftraining[:Depth]);

# convert char factors to integers
dftesting[:Depth][dftesting[:Depth].=="Topsoil"]="1";
dftesting[:Depth][dftesting[:Depth].=="Subsoil"]="0";
dftesting[:Depth]=int(dftesting[:Depth]);

#delete highly correlated (>0.95) features.
# concat training and testing # 1884 3594
# correlate features # 3594 3594 
# tooHigh <- findCorrelation(cor(rbind(Xtrain,Xtest)), .95)

function acorr(a)
	nr,nc = size(a);
	corrs=zeros(nc,nc);
	for r in 1:nr
		for c in r:nc
			corrs[r,c]=cor(a[:,r],a[:,c])
			corrs[c,r]=corrs[r,c]
		end
	end
	return corrs
end

# {R} findCorrelation {caret}
# This function searches through a correlation matrix 
# and returns a vector of integers corresponding to 
# columns to remove to reduce pair-wise correlations.
# The absolute values of pair-wise correlations are considered.
# If two variables have a high correlation, 
# the function looks at the mean absolute correlation of each variable and removes the variable with the largest mean absolute correlation. 

# this function returns the list of variables to keep
function reduceCorrelatedVariables(corrs,p)
	nr,nc = size(corrs);
	#vars=Set(1:nc)
	vars=Set()
	m=mean(corrs,1)
	lc = corrs.>p
	for r in 1:nr
		for c in (r+1):nc
			if lc[r,c]
				if m[r] > m[c]
					#try
						#pop!(vars,r);
						push!(vars,r);
					#end
				else
					#try
						#pop!(vars,c);
						push!(vars,c);
					#end
				end
			end
		end
	end
	return vars;
end

a=[dftraining[:,2:(end-5)],dftesting[:,2:end]]; # freature matrix with both train and test
b=acorr(a); # correlation matrix
c=reduceCorrelatedVariables(b,0.95); # set c with high correlation indices
d=Set(1:size(b,2)); # set d with all column indices
f=[pop!(d,i) for i in c]; # unsorted set d with indices with low corr
g=sort([i for i in d]); # sorted list of indices with low correlation
h=convert(Array{Int64,1},g)

# Xtrainfiltered <- Xtrain[, -tooHigh] # 1157 27
# Xtestfiltered  <-  Xtest[, -tooHigh] #  727 27
Xtrain=dftraining[:,2:(end-5)]; # (1157,3594)
Ytrain=dftraining[:,(end-4):end]; # (1157,5)
Xtest=dftesting[:,2:end]; # 
Xtrainfiltered = Xtrain[:,h]; # (1157,1711)
Xtestfiltered = Xtest[:,h]; # (727,1711)

