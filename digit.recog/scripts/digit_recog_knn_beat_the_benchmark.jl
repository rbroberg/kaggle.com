# https://www.kaggle.com/c/street-view-getting-started-with-julia/details/knn-tutorial
#Pkg.add("DataFrames")

# suggestion to try: # replace euclidean distance with tangent distance
# https://github.com/pulkitjain1821/Artificial-Intelligence-CS365/blob/master/hw1/HW1/tangent_d.m

addprocs(1) # just in case someone isn't reading
# addprocs(4)

# this 0.96943
# R KNN benchmark 0.96557
# R RF benchmark 0.96829

using DataFrames

datadir="/projects/kaggle.com/digit.recog/data/"
submitdir="/projects/kaggle.com/digit.recog/submissions/"
libdir="/projects/kaggle.com/digit.recog/scripts/"

# data load
dftrain = readtable(datadir*"train.csv");
dftest = readtable(datadir*"test.csv");

# data selection
yTrain = array(dftrain[:,1]);
xTrain = array(dftrain[:,2:end]);
xTest = array(dftest[:,1:end]);

xTrain = xTrain';
xTest = xTest';

@everywhere function euclidean_distance(a, b)
 distance = 0.0 
 for index in 1:size(a, 1) 
  distance += (a[index]-b[index]) * (a[index]-b[index])
 end
 return distance
end

@everywhere function get_k_nearest_neighbors(xTrain, imageI, k)
 nRows, nCols = size(xTrain) 
 imageJ = Array(Float32, nRows)
 distances = Array(Float32, nCols) 
 for j in 1:nCols
  for index in 1:nRows
   imageJ[index] = xTrain[index, j]
  end
  distances[j] = euclidean_distance(imageI, imageJ)
 end
 sortedNeighbors = sortperm(distances)
 kNearestNeighbors = sortedNeighbors[1:k]
 return kNearestNeighbors
end 

@everywhere function assign_label(xTrain, yTrain, k, imageI)
 kNearestNeighbors = get_k_nearest_neighbors(xTrain, imageI, k) 
 counts = Dict{Int, Int}() 
 highestCount = 0
 mostPopularLabel = 0
 for n in kNearestNeighbors
  labelOfN = yTrain[n]
  if !haskey(counts, labelOfN)
   counts[labelOfN] = 0
  end
  counts[labelOfN] += 1 #add one to the count
  if counts[labelOfN] > highestCount
   highestCount = counts[labelOfN]
   mostPopularLabel = labelOfN
  end 
 end
 return mostPopularLabel
end

k = 3 # The CV accuracy shows this value to be the best.
yPredictions = @parallel (vcat) for i in 1:size(xTest, 2)
 nRows = size(xTrain, 1)
 imageI = Array(Float32, nRows)
 for index in 1:nRows
  imageI[index] = xTest[index, i]
 end
 assign_label(xTrain, yTrain, k, imageI)
end

#Convert integer predictions to character
dfresults=DataFrame()
dfresults[:ImageID]=1:size(yPredictions,1)
dfresults[:Label]=yPredictions

#Save predictions
writetable(submitdir*"juliaKNNSubmission.csv", dfresults, separator=',', header=true)
