# https://gist.github.com/a-f-s-i-s/3913d670cb882341a332
#BART (Bayesian Additive Regression Tree) Models

# devtools:::install_github('BayesTree', 'cran')
#library(BayesTree)
library(bartMachine)
library(caret)
 
trainingdata <- read.csv("../data/training.csv")
testdata <- read.csv("../data/sorted_test.csv")
soil_properties <- c("Ca", "P", "pH", "SOC", "Sand")

trainingdata[,3595]=as.integer(trainingdata[,3595] == 'Topsoil')
testdata[,3595]=as.integer(testdata[,3595] == 'Topsoil')

# CO2_bands <- 2656:2670
names(trainingdata)[2656:2670]
 
# take the first derivatives to smoothe out the measurement noise
# training data
MIR_measurements <- trainingdata[, 2:2655]
MIR_DER <- MIR_measurements- cbind(NA, MIR_measurements)[, -(dim(MIR_measurements)[2]+1)]
X_train <- cbind(trainingdata[, 3580:3595], MIR_DER[,-1])
MIR_measurements <- trainingdata[, 2671:3579]
MIR_DER <- MIR_measurements- cbind(NA, MIR_measurements)[, -(dim(MIR_measurements)[2]+1)]
X_train <- cbind(X_train, MIR_DER[, -1])
 
# testing data
MIR_measurements <- testdata[, 2:2655]
MIR_DER <- MIR_measurements- cbind(NA, MIR_measurements)[, -(dim(MIR_measurements)[2]+1)]
X_test <- cbind(testdata[, 3580:3595], MIR_DER[,-1])
MIR_measurements <- testdata[, 2671:3579]
MIR_DER <- MIR_measurements- cbind(NA, MIR_measurements)[, -(dim(MIR_measurements)[2]+1)]
X_test <- cbind(X_test, MIR_DER[, -1])
 
tooHigh <- findCorrelation(cor(rbind(X_train,X_test)), .95)

Xtrainfiltered <- X_train[, -tooHigh]
Xtestfiltered  <-  X_test[, -tooHigh]

# BART predictions
# without Cross-Validation calibration for hyperparameters
 
predictions <- rep(NA, dim(X_test)[1])
for(soil_property in soil_properties){
    bart_model <- bartMachine(Xtrainfiltered, trainingdata[, soil_property])
    predictions <- cbind(predictions,predict(bart_model,Xtestfiltered))
}
 
predictions <- predictions[,-1]
colnames(predictions) <- soil_properties
write.csv(cbind(PIDN= as.character(testdata[,1]), predictions), "../submissions/submit.bartMachine.95.csv", row.names=FALSE)
