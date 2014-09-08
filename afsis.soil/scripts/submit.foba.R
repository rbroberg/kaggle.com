# http://www.kaggle.com/c/afsis-soil-properties/forums/t/10176/simple-linear-regression-starting-kit

# foba: 0.57704

rm(list = ls())
setwd("/projects/kaggle.com/afsis.soil/scripts")
library(caret)

train <- read.csv('../data/training.csv', header = TRUE)
test <- read.csv('../data/sorted_test.csv',header = TRUE)
train$Depth <-  with ( train, ifelse ( ( Depth == 'Subsoil' ), 0 , 1 ) )
test$Depth <-  with ( test, ifelse ( ( Depth == 'Subsoil' ), 0 , 1 ) ) 

Xtrain <- train[,2:3595]
Ytrain <- train[,3596:3600]
Xtest <- test[,2:3595]
IDtest <- test[,1]


#delete highly correlated (>0.95) features.
tooHigh <- findCorrelation(cor(rbind(Xtrain,Xtest)), .95)

Xtrainfiltered <- Xtrain[, -tooHigh]
Xtestfiltered  <-  Xtest[, -tooHigh]

set.seed(1234)
# 10 fold cv
indx <- createFolds(Ytrain[,1], returnTrain = TRUE)
ctrl <- trainControl(method = "cv", index = indx)

# ===================================================================
# Forward Backward Ridge Regression with Variable Selection
# ===================================================================

library(foba)

#predict Ca
fobaTuneCa <- train(x = Xtrainfiltered, y = Ytrain$Ca,
                 method = "foba",
                 trControl = ctrl)
fobaTuneCa
#RMSE = 0.399, k=27, lambda=0.001
fobaResults <- data.frame(PIDN = IDtest,
                          Ca = predict(fobaTuneCa, Xtestfiltered))

#predict P
fobaTuneP <- train(x = Xtrainfiltered, y = Ytrain$P,
                  method = "foba",
                  trControl = ctrl)
fobaTuneP
#RMSE = 0.913, k=14, lambda=0.1
fobaResults$P <- predict(fobaTuneP,Xtestfiltered)

#predict pH
fobaTunepH <- train(x = Xtrainfiltered, y = Ytrain$pH,
                 method = "foba",
                 trControl = ctrl)
fobaTunepH
#RMSE = 0.508, k=27, lambda=0.001
fobaResults$pH <- predict(fobaTunepH,Xtestfiltered)

#predict SOC
fobaTuneSOC <- train(x = Xtrainfiltered, y = Ytrain$SOC,
                  method = "foba",
                  trControl = ctrl)
fobaTuneSOC
#RMSE = 0.512, k=27, lambda=0.001
fobaResults$SOC <- predict(fobaTuneSOC,Xtestfiltered)

#predict Sand
fobaTuneSand <- train(x = Xtrainfiltered, y = Ytrain$Sand,
                   method = "foba",
                   trControl = ctrl)
fobaTuneSand
#RMSE = 0.475, k=27, lambda=0.001
fobaResults$Sand <- predict(fobaTuneSand,Xtestfiltered)


write.csv(fobaResults,file = "../submissions/submit.foba.csv",row.names = FALSE)
