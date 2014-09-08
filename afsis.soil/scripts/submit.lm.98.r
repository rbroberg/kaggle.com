# http://www.kaggle.com/c/afsis-soil-properties/forums/t/10176/simple-linear-regression-starting-kit
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
tooHigh <- findCorrelation(cor(rbind(Xtrain,Xtest)), .98)

Xtrainfiltered <- Xtrain[, -tooHigh]
Xtestfiltered  <-  Xtest[, -tooHigh]

set.seed(1234)
# 10 fold cv
indx <- createFolds(Ytrain[,1], returnTrain = TRUE)
ctrl <- trainControl(method = "cv", index = indx)

#predict Ca
lmTuneCa <- train(x = Xtrainfiltered, y = Ytrain$Ca,
                 method = "lm",
                 trControl = ctrl)
lmTuneCa
#RMSE = 0.357
lmResults <- data.frame(PIDN = IDtest,
                          Ca = predict(lmTuneCa, Xtestfiltered))

#predict P
lmTuneP <- train(x = Xtrainfiltered, y = Ytrain$P,
                  method = "lm",
                  trControl = ctrl)
lmTuneP
#RMSE = 0.909
lmResults$P <- predict(lmTuneP,Xtestfiltered)

#predict pH
lmTunepH <- train(x = Xtrainfiltered, y = Ytrain$pH,
                 method = "lm",
                 trControl = ctrl)
lmTunepH
#RMSE = 0.477
lmResults$pH <- predict(lmTunepH,Xtestfiltered)

#predict SOC
lmTuneSOC <- train(x = Xtrainfiltered, y = Ytrain$SOC,
                  method = "lm",
                  trControl = ctrl)
lmTuneSOC
#RMSE = 0.442
lmResults$SOC <- predict(lmTuneSOC,Xtestfiltered)

#predict Sand
lmTuneSand <- train(x = Xtrainfiltered, y = Ytrain$Sand,
                   method = "lm",
                   trControl = ctrl)
lmTuneSand
#RMSE = 0.495
lmResults$Sand <- predict(lmTuneSand,Xtestfiltered)

write.csv(lmResults,file = "../submissions/submit.lm.98.csv",row.names = FALSE)
