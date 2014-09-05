# http://www.kaggle.com/c/afsis-soil-properties/forums/t/10176/simple-linear-regression-starting-kit

# lm alone: 0.57855
# rf alone:
# lm + rf: 

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
# Stochastic Gradient Boosting
# ===================================================================

library(gbm)

#predict Ca
gbmTuneCa <- train(x = Xtrainfiltered, y = Ytrain$Ca,
                 method = "gbm",
                 trControl = ctrl)
gbmTuneCa
#RMSE = 0.399, k=27, lambda=0.001
gbmResults <- data.frame(PIDN = IDtest,
                          Ca = predict(gbmTuneCa, Xtestfiltered))

#predict P
gbmTuneP <- train(x = Xtrainfiltered, y = Ytrain$P,
                  method = "gbm",
                  trControl = ctrl)
gbmTuneP
#RMSE = 0.913, k=14, lambda=0.1
gbmResults$P <- predict(gbmTuneP,Xtestfiltered)

#predict pH
gbmTunepH <- train(x = Xtrainfiltered, y = Ytrain$pH,
                 method = "gbm",
                 trControl = ctrl)
gbmTunepH
#RMSE = 0.508, k=27, lambda=0.001
gbmResults$pH <- predict(gbmTunepH,Xtestfiltered)

#predict SOC
gbmTuneSOC <- train(x = Xtrainfiltered, y = Ytrain$SOC,
                  method = "gbm",
                  trControl = ctrl)
gbmTuneSOC
#RMSE = 0.512, k=27, lambda=0.001
gbmResults$SOC <- predict(gbmTuneSOC,Xtestfiltered)

#predict Sand
gbmTuneSand <- train(x = Xtrainfiltered, y = Ytrain$Sand,
                   method = "gbm",
                   trControl = ctrl)
gbmTuneSand
#RMSE = 0.475, k=27, lambda=0.001
gbmResults$Sand <- predict(gbmTuneSand,Xtestfiltered)


write.csv(gbmResults,file = "../submissions/submit.gbm.csv",row.names = FALSE)
