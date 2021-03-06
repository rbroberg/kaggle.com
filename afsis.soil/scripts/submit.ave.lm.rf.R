# http://www.kaggle.com/c/afsis-soil-properties/forums/t/10176/simple-linear-regression-starting-kit

# lm alone: 0.57855
# rf alone:  0.67984
# lm + rf: 0.59239

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
# Linear Model
# ===================================================================

#predict Ca
lmTuneCa <- train(x = Xtrainfiltered, y = Ytrain$Ca,
                 method = "lm",
                 trControl = ctrl)
lmTuneCa
#RMSE = 0.409
lmResults <- data.frame(PIDN = IDtest,
                          Ca = predict(lmTuneCa, Xtestfiltered))

#predict P
lmTuneP <- train(x = Xtrainfiltered, y = Ytrain$P,
                  method = "lm",
                  trControl = ctrl)
lmTuneP
#RMSE = 0.925
lmResults$P <- predict(lmTuneP,Xtestfiltered)

#predict pH
lmTunepH <- train(x = Xtrainfiltered, y = Ytrain$pH,
                 method = "lm",
                 trControl = ctrl)
lmTunepH
#RMSE = 0.508
lmResults$pH <- predict(lmTunepH,Xtestfiltered)

#predict SOC
lmTuneSOC <- train(x = Xtrainfiltered, y = Ytrain$SOC,
                  method = "lm",
                  trControl = ctrl)
lmTuneSOC
#RMSE = 0.511
lmResults$SOC <- predict(lmTuneSOC,Xtestfiltered)

#predict Sand
lmTuneSand <- train(x = Xtrainfiltered, y = Ytrain$Sand,
                   method = "lm",
                   trControl = ctrl)
lmTuneSand
#RMSE = 0.495
lmResults$Sand <- predict(lmTuneSand,Xtestfiltered)

# ===================================================================
# Random Forest Model
# ===================================================================

#predict Ca
rfTuneCa <- train(x = Xtrainfiltered, y = Ytrain$Ca,
                 method = "rf",
                 trControl = ctrl)
rfTuneCa
#RMSE = 0.342, mtry=14
rfResults <- data.frame(PIDN = IDtest,
                          Ca = predict(rfTuneCa, Xtestfiltered))

#predict P
rfTuneP <- train(x = Xtrainfiltered, y = Ytrain$P,
                  method = "rf",
                  trControl = ctrl)
rfTuneP
#RMSE = 0.789, mtry = 2
rfResults$P <- predict(rfTuneP,Xtestfiltered)

#predict pH
rfTunepH <- train(x = Xtrainfiltered, y = Ytrain$pH,
                 method = "rf",
                 trControl = ctrl)
rfTunepH
#RMSE = 0.432, mtry = 14
rfResults$pH <- predict(rfTunepH,Xtestfiltered)

#predict SOC
rfTuneSOC <- train(x = Xtrainfiltered, y = Ytrain$SOC,
                  method = "rf",
                  trControl = ctrl)
rfTuneSOC
#RMSE = 0.451, mtry=14
rfResults$SOC <- predict(rfTuneSOC,Xtestfiltered)

#predict Sand
rfTuneSand <- train(x = Xtrainfiltered, y = Ytrain$Sand,
                   method = "rf",
                   trControl = ctrl)
rfTuneSand
#RMSE = 0.366, mtry=14
rfResults$Sand <- predict(rfTuneSand,Xtestfiltered)

# ===================================================================
# Average on Models
# ===================================================================
aveResults = lmResults
aveResults[,2:6]=(lmResults[,2:6]+rfResults[,2:6])/2

write.csv(aveResults,file = "../submissions/submit.ave.lm.rf.csv",row.names = FALSE)
