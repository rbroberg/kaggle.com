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
#RMSE = 0.511
rfResults$SOC <- predict(rfTuneSOC,Xtestfiltered)

#predict Sand
rfTuneSand <- train(x = Xtrainfiltered, y = Ytrain$Sand,
                   method = "rf",
                   trControl = ctrl)
rfTuneSand
#RMSE = 0.495
rfResults$Sand <- predict(rfTuneSand,Xtestfiltered)


write.csv(rfResults,file = "../submissions/submit.rf.csv",row.names = FALSE)
