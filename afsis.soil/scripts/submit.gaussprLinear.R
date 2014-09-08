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
# Bayesian Generalized Linear Model Regression Model
# ===================================================================

#predict Ca
gaussprLinearTuneCa <- train(x = Xtrainfiltered, y = Ytrain$Ca,
                 method = "gaussprLinear",
                 trControl = ctrl)
gaussprLinearTuneCa
#RMSE = 0.399
gaussprLinearResults <- data.frame(PIDN = IDtest,
                          Ca = predict(gaussprLinearTuneCa, Xtestfiltered))

#predict P
gaussprLinearTuneP <- train(x = Xtrainfiltered, y = Ytrain$P,
                  method = "gaussprLinear",
                  trControl = ctrl)
gaussprLinearTuneP
#RMSE = 0.913
gaussprLinearResults$P <- predict(gaussprLinearTuneP,Xtestfiltered)

#predict pH
gaussprLinearTunepH <- train(x = Xtrainfiltered, y = Ytrain$pH,
                 method = "gaussprLinear",
                 trControl = ctrl)
gaussprLinearTunepH
#RMSE = 0.508
gaussprLinearResults$pH <- predict(gaussprLinearTunepH,Xtestfiltered)

#predict SOC
gaussprLinearTuneSOC <- train(x = Xtrainfiltered, y = Ytrain$SOC,
                  method = "gaussprLinear",
                  trControl = ctrl)
gaussprLinearTuneSOC
#RMSE = 0.512
gaussprLinearResults$SOC <- predict(gaussprLinearTuneSOC,Xtestfiltered)

#predict Sand
gaussprLinearTuneSand <- train(x = Xtrainfiltered, y = Ytrain$Sand,
                   method = "gaussprLinear",
                   trControl = ctrl)
gaussprLinearTuneSand
#RMSE = 0.494
gaussprLinearResults$Sand <- predict(gaussprLinearTuneSand,Xtestfiltered)


write.csv(gaussprLinearResults,file = "../submissions/submit.gaussprLinear.csv",row.names = FALSE)
