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
library(arm)

#predict Ca
bayesglmTuneCa <- train(x = Xtrainfiltered, y = Ytrain$Ca,
                 method = "bayesglm",
                 trControl = ctrl)
bayesglmTuneCa
#RMSE = 0.399
bayesglmResults <- data.frame(PIDN = IDtest,
                          Ca = predict(bayesglmTuneCa, Xtestfiltered))

#predict P
bayesglmTuneP <- train(x = Xtrainfiltered, y = Ytrain$P,
                  method = "bayesglm",
                  trControl = ctrl)
bayesglmTuneP
#RMSE = 0.913
bayesglmResults$P <- predict(bayesglmTuneP,Xtestfiltered)

#predict pH
bayesglmTunepH <- train(x = Xtrainfiltered, y = Ytrain$pH,
                 method = "bayesglm",
                 trControl = ctrl)
bayesglmTunepH
#RMSE = 0.508
bayesglmResults$pH <- predict(bayesglmTunepH,Xtestfiltered)

#predict SOC
bayesglmTuneSOC <- train(x = Xtrainfiltered, y = Ytrain$SOC,
                  method = "bayesglm",
                  trControl = ctrl)
bayesglmTuneSOC
#RMSE = 0.512
bayesglmResults$SOC <- predict(bayesglmTuneSOC,Xtestfiltered)

#predict Sand
bayesglmTuneSand <- train(x = Xtrainfiltered, y = Ytrain$Sand,
                   method = "bayesglm",
                   trControl = ctrl)
bayesglmTuneSand
#RMSE = 0.494
bayesglmResults$Sand <- predict(bayesglmTuneSand,Xtestfiltered)


write.csv(bayesglmResults,file = "../submissions/submit.bayesglm.csv",row.names = FALSE)
