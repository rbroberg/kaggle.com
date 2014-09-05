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
# Bagged Cart Model
# ===================================================================
library(ipred)
library(plyr)

#predict Ca
treebagTuneCa <- train(x = Xtrainfiltered, y = Ytrain$Ca,
                 method = "treebag",
                 trControl = ctrl)
treebagTuneCa
#RMSE = 0.472
treebagResults <- data.frame(PIDN = IDtest,
                          Ca = predict(treebagTuneCa, Xtestfiltered))

#predict P
treebagTuneP <- train(x = Xtrainfiltered, y = Ytrain$P,
                  method = "treebag",
                  trControl = ctrl)
treebagTuneP
#RMSE = 0.903
treebagResults$P <- predict(treebagTuneP,Xtestfiltered)

#predict pH
treebagTunepH <- train(x = Xtrainfiltered, y = Ytrain$pH,
                 method = "treebag",
                 trControl = ctrl)
treebagTunepH
#RMSE = 0.553
treebagResults$pH <- predict(treebagTunepH,Xtestfiltered)

#predict SOC
treebagTuneSOC <- train(x = Xtrainfiltered, y = Ytrain$SOC,
                  method = "treebag",
                  trControl = ctrl)
treebagTuneSOC
#RMSE = 0.558
treebagResults$SOC <- predict(treebagTuneSOC,Xtestfiltered)

#predict Sand
treebagTuneSand <- train(x = Xtrainfiltered, y = Ytrain$Sand,
                   method = "treebag",
                   trControl = ctrl)
treebagTuneSand
#RMSE = 0.475
treebagResults$Sand <- predict(treebagTuneSand,Xtestfiltered)


write.csv(treebagResults,file = "../submissions/submit.treebag.csv",row.names = FALSE)
