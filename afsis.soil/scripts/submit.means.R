bayesglmResults=as.data.frame(read.csv(file = "../submissions/submit.bayesglm.csv"))
fobaResults=read.csv(file = "../submissions/submit.foba.csv")
gaussprLinearResults=read.csv(file = "../submissions/submit.gaussprLinear.csv")
gbmResults=read.csv(file = "../submissions/submit.gbm.csv")
lm98Results=read.csv(file = "../submissions/submit.lm.98.csv")
lmResults=read.csv(file = "../submissions/submit.lm.csv")

IDtest=lmResults[,1]

res= cbind(
    as.matrix(bayesglmResults[,2:6]),
    as.matrix(fobaResults[,2:6]),
    as.matrix(gaussprLinearResults[,2:6]),
    as.matrix(gbmResults[,2:6]),
    as.matrix(lm98Results[,2:6]),
    as.matrix(lmResults[,2:6]))

l=6

dmat=matrix(nrow=l,ncol=l)
for (i in 1:l){
    i1=(i-1)*5+1
    i2=(i-1)*5+5
    for (j in i:l) {
        j1=(j-1)*5+1
        j2=(j-1)*5+5
        dmat[i,j]=sum((res[,i1:i2]-res[,j1:j2])^2)
        dmat[j,i]=dmat[i,j]
    }
}

                          
# weighted average LM, LM98, FOBA, GBM
means=(as.matrix(fobaResults[,2:6])
    + 3*as.matrix(gbmResults[,2:6])
    + 2*as.matrix(lm98Results[,2:6])
    + as.matrix(lmResults[,2:6]))/7
wtmeanResults <- data.frame(lmResults[,1],wtmeans)
names(wtmeanResults) <- names(lmResults)
write.csv(wtmeanResults,file = "../submissions/submit.wtmeans.lm.lm98.gbm.foba.csv",row.names = FALSE)

# 
dmat2=matrix(nrow=l,ncol=1)
for (i in 1:l){
    i1=(i-1)*5+1
    i2=(i-1)*5+5
    dmat2[i,1]=sum((res[,i1:i2]-means[,1:5])^2)
}

# simple average LM, LM98, FOBA, GBM
means=(as.matrix(fobaResults[,2:6])
    + as.matrix(gbmResults[,2:6])
    + as.matrix(lm98Results[,2:6])
    + as.matrix(lmResults[,2:6]))/4
meanResults <- data.frame(lmResults[,1],means)
names(meanResults) <- names(lmResults)
write.csv(meanResults,file = "../submissions/submit.means.lm.lm98.gbm.foba.csv",row.names = FALSE)

# 
dmat3=matrix(nrow=l,ncol=1)
for (i in 1:l){
    i1=(i-1)*5+1
    i2=(i-1)*5+5
    dmat3[i,1]=sum((res[,i1:i2]-means[,1:5])^2)
}


# simple average LM, LM98, FOBA, GBM
allmeans=(as.matrix(fobaResults[,2:6])
    + as.matrix(gbmResults[,2:6])
    + as.matrix(bayesglmResults[,2:6])
    + as.matrix(gaussprLinearResults[,2:6])
    + as.matrix(lm98Results[,2:6])
    + as.matrix(lmResults[,2:6]))/6
allmeanResults <- data.frame(lmResults[,1],allmeans)
names(allmeanResults) <- names(lmResults)
write.csv(allmeanResults,file = "../submissions/submit.means.all.6.csv",row.names = FALSE)

# 
dmat4=matrix(nrow=l,ncol=1)
for (i in 1:l){
    i1=(i-1)*5+1
    i2=(i-1)*5+5
    dmat4[i,1]=sum((res[,i1:i2]-allmeans[,1:5])^2)
}

bayesianRidgeResults=read.csv(file = "../submissions/submit.africa.BayesianRidge.csv")
gradientBoostingRegressorResults=read.csv(file = "../submissions/submit.africa.GradientBoostingRegressor.csv")
sum((bayesianRidgeResults[,2:6]-allmeans[,1:5])^2)
sum((gradientBoostingRegressorResults[,2:6]-allmeans[,1:5])^2)
sum((gradientBoostingRegressorResults[,2:6]-bayesianRidgeResults[,2:6])^2)