#A=[[1,1,1],[0,2,5],[2,5,-1]]
#A=transpose(reshape(A,(3,3)))
#B=[6,-4,27]
#*(inv(A),B) # 5, 3, -2

using DataFrames
datadir="/projects/kaggle.com/afsis.soil/data/"
df = readtable(string(datadir,"training.csv"));
size(df) # 1157, 3600

function solveB(X,Y)
	return *(*(inv(*(transpose(X),X)),transpose(X)),Y)
end

# solve B for each freq in the spectrum
X=array(df[:,3596:3600]);
#Bs=[solveB(X,array(df[:,i])) for i in 2:3579]
Bs=zeros(3578,5);
dummy=[Bs[i-1,:]=solveB(X,array(df[:,i])) for i in 2:3579]

# now, given y, find X
y=array(df[1,2:3579])
*(y,Bs)

# experiment
x=X[2,:]
sum((array(df[2,2:3579])-transpose(*(Bs,transpose(x)))).^2)