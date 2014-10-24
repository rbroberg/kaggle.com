ver="ann.ac.var.ent.v01."

using ANN

function auc(tp,fp)
	(0.5*tp*fp)+(tp*(1.-fp))+(0.5(1.-tp)*(1.-fp))
end

cases=[ ["Dog_1",     24,  480,  502],
        ["Dog_2",     42,  500, 1000],
        ["Dog_3",     72, 1440,  907],
        ["Dog_4",     97,  804,  990],
        ["Dog_5",     30,  450,  191],
        ["Patient_1", 18,   50,  195],
        ["Patient_2", 18,   42,  150]];
cases=transpose(reshape(cases,(4,7)));

datadir="/data/www.kaggle.com/c/seizure-prediction/download/";
interictal="_interictal_segment_";
preictal="_preictal_segment_";
testictal="_test_segment_";

clips=Any[];
ictals=Float64[];
for c in 1:size(cases)[1]
	case=cases[c,1]
	npre=cases[c,2]
	ninter=cases[c,3]
	ntest=cases[c,4]
	
	ac=readdlm(datadir*case*"/ac.csv",',');	
	var=readdlm(datadir*case*"/var.csv",',');	
	ent=readdlm(datadir*case*"/ent.csv",',');	
	xtrain=hcat(ac[1:(npre+ninter),:],var[1:(npre+ninter),:],ent[1:(npre+ninter),:]);
	ytrain=vcat(ones(npre),zeros(ninter));
	xtest=hcat(ac[(npre+ninter+1):end,:],var[(npre+ninter+1):end,:],ent[(npre+ninter+1):end,:]);
	nfeatures=size(xtrain)[2]
	ytrain=convert(Vector{Int64},ytrain);
	ann = ArtificialNeuralNetwork(nfeatures^2);
	fit!(ann,xtrain,ytrain,epochs=40,alpha=0.1,lambda=1e-5);
	preds = predict(ann,xtest);
	sum(preds[:,1].>0.5)
	for n in 1:length(preds)
		push!(ictals,preds[n])
		push!(clips,case*testictal*@sprintf("%04d",n)*".mat")
	end
end

train=convert(Vector{Int64},ytrain);


r=hcat(clips,ictals);
dts=strftime("%Y%m%d%H%M%S",ifloor(time())) # datetime stamp
submitdir="/projects/kaggle.com/pred.seizures/submissions/"
sfn=submitdir*"submit."*ver*dts*".csv"

writedlm(sfn,r,',')
println("manually add header to submit file: "*sfn)