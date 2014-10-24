# assume positive
# eliminate known ictal
ver="naive.ac.v01."

function auc(tp,fp)
	(0.5*tp*fp)+(tp*(1.-fp))+(0.5(1.-tp)*(1.-fp))
end

function rowmax(x)
	[maximum(x[i,:]) for i=1:size(x)[1]];
end

function rowmin(x)
	[minimum(x[i,:]) for i=1:size(x)[1]];
end

cases=[ ["Dog_1",     24,  480,  502],
        ["Dog_2",     42,  500, 1000],
        ["Dog_3",     72, 1440,  907],
        ["Dog_4",     97,  804,  990],
        ["Patient_1", 18,   50,  195],
        ["Patient_2", 18,   42,  150]];
cases=transpose(reshape(cases,(4,6)));

cases=cases[1:4,:]; # just dogs

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
	acmax=rowmax(ac);
	acmaxpre=maximum(acmax[1:npre]);
	threshold = sort(acmax[1:npre])[npre-1] + 1;
	# sum(acmax[npre+ninter+1:end].>threshold)
	for n in 1:ntest
		fn=case*testictal*@sprintf("%04d",n)*".mat"
		i=npre+ninter+n
		if acmax[i]>threshold
			push!(clips,fn)
			push!(ictals,0)
		else
			push!(clips,fn)
			push!(ictals,1)
		end
	end
end

r=hcat(clips,ictals);
dts=strftime("%Y%m%d%H%M%S",ifloor(time())) # datetime stamp
submitdir="/projects/kaggle.com/pred.seizures/submissions/"
sfn=submitdir*"submit."*ver*dts*".csv"

writedlm(sfn,r,',')
println("manually add header to submit file: "*sfn)

# data exploration
# concentrate on autocorrelation
#=
'''
ac=readdlm(datadir*case*"/ac.csv",',');
var=readdlm(datadir*case*"/var.csv",',');
ent=readdlm(datadir*case*"/ent.csv",',');

acmax=rowmax(ac);
varmax=rowmax(var);
entmax=rowmax(ent);

acmin=rowmin(ac);
varmin=rowmin(var);
entmin=rowmin(ent);

i=1;j=npre;
acmaxpre=maximum(acmax[1:npre]);
varmaxpre=maximum(varmax[1:npre]);
entmaxpre=maximum(entmax[1:npre]);

i=1;j=npre;
acminpre=minimum(acmin[1:npre]);
varminpre=minimum(varmin[1:npre]);
entminpre=minimum(entmin[1:npre]);

i=npre+1;j=npre+ninter;
sum(acmax[i:j].>acmaxpre) # 147
sum(varmax[i:j].>varmaxpre) # 117
sum(entmax[i:j].>entmaxpre) # 94
sum(acmax[i:j].>acmaxpre) & sum(varmax[i:j].>varmaxpre)   # 17
sum(acmax[i:j].>acmaxpre) & sum(entmax[i:j].>entmaxpre)   # 18
sum(varmax[i:j].>varmaxpre) & sum(entmax[i:j].>entmaxpre) # 84
sum(acmax[i:j].>acmaxpre) & sum(varmax[i:j].>varmaxpre) & sum(entmax[i:j].>entmaxpre) # 16

i=npre+1;j=npre+ninter;
sum(acmin[i:j].<acminpre) # 38
sum(varmin[i:j].<varminpre) # 20
sum(entmin[i:j].<entminpre) # 17
sum(acmin[i:j].<acminpre) & sum(varmin[i:j].<varminpre)   # 4
sum(acmin[i:j].<acminpre) & sum(entmin[i:j].<entminpre)   # 0
sum(varmin[i:j].<varminpre) & sum(entmin[i:j].<entminpre) # 16
sum(acmin[i:j].<acminpre) & sum(varmin[i:j].<varminpre) & sum(entmin[i:j].<entminpre) # 0
'''
=#