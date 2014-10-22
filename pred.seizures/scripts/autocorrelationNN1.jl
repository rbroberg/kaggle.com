using HDF5

case="Dog_1"
npre=24
ninter=480
ntest=502
epochs=10 # not real epochs but e*size(traing) random draws 
function acModel(case,npre,ninter,ntest, epochs)
	datadir="/data/www.kaggle.com/c/seizure-prediction/download/";
	interictal=case*"_interictal_segment_";
	preictal=case*"_preictal_segment_";
	testictal=case*"_test_segment_"
	acpre=zeros(npre,16);
	println("acModel:read preictal")
	for i in 1:npre
		fn=datadir*case*"/"*preictal*@sprintf("%04d", i)*".hdf5";
		data=h5open(fn,"r");
		nc=read(data["nchannels"])["values"][1];
		dat=read(data["data"])["block0_values"];
		acpre[i,:]=[var(xcorr(dat[:,j],dat[:,j])[size(dat)[1]:end]) for j in 1:nc]/1e10;
		close(data);
	end
	acinter=zeros(ninter,16);
	println("acModel:read interictal")
	for i in 1:ninter
		fn=datadir*case*"/"*interictal*@sprintf("%04d", i)*".hdf5";
		data=h5open(fn,"r");
		nc=read(data["nchannels"])["values"][1];
		dat=read(data["data"])["block0_values"];
		acinter[i,:]=[var(xcorr(dat[:,j],dat[:,j])[size(dat)[1]:end]) for j in 1:nc]/1e10;
		close(data);
	end

	println("acModel:scale and center data")
	logacpre=log(acpre,10); # scale downward
	logacpre=logacpre - mean(logacpre); # center
	logacinter=log(acinter,10); # scale downward
	logacinter=logacinter - mean(logacinter); # center

	wgts=randn(137,1); # include bias
	wgts=wgts/(2.*maximum(abs(wgts))); # rand gauss +/- 0.5

	L1=(Int64,Int64)[];
	for i in 1:16
		for j in i:16
			push!(L1,(i,j));
		end
	end

	# train
	println("acModel: train data, epochs = ",epochs," loops = ",epochs*size(logacinter)[1])
	for j in 1:(epochs*size(logacinter)[1])
		k=rand(1:npre); # random draw for precital
		X1=zeros(137);
		for i in 1:length(L1)
			X1[i]=logacpre[k,L1[i][1]]+logacpre[k,L1[i][2]];
		end;
		X1[137]=1; # bias
		Yin = dot(wgts[:,1],X1)
		if Yin < 0
			wgts=wgts+0.05*X1*(1);
		end;

		k=rand(1:ninter); # random draw from interictal
		X1=zeros(137);
		for i in 1:length(L1)
			X1[i]=logacinter[k,L1[i][1]]+logacinter[k,L1[i][2]];
		end
		X1[137]=1; # bias
		Z1=
		Yin = dot(wgts[:,1],X1)
		if Yin > 0
			wgts=wgts+0.05*X1*(1);
		end;
	end

	# test the training
	println("acModel: test the training")
	tp=0
	for k in 1:npre
		for i in 1:length(L1)
			X1[i]=logacpre[k,L1[i][1]]+logacpre[k,L1[i][2]];
		end;
		X1[137]=1; # bias
		Yin = dot(wgts[:,1],X1)
		if Yin > 0
			tp=tp+1
		end
		#println(Yin)
	end
	println("acModel: tp = ",tp,'\t',tp/npre)
	tn=0
	for k in 1:ninter
		for i in 1:length(L1)
			X1[i]=logacinter[k,L1[i][1]]+logacinter[k,L1[i][2]];
		end;
		X1[137]=1; # bias
		Yin = dot(wgts[:,1],X1)
		if Yin < 0
			tn=tn+1
		end
		#println(Yin)
	end
	println("acModel: tn = ",tn,'\t',tn/ninter)

	# read the test data
	println("acModel: read the test data")
	actest=zeros(ntest,16);
	for i in 1:ntest
		fn=datadir*case*"/"*testictal*@sprintf("%04d", i)*".hdf5";
		data=h5open(fn,"r");
		nc=read(data["nchannels"])["values"][1];
		dat=read(data["data"])["block0_values"];
		actest[i,:]=[var(xcorr(dat[:,j],dat[:,j])[size(dat)[1]:end]) for j in 1:nc]/1e10;
		close(data);
	end

	# test the test data
	logactest=log(actest,10); # scale downward
	logactest=logactest - mean(logactest); # center
	p=0;
	results=(ASCIIString,Int64)[];
	for k in 1:ntest
		X1=zeros(137);
		for i in 1:length(L1)
			X1[i]=logactest[k,L1[i][1]]+logactest[k,L1[i][2]];
		end;
		Yin = dot(wgts[:,1],X1);
		if Yin>0
			p=p+1;
			println(testictal*@sprintf("%04d",k),',', 1);
			push!(results,(testictal*@sprintf("%04d",k),1))
		else
			println(testictal*@sprintf("%04d",k),',', 0);
			push!(results,(testictal*@sprintf("%04d",k),0))
		end;
	end;
	println("acModel: p = ",p/ntest)
	return results;
end