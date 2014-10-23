using HDF5

function myentropy(x)                                                                                                                                                                     
    y=x[1:size(x)[2]];                                                                                                                                                                    
    freq=hist(y,[floor(minimum(y)):floor(maximum(y)+1)]);                                                                                                                                 
    probs=.000001+freq[2]/sum(freq[2]);                                                                                                                                                   
    -1.*sum(probs .* log(2,probs));                                                                                                                                                       
end

function extractFeatures(case,npre,ninter,ntest)
	datadir="/data/www.kaggle.com/c/seizure-prediction/download/";
	interictal=case*"_interictal_segment_";
	preictal=case*"_preictal_segment_";
	testictal=case*"_test_segment_"
	acpre=zeros(npre,16);
	varpre=zeros(npre,16);
	entpre=zeros(npre,16);
	println("extractFeatures:read preictal")
	for i in 1:npre
		fn=datadir*case*"/"*preictal*@sprintf("%04d", i)*".hdf5";
		flush(STDOUT)
		println(preictal*@sprintf("%04d", i)*".hdf5")
		data=h5open(fn,"r");
		nc=read(data["nchannels"])["values"][1];
		dat=read(data["data"])["block0_values"];
		acpre[i,:]=[var(xcorr(dat[:,j],dat[:,j])[size(dat)[1]:end]) for j in 1:nc]/1e10;
		varpre[i,:]=[var(dat[:,j]) for j in 1:nc];
		entpre[i,:]=[myentropy(transpose(dat[:,j])) for j in 1:nc];
		close(data);
	end
	acinter=zeros(ninter,16);
	varinter=zeros(ninter,16);
	entinter=zeros(ninter,16);
	println("extractFeatures:read interictal")
	for i in 1:ninter
		fn=datadir*case*"/"*interictal*@sprintf("%04d", i)*".hdf5";
		println(interictal*@sprintf("%04d", i)*".hdf5")
		flush(STDOUT)
		data=h5open(fn,"r");
		nc=read(data["nchannels"])["values"][1];
		dat=read(data["data"])["block0_values"];
		acinter[i,:]=[var(xcorr(dat[:,j],dat[:,j])[size(dat)[1]:end]) for j in 1:nc]/1e10;
		varinter[i,:]=[var(dat[:,j]) for j in 1:nc];
		entinter[i,:]=[myentropy(transpose(dat[:,j])) for j in 1:nc];
		close(data);
	end
	# read the test data
	println("extractFeatures: read the test data")
	actest=zeros(ntest,16);
	vartest=zeros(ntest,16);
	enttest=zeros(ntest,16);
	for i in 1:ntest
		fn=datadir*case*"/"*testictal*@sprintf("%04d", i)*".hdf5";
		println(testictal*@sprintf("%04d", i)*".hdf5")
		flush(STDOUT)
		data=h5open(fn,"r");
		nc=read(data["nchannels"])["values"][1];
		dat=read(data["data"])["block0_values"];
		actest[i,:]=[var(xcorr(dat[:,j],dat[:,j])[size(dat)[1]:end]) for j in 1:nc]/1e10;
		vartest[i,:]=[var(dat[:,j]) for j in 1:nc];
		enttest[i,:]=[myentropy(transpose(dat[:,j])) for j in 1:nc];
		close(data);
	end
	aa=vcat(acpre,acinter,actest);
	vv=vcat(varpre,varinter,vartest);
	ee=vcat(entpre,entinter,enttest);
	writedlm(datadir*case*"/ac.csv",aa,',');
	writedlm(datadir*case*"/var.csv",vv,',');
	writedlm(datadir*case*"/ent.csv",ee,',');
end

extractFeatures("Dog_1",24,480,502);
extractFeatures("Dog_2",42,500,1000);
extractFeatures("Dog_3",72,1440,907);
extractFeatures("Dog_4",97,804,990);
extractFeatures("Patient_1",18,50,195);
extractFeatures("Patient_2",18,42,150);


