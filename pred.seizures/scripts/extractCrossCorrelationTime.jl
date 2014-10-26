using HDF5
using StatsBase

function myxcorr(d)
    nc=size(d)[2];
    n=int((nc^2-nc)/2);
    cc=zeros(n,1);
    k=0;
    for i=1:nc
        for j=(i+1):nc
            k=k+1
            cc[k]=crosscor(d[:,i],d[:,j])[52]
        end
    end
    return(cc)
end

function extractCrossCorr(case,npre,ninter,ntest)
    datadir="/data/www.kaggle.com/c/seizure-prediction/download/";
    interictal=case*"_interictal_segment_";
    preictal=case*"_preictal_segment_";
    testictal=case*"_test_segment_";
    
    println("extractCrossCorr:read first file for channels")
    fn=datadir*case*"/"*preictal*@sprintf("%04d", 1)*".hdf5";
    flush(STDOUT)
    println(preictal*@sprintf("%04d", 1)*".hdf5")
    data=h5open(fn,"r");
    nc=read(data["nchannels"])["values"][1];
    close(data)
    n=int((nc^2-nc)/2);
    chunk=int(size(dat,1)/n)
    ccpre=zeros(npre,n*n);
    println("extractCrossCorrTime:read preictal")
    for i in 1:npre
        fn=datadir*case*"/"*preictal*@sprintf("%04d", i)*".hdf5";
        flush(STDOUT)
        println(preictal*@sprintf("%04d", i)*".hdf5")
        data=h5open(fn,"r");
        dat=read(data["data"])["block0_values"];
        for k in 1:n
            ii=(k-1)*chunk+1;jj=(k*chunk);
            iii=(k-1)*n+1;jjj=(k*n)
            ccpre[i,iii:jjj]=transpose(myxcorr(dat[ii:jj,:]));
        end
        close(data);
    end
    ccinter=zeros(ninter,n*n);
    println("extractCrossCorrTime:read interictal")
    for i in 1:ninter
        fn=datadir*case*"/"*interictal*@sprintf("%04d", i)*".hdf5";
        println(interictal*@sprintf("%04d", i)*".hdf5")
        flush(STDOUT)
        data=h5open(fn,"r");
        dat=read(data["data"])["block0_values"];
        for k in 1:n
            ii=(k-1)*chunk+1;jj=(k*chunk);
            iii=(k-1)*n+1;jjj=(k*n)
            ccinter[i,iii:jjj]=transpose(myxcorr(dat[ii:jj,:]));
        end
        close(data);
    end
    # read the test data
    println("extractCrossCorrTime: read the test data")
    cctest=zeros(ntest,n*n);
    for i in 1:ntest
        fn=datadir*case*"/"*testictal*@sprintf("%04d", i)*".hdf5";
        println(testictal*@sprintf("%04d", i)*".hdf5")
        flush(STDOUT)
        data=h5open(fn,"r");
        dat=read(data["data"])["block0_values"];
        for k in 1:n
            ii=(k-1)*chunk+1;jj=(k*chunk);
            iii=(k-1)*n+1;jjj=(k*n)
            ccteest[i,iii:jjj]=transpose(myxcorr(dat[ii:jj,:]));
        end
        close(data);
    end
    cc=vcat(ccpre,ccinter,cctest);
    writedlm(datadir*case*"/cctime.csv",cc,',');
end

extractCrossCorr("Dog_1",24,480,502);
extractCrossCorr("Dog_2",42,500,1000);
extractCrossCorr("Dog_3",72,1440,907);
extractCrossCorr("Dog_4",97,804,990);
extractCrossCorr("Dog_5",30,450,191);
extractCrossCorr("Patient_1",18,50,195);
extractCrossCorr("Patient_2",18,42,150);
=+
'''
case="Dog_1"
npre=24
ninter=480
ntest=502
'''
+=