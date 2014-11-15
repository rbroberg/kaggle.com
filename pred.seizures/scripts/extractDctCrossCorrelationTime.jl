using HDF5
using StatsBase

function myxcorr_absmax(d)
    nc=size(d)[2];
    n=int((nc^2-nc)/2);
    cc=zeros(n,1);
    k=0;
    for i=1:nc
        for j=(i+1):nc
            k=k+1
            #cc[k]=crosscor(d[:,i],d[:,j])[52]
            c=crosscor(d[:,i],d[:,j])
            cc[k]=c[indmax(abs(c))]
        end
    end
    return(cc)
end

function extractDctCrossCorr(case,npre,ninter,ntest)
    datadir="/data/www.kaggle.com/c/seizure-prediction/download/";
    interictal=case*"_interictal_segment_";
    preictal=case*"_preictal_segment_";
    testictal=case*"_test_segment_";
    
    println("extractDctCrossCorr:read first file for channels")
    fn=datadir*case*"/"*preictal*@sprintf("%04d", 1)*".hdf5";
    flush(STDOUT)
    println(preictal*@sprintf("%04d", 1)*".hdf5")
    data=h5open(fn,"r");
    nc=read(data["nchannels"])["values"][1];
    dat=read(data["data"])["block0_values"];
    close(data)
    n=int((nc^2-nc)/2);
    chunk=int(size(dat,1)/n -0.5) # round down to keep chunks withing boundary
    ccpre=zeros(npre,n*n);
    println("extractDctCrossCorrTime:read preictal")
    for i in 1:npre
        fn=datadir*case*"/"*preictal*@sprintf("%04d", i)*".hdf5";
        println(preictal*@sprintf("%04d", i)*".hdf5")
        flush(STDOUT)
        data=h5open(fn,"r");
        dat=read(data["data"])["block0_values"];
        for k in 1:n
            ii=(k-1)*chunk+1;jj=(k*chunk);
            iii=(k-1)*n+1;jjj=(k*n)
            ccpre[i,iii:jjj]=transpose(myxcorr_absmax(dct(dat[ii:jj,:])));
        end
        close(data);
    end
    ccinter=zeros(ninter,n*n);
    println("extractDctCrossCorrTime:read interictal")
    for i in 1:ninter
        fn=datadir*case*"/"*interictal*@sprintf("%04d", i)*".hdf5";
        println(interictal*@sprintf("%04d", i)*".hdf5")
        flush(STDOUT)
        data=h5open(fn,"r");
        dat=read(data["data"])["block0_values"];
        for k in 1:n
            ii=(k-1)*chunk+1;jj=(k*chunk);
            iii=(k-1)*n+1;jjj=(k*n)
            ccinter[i,iii:jjj]=transpose(myxcorr_absmax(dct(dat[ii:jj,:])));
        end
        close(data);
    end
    # read the test data
    println("extractDctCrossCorrTime: read the test data")
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
            cctest[i,iii:jjj]=transpose(myxcorr_absmax(dct(dat[ii:jj,:])));
        end
        close(data);
    end
    cc=vcat(ccpre,ccinter,cctest);
    writedlm(datadir*case*"/dct.csv",cc,',');
end

extractDctCrossCorr("Dog_1",24,480,502);
extractDctCrossCorr("Dog_2",42,500,1000);
extractDctCrossCorr("Dog_3",72,1440,907);
extractDctCrossCorr("Dog_4",97,804,990);
extractDctCrossCorr("Dog_5",30,450,191);
extractDctCrossCorr("Patient_1",18,50,195);
extractDctCrossCorr("Patient_2",18,42,150);
=+
'''
case="Dog_1"
npre=24
ninter=480
ntest=502
'''
+=
