push!(Sys.DL_LOAD_PATH, "/usr/lib")
using MAT
using DataFrames
#using Wavelets

datadir="/eeg2/kaggle.com/data/"

searchdir(path,key) = filter(x->contains(x,key), readdir(path));

function myentropy(x)
    y=x[1:size(x)[2]];
    freq=hist(y,[floor(minimum(y)):floor(maximum(y)+1)]);
    probs=.000001+freq[2]/sum(freq[2]);
    -1.*sum(probs .* log(2,probs));
end

function getFileEntropies(f)
    vars = matread(f);
    sort([myentropy(vars["data"][i,:]) for i in 1:size(vars["data"])[1]],rev=true);
end

function getFileStds(f)
    vars = matread(f);
    sort([std(vars["data"][i,:]) for i in 1:size(vars["data"])[1]],rev=true);
end

function getDistance(x,y,u,v)
    x=sum([i^2 for i in x-u]);
    y=sum([1000*i^2 for i in y-v]);
    return((x^2+y^2)^0.5);
end

function getAUC(x)
    brks=[0];
    freq=length(x)
    for i in 2:(freq-1)
        if (x[i]>0) != (x[i+1]>0)
            push!(brks,i);
        end
    end
    push!(brks,freq);
    parts=[x[(brks[i]+1):brks[i+1]] for i in 1:(length(brks)-1)];
    [[abs(sum(parts[i])),length(parts[i])] for i in 1:length(parts)]
end            
            
function getFileAUC(f)
    vars = matread(f);
    [getAUC(vars["data"][i,:]) for i in 1:size(vars["data"])[1]];
end

cases=["Dog_1","Dog_2","Dog_3","Dog_4","Patient_1","Patient_2","Patient_3","Patient_4","Patient_5","Patient_6","Patient_7","Patient_8"];
nfiles=[3181,2997,4450,3013,2050,3894,1281,543,2986,2997,3601,1922]

for i in 1:length(cases)

    dir=string(datadir,cases[i])
    # print(string(dir,"\n"))

    szfiles=searchdir(dir,"_ictal_"); szfiles=[string(dir,"/",szfiles[i]) for i in 1:length(szfiles)];
    nofiles=searchdir(dir,"_interictal_"); nofiles=[string(dir,"/",nofiles[i]) for i in 1:length(nofiles)];
    testfiles=[string(cases[i],"_test_segment_",string(j),".mat") for j in 1:nfiles[i]];

    for j in 1:length(szfiles)
        e=getFileEntropies(szfiles[j])
        em=mean(float(e))
        ev=std(float(e))
        v=getFileStds(szfiles[j])
        vm=mean(float(v))
        vv=std(float(v))
        y=getFileAUC(szfiles[j]);
        ym1=mean(float([z[1] for z in y[i]]))
        yv1=std(float([z[1] for z in y[i]]))
        ym2=mean(float([z[2] for z in y[i]]))
        yv2=std(float([z[2] for z in y[i]]))
      
        print(split(szfiles[j],'/')[end],',',i,',',1,',',length(e),',',em,',',ev,',',vm,',',vv,',',ym1,',',yv1,',',ym2,',',yv2,"\n")
    end

    for j in 1:length(nofiles)
        e=getFileEntropies(nofiles[j])
        em=mean(float(e))
        ev=std(float(e))
        v=getFileStds(nofiles[j])
        vm=mean(float(v))
        vv=std(float(v))
        y=getFileAUC(nofiles[j]);
        ym1=mean(float([z[1] for z in y[i]]))
        yv1=std(float([z[1] for z in y[i]]))
        ym2=mean(float([z[2] for z in y[i]]))
        yv2=std(float([z[2] for z in y[i]]))
      
        print(split(nofiles[j],'/')[end],',',i,',',0,',',length(e),',',em,',',ev,',',vm,',',vv,',',ym1,',',yv1,',',ym2,',',yv2,"\n")
    end

    for j in 1:length(testfiles)
        e=getFileEntropies(string(dir,"/",testfiles[j]))
        em=mean(float(e))
        ev=std(float(e))
        v=getFileStds(string(dir,"/",testfiles[j]))
        vm=mean(float(v))
        vv=std(float(v))
        y=getFileAUC(string(dir,"/",testfiles[j]));
        ym1=mean(float([z[1] for z in y[i]]))
        yv1=std(float([z[1] for z in y[i]]))
        ym2=mean(float([z[2] for z in y[i]]))
        yv2=std(float([z[2] for z in y[i]]))
      
        print(split(testfiles[j],'/')[end],',',i,',',2,',',length(e),',',em,',',ev,',',vm,',',vv,',',ym1,',',yv1,',',ym2,',',yv2,"\n")
    end

end



