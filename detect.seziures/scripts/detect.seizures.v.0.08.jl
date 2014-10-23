push!(Sys.DL_LOAD_PATH, "/usr/lib")
using MAT
using DataFrames

datadir="/eeg2/kaggle.com/data/"

searchdir(path,key) = filter(x->contains(x,key), readdir(path));

function entropy(x)
    y=x[1:size(x)[2]];
    freq=hist(y,[floor(minimum(y)):floor(maximum(y)+1)]);
    probs=.000001+freq[2]/sum(freq[2]);
    z=-1.*sum(probs .* log(2,probs));
    if isequal(z,NaN) 0 else z end;
end

function getFileEntropies(f)
    vars = matread(f);
    sort([entropy(vars["data"][i,:]) for i in 1:size(vars["data"])[1]],rev=true);
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

cases=["Dog_1","Dog_2","Dog_3","Dog_4","Patient_1","Patient_2","Patient_3","Patient_4","Patient_5","Patient_6","Patient_7","Patient_8"];
nfiles=[3181,2997,4450,3013,2050,3894,1281,543,2986,2997,3601,1922]

for i in 1:length(cases)

    dir=string(datadir,cases[i])
    # print(string(dir,"\n"))

    szfiles=searchdir(dir,"_ictal_"); szfiles=[string(dir,"/",szfiles[i]) for i in 1:length(szfiles)]
    nofiles=searchdir(dir,"_interictal_"); nofiles=[string(dir,"/",nofiles[i]) for i in 1:length(nofiles)]
    testfiles=[string(cases[i],"_test_segment_",string(j),".mat") for j in 1:nfiles[i]];

    for j in 1:length(szfiles)
        e=getFileEntropies(szfiles[j])
        v=getFileStds(szfiles[j])
        print(split(szfiles[j],'/')[end],',',1,',',length(e),',',e,',',v,"\n")
    end

    for j in 1:length(nofiles)
        e=getFileEntropies(nofiles[j])
        v=getFileStds(nofiles[j])
        print(split(nofiles[j],'/')[end],',',0,',',length(e),',',e,',',v,"\n")
    end

    for j in 1:length(testfiles)
        e=getFileEntropies(string(dir,"/",testfiles[j]))
        v=getFileStds(string(dir,"/",testfiles[j]))
        print(split(testfiles[j],'/')[end],',',2,',',length(e),',',e,',',v,"\n")
    end

end



