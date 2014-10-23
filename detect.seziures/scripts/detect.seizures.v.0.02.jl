push!(Sys.DL_LOAD_PATH, "/usr/lib")
using MAT
using DataFrames

datadir="/eeg2/kaggle.com/data/"
#datadir="/google/copy/projects/kaggle/seizures/data/"

searchdir(path,key) = filter(x->contains(x,key), readdir(path));

function entropy(x)
    y=x[1:size(x)[2]];
    freq=hist(y,[floor(minimum(y)):floor(maximum(y)+1)]);
    probs=.000001+freq[2]/sum(freq[2]);
    -1.*sum(probs .* log(2,probs));
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

sampleSubmission="../download/sampleSubmission.csv"
dfSS = readtable(sampleSubmission);

for i in 1:length(cases)
#for i in 1:2

    dir=string(datadir,cases[i])
    # print(string(dir,"\n"))

    szfiles=searchdir(dir,"_ictal_"); szfiles=[string(dir,"/",szfiles[j]) for j in 1:length(szfiles)]
    nofiles=searchdir(dir,"_interictal_"); nofiles=[string(dir,"/",nofiles[j]) for j in 1:length(nofiles)]

    szents=[getFileEntropies(fn) for fn in szfiles];
    noents=[getFileEntropies(fn) for fn in nofiles];
    szstds=[getFileStds(fn) for fn in szfiles];
    nostds=[getFileStds(fn) for fn in nofiles];

    sz=[[1,cases[i],szents[j],szstds[j]] for j in 1:length(szfiles)]
    no=[[0,cases[i],noents[j],nostds[j]] for j in 1:length(nofiles)]
    all=[sz,no]
    for j in 1:length(all)
        print(all[j],"\n")
    end
end



