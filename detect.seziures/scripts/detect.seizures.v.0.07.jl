push!(Sys.DL_LOAD_PATH, "/usr/lib")
using MAT
using DataFrames

datadir="/eeg2/kaggle.com/data/"

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

    dir=string(datadir,cases[i])
    # print(string(dir,"\n"))

    szfiles=searchdir(dir,"_ictal_"); szfiles=[string(dir,"/",szfiles[i]) for i in 1:length(szfiles)]
    nofiles=searchdir(dir,"_interictal_"); nofiles=[string(dir,"/",nofiles[i]) for i in 1:length(nofiles)]
    testfiles=[string(cases[i],"_test_segment_",string(j),".mat") for j in 1:nfiles[i]];

    szents=[getFileEntropies(fn) for fn in szfiles];
    noents=[getFileEntropies(fn) for fn in nofiles];
    szentrng=[szents[i][1]-szents[i][16] for i in 1:size(szents)[1]];
    noentrng=[noents[i][1]-noents[i][16] for i in 1:size(noents)[1]];
    mean_noents=mean(noents);

    szstds=[getFileStds(fn) for fn in szfiles];
    nostds=[getFileStds(fn) for fn in nofiles];
    szstdrng=[szstds[i][1]-szstds[i][16] for i in 1:size(szstds)[1]];
    nostdrng=[nostds[i][1]-nostds[i][16] for i in 1:size(nostds)[1]];
    mean_nostds=mean(nostds);

    # szDs=[getDistance(szstds[i],szents[i],mean_nostds,mean_noents) for i in 1:length(szfiles)];
    # noDs=[getDistance(nostds[i],noents[i],mean_nostds,mean_noents) for i in 1:length(nofiles)];

    szDents=[sum([i^2 for i in x-mean_noents]) for x in szents];
    noDents=[sum([i^2 for i in x-mean_noents]) for x in noents];

    szdents=[sum([i for i in x-mean_noents]) for x in szents];
    nodents=[sum([i for i in x-mean_noents]) for x in noents];

    szDstds=[sum([i^2 for i in x-mean_nostds]) for x in szstds];
    noDstds=[sum([i^2 for i in x-mean_nostds]) for x in nostds];

    szdstds=[sum([i for i in x-mean_nostds]) for x in szstds];
    nodstds=[sum([i for i in x-mean_nostds]) for x in nostds];

    if i < 5
        idx = 8
    else 
        idx = 3
    end
    Pent=(median(szDents)+median(noDents))/idx
    Pstd=(median(szDstds)+median(noDstds))/idx
    Pentrng=(median(szentrng)+median(noentrng))/idx
    Pstdrng=(median(szstdrng)+median(nostdrng))/idx
    Pdent=(median(szdents)+median(nodents))/idx
    Pdstd=(median(szdstds)+median(nodstds))/idx

    for j in 1:nfiles[i]
        this_ents=getFileEntropies(string(dir,"/",testfiles[j]))
        this_stds=getFileStds(string(dir,"/",testfiles[j]))
        this_Dents=sum([i^2 for i in this_ents-mean_noents])
        this_Dstds=sum([i^2 for i in this_stds-mean_nostds])
        this_entrng=this_ents[1]-this_ents[16]
        this_stdrng=this_stds[1]-this_stds[16]
        this_dents=sum([i for i in this_ents-mean_noents])
        this_dstds=sum([i for i in this_stds-mean_nostds])
        
        if (this_Dstds > Pstd && this_Dents > Pent && this_entrng > Pentrng && 
                this_stdrng > Pstdrng  && this_dstds > Pdstd && this_dents > Pdent)
            print(string(testfiles[j],",","1,1\n"))
            tf = convert(DataArray{Bool, 1}, [utf8(testfiles[j])==k for k in dfSS[1]]);
            dfSS[tf,2]=1;
        else
            print(string(testfiles[j],",","0,0\n"))
        end
    end
end



