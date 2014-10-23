cases=["Dog_1","Dog_2","Dog_3","Dog_4","Patient_1","Patient_2","Patient_3","Patient_4","Patient_5","Patient_6","Patient_7","Patient_8"];
nfiles=[3181,2997,4450,3013,2050,3894,1281,543,2986,2997,3601,1922]

for i in 1:length(cases)
    testfiles=[string(cases[i],"_test_segment_",string(j),".mat") for j in 1:nfiles[i]];
    for k in 1:nfiles[i]
            print(string(testfiles[k],",","0,0\n"))
    end
end

