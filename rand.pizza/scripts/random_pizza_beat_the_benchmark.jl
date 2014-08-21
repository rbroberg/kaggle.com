# sentiment analysis on title and text
# extract words from title and text
# create dictionary
# find pizza prob for each word
# predict by summing probs

# score 0.53751

# caveat: I am just learning Julia and this is an exercise.

# rbroberg: 20140822

using DataFrames
using JSON

datadir="/projects/kaggle.com/rand.pizza/data/"
submitdir="/projects/kaggle.com/rand.pizza/submissions/"
libdir="/projects/kaggle.com/rand.pizza/scripts/"

include(libdir*"cleanString.jl")

# http://rosettacode.org/wiki/Flatten_a_list#Julia
function flat(A)
   result = {}
   grep(a) = for x in a 
               isa(x,Array) ? grep(x) : push!(result,x)
             end
   grep(A)
   result
end

jtrain=JSON.parsefile(datadir*"train.json");

# -----------------------------------------------------------------------------
# sentiment analysis
# -----------------------------------------------------------------------------

# extract words from title and text
mrow=size(jtrain)[1];
zas=[jtrain[i]["requester_received_pizza"]*1 for i in 1:mrow];
words=[split(cleanString(jtrain[i]["request_title"])*" "
        *cleanString(jtrain[i]["request_text"])) for i in 1:mrow];

wordbag=flat(words);
worduniq=Set(wordbag);
length(worduniq)
a=collect(worduniq);

# create dictionary
mrow=length(a)
wordfreq={i => 0 for i in a};
wordscore={i => 0 for i in a};
wordprobs={i => 0 for i in a};

[wordfreq[i]=wordfreq[i]+1 for i in wordbag];
mrow=size(words)[1];
[[wordscore[j]=wordscore[j]+zas[i] for j in words[i]] for i in 1:mrow];

# find pizza prob for each word
wordprobs={i => 0 for i in a};
[wordprobs[i]=wordscore[i]/wordfreq[i] for i in wordbag];

# test against training data		
mrow=length(words)
b1=[[try wordprobs[j] catch -1 end for j in words[i]] for i in 1:mrow];
b2=[mean(b) for b in b1];

mean(b2[zas.==1]) # 0.3168142155418048
mean(b2[zas.==0]) # 0.2750568619957863 
maximum(b2[zas.==0]) # 0.31237440944409317
minimum(b2[zas.==1]) # 0.27177955004277665
cutoff=.28;
bpreds=b2.>cutoff;

# precentage of correct predictions by sentiment in training data
sum([bpreds[i]==zas[i] for i in 1:mrow])/mrow # 0.696782

# predict test by summing probs
# but need to adjust for words not in sentiment dictionary
jtest=JSON.parsefile(datadir*"test.json");
mrow=size(jtest)[1];
cwords=[split(cleanString(jtest[i]["request_title"])*" "
        *cleanString(jtest[i]["request_text_edit_aware"])) for i in 1:mrow];
		
mrow=length(cwords);
c2=zeros(mrow)
for i in 1:mrow
	idx=c1[i].!=-1;
	c2[i]=sum(c1[i][idx])/sum(idx);
end

c2=[mean(c) for c in c1];
cpreds=c2.>cutoff;

# take a look ...
sum(zas)/length(zas) # .24604 for training data
sum(cpreds)/length(cpreds) # .31821 for predictions in test data

# create submission files
mrow=size(jtest)[1];
reqid=[jtest[i]["request_id"] for i in 1:mrow];
sarr=zeros(length(reqid),2);
sarr=convert(Array{Any,2},sarr);
sarr[:,1]=reqid;
sarr[:,2]=cpreds*1;
df=DataFrame(sarr);
names!(df,[symbol("request_id"),symbol("requester_received_pizza")]);
writetable(submitdir*"submit.sentiment.title.text.csv",
	df[:,[1,2]],separator=',',header=true);
