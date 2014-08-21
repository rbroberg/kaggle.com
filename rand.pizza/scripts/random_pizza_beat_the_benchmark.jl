# caveat: I am just learning Julia and this is an exercise.

# rbroberg: 20140821

#using DataFrames
#using DecisionTree
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

# this doesn't help much,
# just a 1D data frame of dictionaries
# dftrain=DataFrame(jtrain);
# dftrain

# -----------------------------------------------------------------------------
# extract days since first post
# -----------------------------------------------------------------------------
#=
nrow=size(jtrain)[1];
tarr=zeros((nrow,2));
[tarr[i,:]=[jtrain[i]["requester_days_since_first_post_on_raop_at_request"],
	jtrain[i]["requester_received_pizza"]*1] for i in 1:nrow];
dftrain=DataFrame(tarr);

mean(dftrain[dftrain[:,2].==0,1]) # 12.45
mean(dftrain[dftrain[:,2].==1,1]) # 28.56

median(dftrain[dftrain[:,2].==0,1]) # 0.0 
median(dftrain[dftrain[:,2].==1,1]) # 0.0
=#

# not good enough by itself

# -----------------------------------------------------------------------------
# sentiment analysis
# -----------------------------------------------------------------------------

# extract words from title and text
# create dictionary
# find pizza prob for each word
# predict by summing probs


# extract words from title and text
nrow=size(jtrain)[1];
zas=[jtrain[i]["requester_received_pizza"]*1 for i in 1:nrow];
words=[split(cleanString(jtrain[i]["request_title"])*" "
        *cleanString(jtrain[i]["request_text"])) for i in 1:nrow];

#TO-DO: clean up [Request], punctuation, trailing ...
wordbag=flat(words);
worduniq=Set(wordbag);
length(worduniq)
a=collect(worduniq);

# create dictionary
nrow=length(a)
wordfreq={i => 0 for i in a};
wordscore={i => 0 for i in a};
wordprobs={i => 0 for i in a};

[wordfreq[i]=wordfreq[i]+1 for i in wordbag];
nrow=size(words)[1];
[[wordscore[j]=wordscore[j]+zas[i] for j in words[i]] for i in 1:nrow];

# find pizza prob for each word
wordprobs={i => 0 for i in a};
[wordprobs[i]=wordscore[i]/wordfreq[i] for i in wordbag];

# predict by summing probs
jtest=JSON.parsefile(datadir*"test.json");
nrow=size(jtest)[1];
twords=[split(cleanString(jtest[i]["request_title"])*" "
        *cleanString(jtest[i]["request_text_edit_aware"])) for i in 1:nrow];

b=[[try wordprob[j] catch -1 end for j in twords[i]] for i in 1:nrow];


