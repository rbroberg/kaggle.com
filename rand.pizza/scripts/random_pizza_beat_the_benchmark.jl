# caveat: I am just learning Julia and this is an exercise.

# rbroberg: 20140821

using DataFrame
using DecisionTree
using JSON

datadir="/projects/kaggle.com/rand.pizza/data/"
submitdir="/projects/kaggle.com/rand.pizza/submissions/"

jtrain=JSON.parsefile(datadir*"train.json");