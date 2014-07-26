from data_set import TrainingSet, TestSet
from theano_analysis import Analysis

print "Reading dataset...."
#xs = TrainingSet("../data/training_sub_set.csv")
xs = TrainingSet("../data/training.csv")

print "Building model...."
analysis = Analysis(xs)
analysis.train_SdA()

#analysis.plotErrors()
#analysis.plotAMSvsRank(amss)
#analysis.plotAMSvsScore(scores, amss)

xsTest = TestSet("../data/test.csv")
analysis.computeSubmission(xsTest, "../data/submission.csv")