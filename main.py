from data_set import TrainingSet, TestSet
from analysis import Analysis

print 'Loading training data....'
#xs = TrainingSet("../data/training_sub_set_1.csv")
xs = TrainingSet("../data/training.csv")

analysis = Analysis(xs)

print 'Evaluating....'
analysis.evaluate()

print 'Training...'
analysis.train()

print 'Saving....'
analysis.save()

#analysis.load()

print 'Computing submission...'
xsTest = TestSet("../data/test.csv")
analysis.computeSubmission(xsTest, "../data/submission.csv")