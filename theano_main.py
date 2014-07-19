from data_set import TrainingSet, TestSet
from theano_analysis import Analysis

print "Reading dataset...."
xs = TrainingSet("../data/training_sub_set.csv")
#xs = TrainingSet("../data/training.csv")

print "Building model...."
analysis = Analysis(xs)
#analysis.train(learning_rate=0.004, L1_reg=0.00, L2_reg=0.0005, 
#			   n_epochs=50, batch_size=50, n_hidden=250)

analysis.train_SdA()

scores = analysis.getTestScores()  
print ("Training completed, test scores: " , scores.shape)
print scores

analysis.plotErrors()

amss = analysis.calculateAMS(scores)
analysis.plotAMSvsRank(amss)
analysis.plotAMSvsScore(scores, amss)

xsTest = TestSet("../data/test.csv")
analysis.computeSubmission(xsTest, "submission.csv")