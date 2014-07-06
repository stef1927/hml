from training_set import TrainingSet
from theano_analysis import Analysis

print "Reading dataset...."
xs = TrainingSet("../data/training_sub_set.csv")

print "Building model...."
analysis = Analysis(xs)
analysis.train(learning_rate=0.001, L1_reg=0.00, L2_reg=0.0005, n_epochs=10, batch_size=50, n_hidden=100)
    
print "Training completed, test scores:"  
scores = analysis.getTestScores()  
print scores

#analysis.plotErrors()

#amss = analysis.calculateAMS(scores)
#analysis.plotAMSvsRank(amss)
#analysis.plotAMSvsScore(scores, amss)

#xsTest = TrainingSet("../data/test.csv")
#analysis.computeSubmission(xsTest, "submission.csv")