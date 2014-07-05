from training_set import TrainingSet
from theano_analysis import Analysis

xs = TrainingSet("../data/training_sub_set.csv")

analysis = Analysis(xs)

analysis.train(learning_rate=0.001, L1_reg=0.00, L2_reg=0.0005, n_epochs=10, batch_size=50, n_hidden=100)
    
print analysis.testScores 

analysis.plot()

analysis.calculateAMS()
analysis.plotAMSvsRank()
analysis.plotAMSvsScore()

xsTest = TrainingSet("../data/test.csv")
analysis.computeSubmission(xsTest, "submission.csv")