import os, sys
from askew_utils import DF_Magic as dfm 
try:
    import csv
except:
    os.system('pip install csv' )
    import csv
try:
    import math
except:
    os.system('pip install math' )
    import math
try:
    import random
except:
    os.system('pip install random' )
    import random

dataset = dfm.get_df('pima-indians-diabetes.csv')
#dataset.fillna(-99999, inplace = True)
#-------------------------------------#
def loadDataset():
#-------------------------------------#
    lines = csv.reader(open(r'pima-indians-diabetes.csv'))
    dataset = list(lines)
    for i in range(len(dataset)):
        dataset[i] = [float(x) for x in dataset[i]]
    return dataset
#-------------------------------------#
def splitDataset(dataset, split):
#-------------------------------------#
    if split > 1:
        split = ( split * .01)

    testsize = int(len(dataset) * ( 1 - split) )
    train = list(dataset)
    test = []
    while len(test) < testsize:
        index = random.randrange(len(train))
        test.append(train.pop(index))
    return train, test
#-------------------------------------#
def separatedByClass(dataset):
#-------------------------------------#
    separated = {}
    for i in range(len(dataset)):
        vector = dataset[i]
        if vector[-1] not in separated:
            separated[vector[-1]] = []
        separated[vector[-1]] .append(vector)
    return separated
#-------------------------------------#
def mean(numbers):
#-------------------------------------#
    return  sum(numbers)/ float(len(numbers))
#-------------------------------------#
def stdev(numbers):
#-------------------------------------#
    avg = mean(numbers)
    variance = sum([pow(x - avg, 2) for x in numbers])/float(len(numbers) -1)
    print("stdev function: variance:", variance, "stdev:", math.sqrt(variance))
    return math.sqrt(variance)
#-------------------------------------#
def summarize(dataset):
#-------------------------------------#
    summaries = [(mean(attribute), stdev(attribute)) for attribute in zip(*dataset)]
    del summaries[-1]
    return summaries
#-------------------------------------#
def summarizeByClass(dataset):
#-------------------------------------#
    separated = separatedByClass(dataset)
    summaries = {}
    for classValue, instances in separated.items():
        summaries[classValue] = summarize(instances)
    return summaries
#-------------------------------------#
def calculateProbability(x, mean, stdev):
#-------------------------------------#
    exponent = math.exp(-(math.pow(x-mean,2)/(2*math.pow(stdev,2))))
    return (1/(math.sqrt(2*math.pi)*stdev))*exponent
#-------------------------------------#
def calculateClassProbabilities(summaries, inputVector):
#-------------------------------------#
    probabilities = {}
    for classValue, classSummaries in summaries.items():
        probabilities[classValue] = 1
        for i in range(len(classSummaries)):
            mean, stdev = classSummaries[i]
            x = inputVector[i]
            probabilities[classValue] *= calculateProbability(x, mean, stdev)
        return probabilities
#-------------------------------------#
def predict(summaries, inputVector):
#-------------------------------------#
    probabilities = calculateClassProbabilities(summaries, inputVector)
    bestLabel, bestProb = None, -1
    for classValue, probability in probabilities.items():
        if bestLabel is None or probability > bestProb:
            bestProb = probability
            bestLabel = classValue
    return bestLabel

#-------------------------------------#
def getPredictions(summaries, testSet):
#-------------------------------------#
    predictions = []
    for i in range(len(testSet)):
        result = predict(summaries, testSet[i])
        predictions.append(result)
    return predictions
#-------------------------------------#
def getAccuracy(testSet, predictions):
    correct = 0
    for x in range(len(testSet)):
        if testSet[x][-1] == predictions[x]:
            correct += 1
    return (correct/float(len(testSet)))*100.0

#######################################
# MAIN LOGIC
#######################################
dataset = loadDataset()
train, test = splitDataset(dataset, .66)
print("len of train:", len(train))
print("len of test:", len(test))
summaries = summarizeByClass(train)
predictions = getPredictions(summaries, test)
accuracy = getAccuracy(test, predictions)
print('Accuracy: {0}%'.format(accuracy))