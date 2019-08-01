import os, sys
try:
    import csv
except:
    os.system('pip install csv')
    import csv
from askew_utils import DF_Magic as dfm
try:
    import math
except:
    os.system('pip install math')
    import math
try:
    import operator
except:
    os.system('pip install operator')
    import operator
try:
    import pandas as pd 
except:
    os.system('pip install pandas')
    import pandas as pd 
try:
    import random
except:
    os.system('pip install random')
    import random
try:
    dfm.get_df('iris.csv')
except:
    pd.read_csv('iris.csv')
#-------------------------------------#
def loadDataset(filename, split, object, train = [], test = []):
#-------------------------------------#
    with open(filename, 'r') as csvfile:
        lines = csv.reader(csvfile)
        dataset = list(lines)
        for x in range(1,len(dataset) -1):
            for y in range(4):
                dataset[x][y] = float(dataset[x][y])
            if random.random() < split:
                train.append(dataset[x])
            else:
                test.append(dataset[x])
    if object == 'df':
        train = pd.DataFrame(train)
        test = pd.DataFrame(test)
    return train, test
#-------------------------------------#
def euclideanDistance(instance1, instance2, length):
#-------------------------------------#
    distance = 0
    for x in range(length):
        distance += pow((instance1[x] - instance2[x]), 2) # square the difference between 2 points
    return math.sqrt(distance)
#-------------------------------------#
def getNeighbors(train, test, k):
#-------------------------------------#
    distances = []
    length = len(test) - 1
    for x in range(len(train)):
        dist = euclideanDistance(train[x], test, length) # This uses a singleton for test
        distances.append((train[x], dist))
    distances.sort(key = operator.itemgetter(1))
    neighbors = []
    for x in range(k):
        neighbors.append(distances[x][0])
    return neighbors
#-------------------------------------#
def getResponse(neighbors):
#-------------------------------------#
    classVotes = {}
    for x in range(len(neighbors)):
        response = neighbors[x][-1]
        if response in classVotes:
            classVotes[response] += 1
        else:
            classVotes[response] = 1
    sortedVotes = sorted(classVotes.items(), key = operator.itemgetter(1), reverse = True)
    return sortedVotes[0][0]
#-------------------------------------#
def getAccuracy(test, predictions):
#-------------------------------------#
    correct = 0
    for x in range(len(test)):
        if test[x][-1] in predictions[x]:
            correct += 1
    return (correct/float(len(test))) * 100

#--------------------------------------#
def main(k):
#--------------------------------------#
#Build lists for train and test which is a split of the csv file, skipping the header row
    train = []
    test = []
    split = .67
    train, test = loadDataset('iris.csv', .66, 'list') #Split train at 66% of data, 34% is for test
    print("train recs;", repr(len(train)))
    print("test recs;", repr(len(test)))
    predictions = []
 
    for x in range(len(test)):
        neighbors = getNeighbors(train, test[x], k)
        result = getResponse(neighbors)
        predictions.append(result)
        print("predicted >", repr(result), "actual:", repr(test[x][-1]))
    accuracy = getAccuracy(test, predictions)
    print("Model Accuracy using prediction", repr(accuracy))

#######################################
# MAIN LOGIC STARTS HERE
#######################################
#
# HOUSEKEEPING: Set up dummy data to test our functions
#                   prior to running main code.
#
#######################################
data1 = [2, 2, 2, 'a'] # Dummy data for testing our function
data2 = [4, 4, 4, 'b'] # Dummy data for testing our function
train = [[2, 2, 2, 'a'], [4, 4, 4, 'b']] #data1, data2]
test  = [5, 5, 5]
k = 1
neighbors = [[1, 1, 1, 'a'], [2, 2, 2, 'a'], [3, 3, 3, 'b']]
testSet = neighbors
predictions = ['a', 'a', 'a']
#--------------------------------------#
# Validate results before preceeding
#--------------------------------------#
distance = euclideanDistance(data1, data2, 3)
print("distance:", distance)
neighbors = getNeighbors(train, test, k)
print("neighbors for k of", k, neighbors)
response = getResponse(neighbors)
print("getResponse response:", response)
accuracy = getAccuracy(testSet, predictions)
print("Model Accuracy using prediction", predictions,":", accuracy)
#######################################
#
# Start Processing Data
#
#######################################
k = 3
main(k)





