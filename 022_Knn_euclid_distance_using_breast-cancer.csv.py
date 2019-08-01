import os, sys
try:
    import numpy as np 
except:
    os.system('pip install numpy')
    import numpy as np 
try:
    from math import sqrt
except:
    os.system('pip install math')
    from math import sqrt
try:
    import pandas as pd 
except:
    os.system('pip install pandas')
    import pandas as pd
try:
    from collections import Counter
except:
    os.system('pip install collections')
try:
    import warnings
except:
    os.system('pip install warnings')
    import warnings
try:
    import random
except:
    os.system('pip install random')
    import random
from collections import Counter


#-------------------------------------#
def k_nearest_neighbors(data, predict, k = 3):
#-------------------------------------#
    if len(data) >= k:
        print("#######################################")
        print("K is set to a value less than total voting groups")
        print("#######################################")
    distances = []
    for group in data:
        for features in data[group]:
            # euclidean_distance = sqrt( ( ( features[0] - predict[0]) **2 )  + ( ( features[1] - predict[1] ) **2 ) )
            # print("#-------------------------------------#")
            # print("group:", group, "data[group]", data[group], "euclidean_distance", euclidean_distance)
            # print("#-------------------------------------#")
            euclidean_distance = np.linalg.norm(np.array(features) - np.array(predict)) # Use linalg package for euclid distance
            distances.append([euclidean_distance, group])


    votes = [i[1] for i in sorted(distances)[:k]]
    print("#-------------------------------------#")
    print("Function k_nearest_neighbors--> Counter(votes)", Counter(votes))
    print("#-------------------------------------#")
    print("#-------------------------------------#")
    print("Function k_nearest_neighbors--> Counter(votes).most_common:", Counter(votes).most_common)
    print("#-------------------------------------#")
    print("#-------------------------------------#")
    print("Function k_nearest_neighbors--> Counter(votes).most_common(1):", Counter(votes).most_common(1))
    print("#-------------------------------------#")
    print("#-------------------------------------#")
    print("Function k_nearest_neighbors--> Counter(votes).most_common(1)[0][0]:", Counter(votes).most_common(1)[0][0])
    print("#-------------------------------------#")
    vote_result = Counter(votes).most_common(1)[0][0]
    confidence = Counter(votes).most_common(1)[0][1] / k

    return vote_result, confidence

#######################################
# MAIN LOGIC STARTS HERE
#######################################

df = pd.read_csv('breast-cancer.csv')
df.replace('?', -99999, inplace = True)
df.drop(['id'], axis = 1, inplace = True)
full_data  = df.astype(float).values.tolist() # The '?" in original spreadsheet made column a string, not number.'

print("#-------------------------------------#")
print("full_data BEFORE random shuffle")
print("#-------------------------------------#")
print(full_data[:5])
random.shuffle(full_data)
print("#-------------------------------------#")
print("full_data AFTER random shuffle")
print("#-------------------------------------#")
print(full_data[:5])

test_size = 0.2
train_set = {2:[], 4:[]}
test_set  = {2:[], 4:[]}
train_data = full_data[:-int(test_size * len(full_data))]
test_data = full_data[-int(test_size * len(full_data)) :]

for i in train_data:
    train_set[i[-1]].append(i[:-1])
for i in test_data:
    test_set[i[-1]].append(i[:-1])

correct = 0 
total = 0
 
for group in test_set:
    for data in test_set[group]:
        vote, confidence = k_nearest_neighbors(train_set, data, k = 5)
        if group == vote:
            correct += 1
        else:
            print(80 * '#')
            print("Not totally confident with, test_set[group]:", data,"confidence is:", confidence)
            print(80 * '#')
        total += 1
print(39 * '#')
print("Accuracy", correct/total, "")
print(39 * '#')

