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
    import matplotlib.pyplot as plt 
except:
    os.system('pip install matplotlib')
    import matplotlib.pyplot as plt 
from matplotlib import style
try:
    from collections import Counter
except:
    os.system('pip install collections')
try:
    import warnings
except:
    os.system('pip install warnings')
    import warnings
from collections import Counter
style.use('fivethirtyeight')

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
            euclidean_distance = sqrt( ( ( features[0] - predict[0]) **2 )  + ( ( features[1] - predict[1] ) **2 ) )
            print("#-------------------------------------#")
            print("group:", group, "data[group]", data[group], "euclidean_distance", euclidean_distance)
            print("#-------------------------------------#")
            #euclidean_distance = np.linalg.norm(np.array(features) - np.array(predict))
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

    return vote_result

#######################################
# MAIN LOGIC STARTS HERE
#######################################

dataset = {'k': [[1,2,], [2,3], [3,1]], 'r':[[6,5], [7,7,], [8,6]]}
new_features = [5,7]

for i in dataset:
    for ii in dataset[i]:
            plt.scatter(ii[0], ii[1], s=100, color = [i])
plt.scatter(new_features[0], new_features[1], s=200, color = 'b')
plt.legend([new_features], title = 'BEFORE KNN\nblue dot is new_features ', fontsize = 10)
plt.show()


result = k_nearest_neighbors(dataset, new_features, k =3)
print("#######################################")
print("k_nearest_neighbors result:", result)
print("#######################################")


for i in dataset:
    for ii in dataset[i]:
            plt.scatter(ii[0], ii[1], s=100, color = [i])
plt.scatter(new_features[0], new_features[1], s=200, color = result)
plt.legend([new_features], title = 'AFTER KNN]\nnotice new dot color\nshowing group association ', fontsize = 10)
plt.show()


