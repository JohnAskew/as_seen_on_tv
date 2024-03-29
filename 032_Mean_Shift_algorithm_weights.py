import os, sys
try:
    import matplotlib.pyplot as plt 
except:
    os.system('pip install matplotlib')
    import matplotlib.pyplot as plt 
from matplotlib import style
style.use('ggplot')
try:
    import numpy as np 
except:
    os.system('pip install numpy')
    import numpy as np 
try:
    from sklearn.datasets.samples_generator import make_blobs
except:
    os.system('pip install sklearn')
    from sklearn.datasets.samples_generator import make_blobs
try:
    import random
except:
    os.system('pip install random')
    import random




#-------------------------------------#
class Mean_Shift:
#-------------------------------------#
    def __init__(self, radius = None, radius_norm_step = 100):
        self.radius = radius
        self.radius_norm_step = radius_norm_step

    def fit(self, data):

        if self.radius == None:
            all_data_centroid = np.average(data, axis = 0)
            all_data_norm = np.linalg.norm(all_data_centroid)
            self.radius = all_data_norm / self.radius_norm_step
            print("#-------------------------------------#")
            print("FIT setting self.radius to", self.radius)
            print("#-------------------------------------#")


        centroids = {} # Create centroids dictionary **

        for i in range(len(data)):
            centroids[i] = data[i]

        while True:
            new_centroids = []
            for i in centroids:
                in_bandwidth = []
                centroid = centroids[i]

                weights = [i for i in range(self.radius_norm_step)][::-1] # reverse the list


                for featureset in data:
                    distance = np.linalg.norm(featureset - centroid)
                    if distance == 0:
                        distance = 0.00000001
                    weight_index = int(distance/self.radius)
                    if weight_index > (self.radius_norm_step - 1):
                        weight_index = (self.radius_norm_step - 1)
                    to_add = (weights[weight_index] ) *  ( [featureset]) #**2 ) *  ( [featureset])

                    in_bandwidth += to_add




                    # if np.linalg.norm(featureset - centroid) < self.radius:
                    #     in_bandwidth.append(featureset)

                new_centroid = np.average(in_bandwidth, axis = 0)
                new_centroids.append(tuple(new_centroid)) # Simply convert to tuple
                
            uniques = sorted(list(set(new_centroids)))
            
            to_pop  = []

            for i in uniques:
                for ii in uniques:
                    if i == ii:
                        pass
                    elif np.linalg.norm(np.array(i) - np.array(ii)) <= self.radius -1:
                        to_pop.append(ii)
                        print("Adding to to_pop", ii)
                        break
            for i in to_pop:
                try:
                    uniques.remove(i)
                except:
                    print("POP failed to remove i:", i)

            prev_centroids = dict(centroids)
            
            centroids = {} # ** Redefining the original centroids dictionary
            for i in range(len(uniques)):
                centroids[i] = np.array(uniques[i])

            optimized = True

            for i in centroids:
                if not np.array_equal(centroids[i], prev_centroids[i]):
                    optimized = False
                if not optimized:
                    break

            if optimized:
                break

        self.centroids = centroids

        self.classifications = {}

        for i in range(len(self.centroids)):
            self.classifications[i] = []

        for featureset in data:
            distances = [np.linalg.norm(featureset - centroids[centroid]) for centroid in self.centroids]
            classification = distances.index(min(distances))
            self.classifications[classification].append(featureset)

    def predict(self, data):
        pass

#######################################
# MAIN LOGIC STARTS HERE
#######################################
centers = random.randrange(2,5)
X, y = make_blobs(n_samples = 20, centers = 3, n_features = 2)


# X = np.array( [ [1, 2],
#                 [1.5, 1.8],
#                 [5, 8],
#                 [8, 8],
#                 [1, .6],
#                 [9, 11],
#                 [8, 2],
#                 [10, 2],
#                 [9, 3]
#               ]
#             )
colors = 10 * ('g', 'r', 'c', 'b', 'k')
clf = Mean_Shift()
clf.fit(X)


centroids = clf.centroids

for classification in clf.classifications:
    color = colors[classification]
    for featureset in clf.classifications[classification]:
        plt.scatter(featureset[0], featureset[1], marker = 'x', color = color, s=150, linewidths = 5)

#plt.scatter(X[:, 0], X[:,1], s = 100, c = 'g')

for c in centroids:
    plt.scatter(centroids[c][0], centroids[c][1], color = 'k', marker = '*', s=250)
plt.show()









