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

X = np.array( [ [1, 2],
                [1.5, 1.8],
                [5, 8],
                [8, 8],
                [1, .6],
                [9, 11],
                [8, 2],
                [10, 2],
                [9, 3],
                [11,4],
                [2, 1]
              ]
            )

# colors =int((len(X)/3) + int(len(X)%3)) * ('g', 'r', 'm','bisque', 'forestgreen', 'slategrey' )
color_list = []
colors = ('g', 'r', 'olivedrab', 'salmon', 'hotpink' , 'thistle',
          'dimgray', 'darkorange', 'turquoise' , 'lightsteelblue',
          'burlywood', 'royalblue','midnightblue', 'darkgoldenrod',
          'lightseagreen', 'mediumslateblue', 'rosybrown' , 'limegreen',
          'coral', 'yellow', 'powderblue', 'fuschia',
          'orangered', 'orchid', 'deepskyblue','forestgreen')
for cnt, color in enumerate(colors):
    if cnt < int(len(X)):
        color_list.append(colors[cnt])
       


class Mean_Shift:
    def __init__(self, radius = 4):
        self.radius = radius

    def fit(self, data):
        points = {} # using the dict command to set points[i] = data[i] and then prev_points = dict(points)

        for i in range(len(data)):
            points[i] = data[i]
        
        while True:
            new_centroids = []
            for i in points:
                this_group = []
                point = points[i]
                for featureset in data: # featureset is the index to X. First entry is featureset: [1 2]. Data is whole array
                    if np.linalg.norm(featureset - point) < self.radius:
                        this_group.append(featureset) # Save any point with eud. dist. < radius (set to 4 by default)

                new_centroid = np.average(this_group, axis = 0) # Avg. of all stored points in this_group
                new_centroids.append(tuple(new_centroid))
            uniques = sorted(list(set(new_centroids)))
            prev_points = dict(points)
                       
            points = {}
            for i in range(len(uniques)):
                points[i] = np.array(uniques[i])
               
            optimized = True
            for i in points:
                if not np.array_equal(points[i], prev_points[i]):
                    optimized = False
                    
                if not optimized:
                    break

            if optimized:
                break

        self.points = points

    def predict(self, data):
        pass
#######################################
# MAIN LOGIC
#######################################
clf = Mean_Shift() #Instantiate our model or classifer as the Mean_Shift

clf.fit(X) #Call the fit code, listed above, which constitutes the majority of the code.

points = clf.points # Define points (outside the class) as poiting to the class' self.points = points

plt.scatter(X[:, 0], X[:,1], s = 100, color = color_list)
plt.title("Raw Data before calculating KMeans")
plt.show() 


plt.scatter(X[:, 0], X[:,1], s = 100, color = color_list)


for c in points:
    plt.scatter(points[c][0], points[c][1], color = 'k', marker = '*', s=250)
    plt.annotate('center', (points[c][0]+.10, points[c][1] +.15), size=8)
    plt.annotate((round(points[c][0],2),round(points[c][1],2)), (points[c][0]+ .15, points[c][1] -.18), size=8)
    circlex = plt.Circle((points[c][0] , points[c][1]), 2.9, color = 'gold', alpha = .1)
    plt.gcf().gca().add_artist(circlex) # gcf = get current figure; gca = get current axes (ax). Replaces fig, ax = plt.figure, ax = ???

plt.title("KMeans clustering into groups")
plt.show()









