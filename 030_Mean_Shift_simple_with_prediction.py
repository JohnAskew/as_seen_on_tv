try:
    from mpl_toolkits.mplot3d import Axes3D
except:
    os.system('pip install mpl_toolkits')
    from mpl_toolkits.mplot3d import Axes3D
try:
    import numpy as np 
except:
    os.system('pip install numpy')
    import numpy as np 
try:
    import matplotlib.pyplot as plt 
except:
    os.system('pip install matplotlib')
    import matplotlib.pyplot as plt 
try:
    from sklearn.cluster import MeanShift
    from sklearn.datasets.samples_generator import make_blobs
except:
    os.system('pip install sklearn')
    from sklearn.cluster import MeanShift
    from sklearn.datasets.samples_generator import make_blobs
from matplotlib import style
style.use('ggplot')

centers = [[1, 1, 1], [5, 5, 5], [3, 10, 10]]# Play with centers to change results, [7, 8, 9]]
X, _ = make_blobs(n_samples = 100, centers = centers, cluster_std = 1)
#-------------------------------------# Play with bandwidth 
ms = MeanShift(bandwidth = 3)         #   to change results
#-------------------------------------#
ms.fit(X)
prediction_vars = [] # Keep me for downstream code
prediction_vars = [[2, 2, 3], [5, 7, 4], [4, 9, 9]] # Can comment out to change results
#-------------------------------------# Play with predict
if len(prediction_vars) > 0:
    pv_predict = ms.predict(prediction_vars)  #   to change results

#-------------------------------------#
labels = ms.labels_
cluster_centers = ms.cluster_centers_
for j in range(len(centers)):
    print("#-------------------------------------#")
    print("Determining cluster center for ", centers[j], " to be:[", cluster_centers[j][0],
                                                           ",",
                                                           cluster_centers[j][1],
                                                           ","
                                                           ,cluster_centers[j][2],
                                                            "]"
        )
    print("#-------------------------------------#")

if len(prediction_vars) > 0:
    print("#-------------------------------------#")
    print("Prediction for:", prediction_vars, "is",pv_predict)
    print("#-------------------------------------#")
n_clusters = len(np.unique(labels))
print(39 * '#')
print("Number of estimated clusters:", n_clusters)
print(39 * '#')

colors = 10*['r','g','b','c','k','y','m']
fig = plt.figure()
ax = fig.add_subplot(1,1,1, projection = '3d')

if len(prediction_vars) > 0:
    if (pv_predict[0] != 0 ) or (pv_predict[1] != 1) or (pv_predict[2] != 2):
        plt.title("Prediction Skipped - not confident")
    else:
        for i in range(len(prediction_vars)):
            ax.scatter(prediction_vars[i][0], prediction_vars[i][1], prediction_vars[i][2], marker = '*', s = 200, c = colors[i])
            plt.title("Stars = prediction variable\ncolor shows cluster assignment")
            

for i in range(len(X)):
    ax.scatter(X[i][0], X[i][1], X[i][2], marker = 'o', c = colors[labels[i]])

ax.scatter(cluster_centers[:,0], cluster_centers[:,1],cluster_centers[:,2],marker = 'X', color = 'brown', s=100, linewidths = 5, zorder = 10)


plt.show()