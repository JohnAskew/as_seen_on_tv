import os, sys
try:
    import pandas as pd 
except:
    os.system("pip install pandas")
    import pandas as pd 
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
    import random
except:
    os.system('pip install random')
    import random
try:
    from sklearn import precrocessing, neighbors
except:
    os.system('pip install sklearn')
    from sklearn import preprocessing, neighbors
from sklearn.model_selection import cross_validate, train_test_split
from matplotlib import style
style.use('fivethirtyeight')

#
df = pd.read_csv('breast-cancer.csv')

df.replace('?', -99999, inplace = True)
df.drop(['id'], 1, inplace = True) # drop column named "id"

X = np.array(df.drop(['class'], 1))
y = np.array(df['class'])
# fig = df.hist(figsize = (20,16), color = 'g')
# [x.title.set_size(10) for x in fig.ravel()]
# plt.suptitle("Breast-Cancer Features")
# plt.show()

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)

clf = neighbors.KNeighborsClassifier()
clf.fit(X_train, y_train)
accuracy = clf.score(X_test, y_test)
print("#--------------------------------------#")
print("clf.accuracy", accuracy)
print("#--------------------------------------#")



example_measures = np.array([4,2,1,1,1,2,3,2,1]) # Ensure entry 4,2,1,1,1,2,3,2,1 does not appear in breast-cancer.csv
example_measures  = example_measures.reshape(1, -1) #Reshape starts with 1 and ends with -1
prediction = clf.predict(example_measures)
print("#--------------------------------------#")
print("# First prediction using single array:", prediction)
print("#--------------------------------------#")


example_measures = np.array(([4,2,1,1,1,2,3,2,1], [4,2,1,2,2,2,3,2,1]))
example_measures  = example_measures.reshape(len(example_measures), -1)
prediction = clf.predict(example_measures)
print("#--------------------------------------#")
print("# Second prediction using 2D array:", prediction)
print("#--------------------------------------#")

