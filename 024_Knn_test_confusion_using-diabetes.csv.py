import os, sys
from askew_utils import DF_Magic as dfm
try:
    import numpy as np  
except:
    os.system('pip install numpy ')
    import numpy as np 
try:
    import panda as pd 
except:
    os.system('pip install pandas')
    import pandas as pd
try:
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.metrics import confusion_matrix, f1_score, accuracy_score
except:
    os.system('pip install sklearn')
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.metrics import confusion_matrix, f1_score, accuracy_score

dataset = dfm.get_df('diabetes.csv')#pd.read_csv('diabetes.csv')
print("length of dataset:", len(dataset))
print(dataset.sample(n=3))

#-------------------------------------#
# Set 0 values to mean of column, where mean calc skips NaN.
#-------------------------------------#
zero_not_accepted = ['glucose','bloodpressure', 'skinthickness', 'bmi', 'insulin']
for column in zero_not_accepted:
    dataset[column] = dataset[column].replace(0, np.NaN)
    mean = int(dataset[column].mean(skipna  = True))
    dataset[column] = dataset[column].replace(np.NaN, mean)

#-------------------------------------#
# Save off last column as 'y' and then split datasets
#-------------------------------------#
X = dataset.iloc[:, 0:8] # all rows, columns 0-8
y = dataset.iloc[:, 8] #all rows, just column 8
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 42, test_size=0.20)
print("Length of X_train:", len(X_train))
print("Length of X_test:", len(X_test))
print("Length of y_train:", len(y_train))
print("Length of y_test:", len(y_test))

sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.fit_transform(X_test)

clf = KNeighborsClassifier(n_neighbors =30, p = 2, metric = 'euclidean')# n_jobs = 1)# leaf_size = 30, weights = 'uniform')
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)
print("f1_score:", f1_score(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))











