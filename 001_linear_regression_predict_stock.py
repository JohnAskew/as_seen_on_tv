import os, sys
try:
    import pickle
except:
    os.system('pip install pickle')
    import pickle
try:
    import datetime
except:
    os.system('pip install datetime')
    import datetime
try:
    import math
except:
    os.system('pip install math')
    import math
try:
    import matplotlib.pyplot as plt 
except:
    os.system('pip install matplotlib')
    import matplotlib.pyplot as plt 
from matplotlib import style
try:
    import numpy as np 
except:
    os.system('pip install numpy')
    import numpy as np 
try:
    import pandas as pd 
except:
    os.system('pip install pandas')
    import pandas as pd 
try:
    import pickle
except:
    os.system("pip install pickle")
    import pickle
try:
    import quandl
except:
    os.system('pip install quandl')
    import quandl
try:
    from sklearn import preprocessing, svm
except:
    os.system('pip install sklearn')
    from sklearn import preprocessing, svm
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_validate, train_test_split

style.use('ggplot')




try:
    if os.path.exists('000-wiki-googl.pickle'):
        with open("000-wiki-googl.pickle", 'rb') as in_file:
            df = pickle.load(in_file)
            print("Loading 000-wiki-googl.pickle")
    else:
         df = quandl.get('WIKI/GOOG')
         with open("000-wiki-googl.pickle", "wb") as in_file:    #Pickle saves results as reuable object
             pickle.dump(df, in_file)  
except:
    print("Unable to load nor access needed inputs....Aborting")
    sys.exit()

#######################################
# M A I N   L O G I C
#######################################


df = df [['Adj. Open', 'Adj. High', 'Adj. Low','Adj. Close','Adj. Volume',]]
df['HL_PCT'] = (df['Adj. High'] - df['Adj. Close'] ) / df['Adj. Close'] * 100.0 # Create new column
df['PCT_change'] = (df['Adj. Close'] - df['Adj. Open'] ) / df['Adj. Open'] * 100.0 # Create new column

df = df[['Adj. Close','HL_PCT','PCT_change','Adj. Volume']] # Reduce DataFrame's number of columns to just these

forecast_col = 'Adj. Close' # set alias - will be referenced down in code
df.fillna(-99999, inplace = True) # Machine Learning does not work with nan (nulls). Replace with marker of -99999

forecast_out = int(math.ceil(0.089 * len(df))) # Don't make 90 days too complicated..which we did
print("The forecasted days in advance (forecast_out) is: ",forecast_out)

df['label'] = df[forecast_col].shift(-forecast_out) # shifting moves up 343 rows
#-------------------------------------#
# Convert DataFrame to np.array
#-------------------------------------#
X = np.array(df.drop(['label'], 1)) # Drop the column "label". Will be picked up by variable "y'"
X = preprocessing.scale(X) # Scale to normalize - must include training data features
X_lately = X[-forecast_out:] # Save this off for predictions - used later down in code
X = X[:-forecast_out] # Train with this


#X = X[:-forecast_out + 1]
df.dropna(inplace = True)
y = np.array(df['label'])

#######################################
# Split data between training data and testing data
#######################################
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2) # Save off 20% data for testing.



#######################################
# SVM - Support Vector Machine
#######################################
clf = svm.SVR(kernel = 'linear') # Model is LinearRegression
clf.fit(X_train, y_train) # fit = train model using parameters
accuracy = clf.score(X_test, y_test) # Classifier score: Confidence of predictions using test data (from above)
print("svm.SVR\taccuracy:", accuracy)

#######################################
# Linear Regression
#######################################
clf = LinearRegression(n_jobs = -1) # Model is LinearRegression
clf.fit(X_train, y_train) # fit = train model using parameters
#
# Save off trained dataset
#
with open('001-linearregression_pickle', 'wb') as f:
    pickle.dump(clf, f)
#
## Process training score
#
accuracy = clf.score(X_test, y_test) # Classifier score: Confidence of predictions using test data (from above)
print("LinRegr\taccuracy:", accuracy)
forecast_set = clf.predict(X_lately) #Predict next 31 days from last entry in X_train.

df['Forecast'] = np.nan

#######################################
# End Modeling; Set up dates for Graphing using indices (see loc function)
#######################################
last_date = df.iloc[-1].name # last record - date value
last_unix = last_date.timestamp()
one_day = 86400 # Number of seconds in 24 hours (a full day)
next_unix  = last_unix + one_day # Bring next_unix current to today as point in time

#######################################
# From above: forecast_set = clf.predict(X_lately)
# From above: X_lately = X_lately = X[-forecast_out:] 
# From above: forecast_out (31) = forecast_out = int(math.ceil(0.009 * len(df)))
#
# Here is what your data will look like:
# 
'''                 Adj. Close  HL_PCT  ...  label     Forecast
Date                                     ...                    
2018-02-09 19:00:00         NaN     NaN  ...    NaN  1078.225051
2018-02-10 19:00:00         NaN     NaN  ...    NaN  1073.834164
2018-02-11 19:00:00         NaN     NaN  ...    NaN  1071.452728
2018-02-12 19:00:00         NaN     NaN  ...    NaN  1089.996808
2018-02-13 19:00:00         NaN     NaN  ...    NaN  1108.473414'''
#######################################

for i in forecast_set: 
    next_date = datetime.datetime.fromtimestamp(next_unix)
    next_unix += one_day
    #print("forecast_set iterator i:", i)
    df.loc[next_date] = [np.nan for _ in range(len(df.columns) -1)] + [i]
    #print("next_date", next_date, "df.loc[next_date]", df.loc[next_date])


df['Adj. Close'].plot()
df['Forecast'].plot()
plt.legend(loc = 4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()





