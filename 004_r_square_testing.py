import os, sys
try:
    from statistics import mean
except:
    os.system('pip install statistics')
    from statistics import mean
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
from matplotlib import style
style.use('fivethirtyeight')

#
## Change variance for impacts to R2 or coefficient of determination. Smaller the variance, greater the R-squared value.
#

#-------------------------------------#
def create_dataset(hm, variance, step =2 , correlation = False):
#-------------------------------------#
    val = 1
    ys = []
    for i in range(hm):
        y = val + random.randrange(-variance, variance)
        ys.append(y)
        if correlation and correlation == 'pos':
            val += step
        elif correlction and correlation == 'neg':
            val -= step
    xs = [i for i in range(len(ys))]

    return np.array(xs, dtype = np.float64), np.array(ys, dtype = np.float64)
#-------------------------------------#
def best_fit_slope_and_intercept(xs, ys):
#-------------------------------------#
    m = ( ( (mean(xs) * mean(ys) ) - mean(xs*ys)) /
         ((mean(xs)**2) - mean(xs*xs)) )

    b = mean(ys) - (m * mean(xs))

    return m, b
#-------------------------------------#
def squared_error(ys_orig, ys_line):
#-------------------------------------#
    return sum( ( ys_line - ys_orig) ** 2)

def coefficient_of_determination(ys_orig, ys_line):
    y_mean_line = [mean(ys_orig) for y in ys_orig]
    squared_error_regr = squared_error(ys_orig, ys_line)
    squared_error_y_mean = squared_error(ys_orig, y_mean_line)
    return 1 - (squared_error_regr / squared_error_y_mean)

xs, ys = create_dataset(40, 20, 2, correlation = 'pos') # 2nd number is variance. Change to impact coefficient of determination.

m, b = best_fit_slope_and_intercept(xs, ys)

#regression_line = [(m*x + b for x in xs)] # generators not suppored with this version of python/matplotlib
regression_line = []
for x in xs:
    regression_line.append(m*x + b)

predict_x = 8
predict_y = (m*predict_x) + b

r_squared = coefficient_of_determination(ys, regression_line)
print("R-Squared:",r_squared)

plt.scatter(xs, ys)
plt.scatter(predict_x, predict_y, color = 'g', s = 100)
plt.plot(xs, regression_line, color = 'r')
plt.show()



