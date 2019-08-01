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
from matplotlib import style
style.use('fivethirtyeight')


xs = np.array([1,2,3,4,5,6], dtype = np.float64)
ys = np.array([5,4,6,5,6,7], dtype = np.float64)

def best_fit_slope_and_intercept(xs, ys):
    m = ( ( (mean(xs) * mean(ys) ) - mean(xs*ys)) /
         ((mean(xs)**2) - mean(xs*xs)) )

    b = mean(ys) - m*mean(xs)

    return m, b

m, b = best_fit_slope_and_intercept(xs, ys)
print(m, b)

#regression_line = [(m*x + b for x in xs)]
regression_line = []
for x in xs:
    regression_line.append(m*x + b)

predict_x = 8
predict_y = (m*predict_x) + b

plt.scatter(xs, ys)
plt.scatter(predict_x, predict_y, color = 'g')
plt.plot(xs, regression_line, color = 'r')
plt.show()



