import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import numpy as np
datas = []
file = open("preds.txt","r")
for line in file.read().split("\n"):
    data=float(line)
    datas.append(data)

plt.plot(figsize=(8,5),)


# create dataframe

dataframe = pd.DataFrame({'time': np.array([i for i in range(len(datas))]),
                          'classes': datas})

# Plotting the time series of given dataframe
from scipy.interpolate import make_interp_spline

X_Y_Spline = make_interp_spline(dataframe.time, dataframe.classes)
X_ = np.linspace(dataframe.time.min(), dataframe.time.max(), 500)
Y_ = X_Y_Spline(X_)
plt.plot(X_, Y_)

# Giving title to the chart using plt.title
plt.title('Confidence with time')

# rotating the x-axis tick labels at 30degree
# towards right
plt.xticks(rotation=30, ha='right')
plt.ylim(0,5)

# Providing x and y label to the chart
plt.xlabel('Time')
plt.ylabel('Confidence')
plt.show()