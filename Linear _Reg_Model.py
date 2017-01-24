import sklearn
import scipy.stats as stats
from sklearn.datasets import load_boston
import pandas as pd 
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pylab as plt


df=pd.read_csv('03_02_force_35_010.txt',sep=',',skiprows=13,header=None,na_values="null",delimiter=',')

df.columns=['Force_sensor','EMG_radial_1','EMG_radial_2','EMG_radial_3','EMG_radial_4',
            'EMG_radial_5','EMG_radial_6','EMG_special_1','EMG_special_2','EMG_special_3','EMG_special_4']

# Droping the force sensor column 
X=df.drop('Force_sensor',axis=1)

# Creat the Linear Regression Model 
lm=LinearRegression()

# Fit the Model 
lm.fit(X, df.Force_sensor)

print('Estimated intercept coefficeint :',lm.intercept_)
print('Number of Coefficient :', len(lm.coef_))
print('THE Coffecient is :',lm.coef_)

s=pd.Series(lm.coef_,index=X.columns)

print(s)
plt.scatter(df.EMG_radial_1,df.Force_sensor)
plt.xlabel('THE FIRST CHANNEL FROM THE EMG')
plt.ylabel('FORCE SENSOR VALUES')
plt.title('THE RELATIONSHIP BETWEEN FORCE SENSOR AND THE FIRST EMG CHANNEL')
plt.show()


#print(lm.predict(X)[0:10])

plt.scatter(df.Force_sensor,lm.predict(X))
plt.xlabel('Measured Force')
plt.ylabel('Predicted Force')
plt.title('Measured Force Vs Expected Force')

plt.show()


# Caluclate the mean squared Error :

mseFull=np.mean((df.Force_sensor)-lm.predict(X)**2)
print('the mean squared Error is ',(mseFull))

