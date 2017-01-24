import sklearn
import scipy.stats as stats
from sklearn.datasets import load_boston
import pandas as pd 
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pylab as plt


df=pd.read_csv('02_02_precision_max_C_1.txt',sep=',',skiprows=13,header=None,na_values="null",delimiter=',')

df.columns=['Force_sensor','EMG_radial_1','EMG_radial_2','EMG_radial_3','EMG_radial_4',
            'EMG_radial_5','EMG_radial_6','EMG_special_1','EMG_special_2','EMG_special_3','EMG_special_4']


df2=pd.read_csv('02_01_precision_05_050.txt',sep=',',skiprows=13,header=None,na_values="null",delimiter=',')

df2.columns=['Force_sensor','EMG_radial_1','EMG_radial_2','EMG_radial_3','EMG_radial_4',
            'EMG_radial_5','EMG_radial_6','EMG_special_1','EMG_special_2','EMG_special_3','EMG_special_4']


# Droping the force sensor column 

X=df.drop(['Force_sensor','EMG_special_4','EMG_special_2','EMG_special_3','EMG_special_1'],inplace=False,axis=1,errors='ignore')


X_train, X_test,Y_train,Y_test=sklearn.cross_validation.train_test_split(X,df.Force_sensor,test_size=0.33,random_state=5)



print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)
print(Y_test.shape)

lm=LinearRegression()
lm.fit(X_train, Y_train)
pred_train=lm.predict(X_train)
pred_test=lm.predict(X_test)

print('Fit a model X_train, and calculate MSE with Y_train: :',np.mean(Y_train-lm.predict(X_train))**2)
print ('Fit a model X_train, and calculate MSE with X_test, Y_test ',np.mean(Y_test-lm.predict(X_test))**2)

plt.scatter(lm.predict(X_train),lm.predict(X_train)-Y_train,c='b',s=40,alpha=0.5)
plt.scatter(lm.predict(X_test),lm.predict(X_test)-Y_test,c='g',s=40)
plt.hlines(y=0,xmin=0,xmax=50)
plt.title('Residual plot using training (BLUE) and test(GREEN) data')
plt.ylabel('Residuals')
plt.show()
