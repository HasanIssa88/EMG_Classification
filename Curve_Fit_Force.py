
'''
Curve_Fit : is used to fit the Median data for each Contraction above certain threshold to a curve used to represent the data

in our case the curve is a linear curve 

author : Hasan Issa 

Data : Feb 2017 

rev : 0.1 


'''

# Import the required Libraries 


import numpy as np 
import itertools
import statistics
import matplotlib.pylab as plt 
import math
from itertools import groupby
from statistics import median , stdev
from scipy.optimize import curve_fit

# manual entry of the data imported for Force and for all Subject not Normalized 


#y=[0.0, 0.11912979622270106, 0.26613398197414401, 0.31457455659733552, 0.42634203803303766, 1.0]

#y=[5.7000888627846278, 11.606864154916295, 18.89572655348298, 21.297540359879839, 26.839272308904924, 55.282774970041537]

#y= [[5.7000888627846278, 11.606864154916295, 18.89572655348298, 21.297540359879839, 26.839272308904924, 55.282774970041537],
#[4.7775069977615896, 8.6767668685125532, 14.437425409734647, 20.411057361027339, 25.16200796226186, 37.016762982642575],
#[5.7185240975148695, 10.429083052331469, 15.447395043968912, 21.904082295625997, 26.758721199145072, 47.959452221647965],
#[4.4444791510207216, 8.6711879810491013, 14.315003402186765, 19.591497140564933, 24.098709643685432, 37.181329611664694],
#[5.2518749456940235, 8.7084284073982392, 14.923920520053088, 21.503794348720838, 28.473397749910301, 40.351232201510044],
#]

# manual entry of the data imported for Force and for all Subject Normalized 

y= [[0.0, 0.11912979622270106, 0.26613398197414401, 0.31457455659733552, 0.42634203803303766, 1.0],
[0.0, 0.12094757622755227, 0.29963217564646039, 0.48492280251744352, 0.63228819467979791, 1.0],
[0.0, 0.11151646433936575, 0.23031858859407361, 0.38317240924599838, 0.49809978227276486, 1.0],
[0.0, 0.12911165156555612, 0.30151111399773545, 0.46269014203898007, 0.60037023159246483, 1.0],
[0.0, 0.098479109931036118, 0.27556190000477548, 0.46302612565174101, 0.66159396125034275, 1.0]]

# combine all subjects data in one list 

y= list(itertools.chain.from_iterable(y))

# sort the data 
y = sorted(y)

# Generate the x axis 
x = np.linspace(0,100,len(y))

# Calculate the error 

Std =  np.exp ((-1 /np.std(y)))

# repeat the error for all points 
e = np.repeat(Std, len(y))

# defining the line function which describe the relationship between the data which is a line in our case 

def line(x,a,b):
    return(a * x + b)

# get the best fit of the data 

popt, pcov = curve_fit(line, x, y, sigma=e)

# popt is the  Optimal values for the parameters so that the sum of the squared error of f(xdata, *popt) - ydata is minimized
# pcov is The estimated covariance of popt.

print ("a =", popt[0], "+/-", pcov[0,0]**0.5)
print ("b =", popt[1], "+/-", pcov[1,1]**0.5)

# plot the result 

plt.errorbar(x, y, yerr=e, fmt=None)
xfine = np.linspace(0., 100., len(y))  # define values to plot the function for
plt.plot(xfine, line(xfine, popt[0], popt[1]), 'r-')
plt.xlabel('Subject')
plt.ylabel('MVC_Level')

plt.show()

















#x = np.random.uniform(0,100,100)

#y = 3 * x +2 +np.random.normal(0,10,100)

#e =  np.repeat(10, 100)


#plt.errorbar(x,y,yerr=e, fmt=None)

#plt.show()



#def line (x,a,b):
    #return a * x + b 

#popt , pcov = curve_fit(line,x,y,sigma=e)

#print ("a =", popt[0], "+/-", pcov[0,0]**0.5)
#print ("b =", popt[1], "+/-", pcov[1,1]**0.5)
