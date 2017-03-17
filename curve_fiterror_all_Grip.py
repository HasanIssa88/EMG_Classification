
'''
Curve_Fit_Force : is used to fit the Median data for each Contraction above certain threshold to a curve used to represent the data

in our case the curve is a linear curve

the function is reading two data frame one represent the Median Force values normalized to the maximum and another is not normalized 


author : Hasan Issa 

Data : Feb 2017 

rev : 0.1 


'''

# Import the required Libraries 

import pandas as pd 
import numpy as np 
import itertools
import statistics
import matplotlib.pylab as plt 
import math
from itertools import groupby
from statistics import median , stdev
from scipy.optimize import curve_fit
from matplotlib.pylab import rc


# reading the data frame for Pinch Movment 
def Force_reg (name):
    
            
        df = pd.read_csv(name,sep = ',' , index_col=0,skipfooter=1 )
        #df2 = pd.read_csv('Force_Median.txt',sep = ',',index_col= 0  )
        
        
        # convert the data frame into list 
        
        # Normalize data 
        Normalized  = df.values.tolist()
        Normalized = list(itertools.chain.from_iterable(Normalized))
        
        # Not Normalized data 
        #Not_Normalized  = df2.values.tolist()
        #Not_Normalized = list(itertools.chain.from_iterable(Not_Normalized))
        
        y = Normalized
        
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
        
        #popt, pcov = curve_fit(line, x, y, sigma=,full_output = True)
        
        popt, pcov, infodict, errmsg, ier = curve_fit(line, x, y, full_output = True)
        # popt is the  Optimal values for the parameters so that the sum of the squared error of f(xdata, *popt) - ydata is minimized
        # pcov is The estimated covariance of popt.
        
        print ("a =", popt[0], "+/-", pcov[0,0]**0.5)
        print ("b =", popt[1], "+/-", pcov[1,1]**0.5)
        
        sigma = np.sqrt([pcov[0,0],pcov[1,1]])
        
        values = np.array([
                          line(x, popt[0] + sigma[0], popt[1] + sigma[1]),
                          line(x, popt[0] + sigma[0], popt[1] - sigma[1]),
                          line(x, popt[0] - sigma[0], popt[1] - sigma[1]),
                          line(x, popt[0] - sigma[0], popt[1] + sigma[1],)
                          ])
        
        # the fit error represents the standard deviation of all the possible fit +- uncertainty
        # values at each x position. One could imagine getting the min and max possible deviations
          
        fitError = np.std(values, axis = 0 )
        print(fitError)
        N = len(y)
        dx = (max(x) - min(x))/N
        nSigma = 2 
        
        #fitEquation = r"$\displaystyle\mathrm{fit} =  a e^{-b t} + c$"
        fitEquation = " ax + b "
        
        
        rc('text', usetex=False)
        rc('font', family='serif')
        plt.rc('xtick', labelsize=20) 
        plt.rc('ytick', labelsize=20)
        plt.rcParams['xtick.major.pad'] = 20
        plt.rcParams['ytick.major.pad'] = 20
        
        fig = plt.figure(figsize=(12,8), dpi=100, facecolor='w', edgecolor='k')
        ax = fig.add_subplot(111)
        
        ax.xaxis.labelpad = 20
        ax.yaxis.labelpad = 20
        curveFit = line(x,popt[0], popt[1])
        plt.plot(x, y, 'o')
        plt.hold(True)
        
        plt.plot(x, curveFit, 
            linewidth=2.5, 
            color = 'green',
            alpha = 0.6,
            label = fitEquation)
        
        plt.bar(left=x, 
            height = 2*nSigma*fitError,  
            width=dx, 
            bottom = curveFit - nSigma*fitError, 
            orientation = 'vertical', 
            alpha=0.04, 
            color = 'purple',
            edgecolor = None,
            label = r'$\pm 3\sigma\;\mathrm {error bars} $')
        
        plt.plot(x, curveFit+fitError, 
            linewidth = 1.0, 
            alpha = 0.5, 
            color = 'red',
            label = r'$\pm 1\sigma  error $')
        
        plt.plot(x, curveFit-fitError, 
            linewidth = 1.0, 
            alpha = 0.5, 
            color = 'red')
        
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.2)
        
        plt.xlabel('time', fontsize=24)
        
        plt.ylabel('MVC',fontsize=24)
        
        plt.title('linear fit with $\pm 1\sigma$ and $\pm 3\sigma$ fit errors ',
              fontsize=28, color='k')
        
        ax.legend(fontsize=18,loc = 2)
        
        plt.gca().xaxis.set_major_locator(plt.MaxNLocator(prune='lower'))
        
        plt.savefig('Force_Median_error.jpg', figsize=(6,4), dpi=300)
        
        plt.show()
        
        
Norm_names = ['Precesion_Median_Normalized.txt','Pinch_Median_Normalized.txt','Force_Median_Normalized.txt']
names = ['Precesion_Median.txt','Pinch_Median.txt','Force_Median.txt']

Force_reg('Force_Median_Normalized.txt')











