
'''
Features calulation for EMG channels 
return Data Frame of 8 Features with the label of the channel and the File name 
for further PCA 

'''

import numpy as np
import pandas as pd 
import matplotlib.pylab as plt
import scipy
from scipy import signal
from itertools import groupby
import os


def EMG_Feature(inFile):
    
    #Read the data from txt file and returning them as a labeled DataFrame  
        
    #dataDir ="C:/Users/Hasan_Issa/Downloads/WinPython-64bit-3.4.3.5/python-3.4.3.amd64/MY_Examples/Project_Files/DATA RMS"
    
    dataDir = "C:/Users/Hasan_Issa/Documents/University/Project's Documents/Hasan/02_rawData"
    
    os.chdir(dataDir)
    
    df = pd.read_csv(inFile,sep=',',skiprows=13,index_col=None,na_values="null",header=None,skipfooter=1,engine='python')
    
    sensors = ['Force_sensor','EMG_radial_1','EMG_radial_2','EMG_radial_3','EMG_radial_4',
                'EMG_radial_5' ,'EMG_radial_6','EMG_special_1','EMG_special_2','EMG_special_3','EMG_special_4']    
    
    df.columns= sensors 
    
    fs =10e3 # Sampling Frequency 
    
    df.drop('Force_sensor',axis = 1,inplace=True)
    
    # List of features to be appended by iterating through the EMG Channels 
    
    MAV = [] # Mean Absoulte Value 
    WAV = [] # Wave Form Length
    ZC = [] # Zero Crossing 
    Mean_Power = [] # Mean Power of the singal 
    Peak_Freq  = [] # Peak Frequency of the signal 
    Total_Power = [] # Total Power of the signal 
    SM1 = [] # First Spectral Moment 
    SM2 = [] # Second Spectral Moment 
    new1 = [] # empty list to append the SM1 
    new2 = []  # empty list to append the SM2 
    channel = []
    max_feature = []  # empty list to store the maximum feature 
    
    min_feature = []  # empy list to stort the minimum feature 
    
    for name, values in df.iteritems():
        
        # Mean Absoulute value 
        Mean_Abs_Val = (1/len(values)) * (sum(abs(values[0:len(values)])))
        MAV.append(Mean_Abs_Val)  
        
        # waveform lenght 
        Wav_Leng = abs(np.diff((values)))
        diff_sum = sum((Wav_Leng))
        WAV.append(diff_sum)        

        # Zero Crossing(ZC)
        Zero_crossing = np.sign(values)
        Zero_Cros = np.where (Zero_crossing > -1, Zero_crossing, 0)
        Zero_sum = sum(Zero_Cros)
        ZC.append(Zero_sum)
        
        # get the power density and the frequenices 
        f, Pxx_den = signal.periodogram(values, fs)
        
        mean_Power  =  (sum(Pxx_den)) / (len(f))
        Mean_Power.append(mean_Power)
        
        Peak = max(Pxx_den)
        Peak_Freq.append(Peak)
        
        Tot_pwr = sum(Pxx_den)
        Total_Power.append(Tot_pwr)
        
        for frequency,power in zip (f,Pxx_den):
            
            first_Spect = frequency* power
            second_Spect = (frequency)*(power**2)

            new1.append(first_Spect)
            new2.append(second_Spect)
        
        first_Spect = sum(new1)
        second_Spect = sum(new2)
            
        SM1.append(first_Spect)
        SM2.append(second_Spect)
        
        label  = r'{}'.format(inFile)
        
        column_label = r'{}'.format(name)
        channel.append(column_label)
    
    if label.endswith("precision",6,15):  
        movment_type = np.repeat('Precision', 10)
        
    if label.endswith("pinch",6,11):
        movment_type = np.repeat('Pinch', 10)
        
    if label.endswith("force",6,11):    
        movment_type = np.repeat('Force', 10)
        
    
        
        
    
      
    features = {'Mean Absolute value' : MAV,'Wave form length':MAV,'Zero Corssing' : ZC, 'Mean Power':Mean_Power,'Peak Frequency' : Peak_Freq,'Total Power':Total_Power,'First Spectral Moment':SM1,
                'Second Spectral Moment':SM2,'Channel label':channel,'File Name':movment_type}
    
    DataFrame = pd.DataFrame.from_dict(features)
    
    #print(DataFrame.head())
    #print(DataFrame.tail())
    
    
    return(DataFrame)
    
Data = EMG_Feature('05_02_force_20_100.txt')
print(Data.head())
print(Data.tail())