'''
This function will read the EMG data after the RMS filter and will make thresholding for the force and the EMG Channels by deleting 
all values which are below a certain Thershold and will keep the values which is above and plot all channels configuration for this 
and will construct a DataFrame from all channels informaiton after deleting all values below the Thereshold 

'''

import os 
import numpy as np
import pandas as pd
from statistics import median
from itertools import groupby
import matplotlib.pylab as plt 



def EMG_Threshold(inFile,threshold = 0.3):
    
    '''Read the data from txt file and returning them as a labeled DataFrame  '''
    
    #dataDir ="C:/Users/Hasan_Issa/Downloads/WinPython-64bit-3.4.3.5/python-3.4.3.amd64/MY_Examples/Project_Files/DATA RMS"
    dataDir = "C:/Users/Hasan_Issa/Documents/University/Project's Documents/Hasan/02_rawData"
    os.chdir(dataDir)
    
    df = pd.read_csv(inFile,sep=',',skiprows=13,index_col=None,na_values="null",header=None)
    
    sensors = ['Force_sensor','EMG_radial_1','EMG_radial_2','EMG_radial_3','EMG_radial_4',
                'EMG_radial_5' ,'EMG_radial_6','EMG_special_1','EMG_special_2','EMG_special_3','EMG_special_4']    
    
    df.columns= sensors 
    
    Threshold = max(df.Force_sensor) * threshold
    
                                               
    # convert the Force sensor data to an array 
    Force_array = np.array(df.Force_sensor)
    
    # finding all force element which is beigger than threshold and assign zero for the rest 
    
    Force = np.where(Force_array > Threshold , Force_array, 0 )
    
    EMG_radial_1 = np.where(Force_array > Threshold, df.EMG_radial_1,0)  
    EMG_radial_2 = np.where(Force_array > Threshold ,df.EMG_radial_2,0)
    EMG_radial_3 = np.where(Force_array > Threshold ,df.EMG_radial_3,0)
    EMG_radial_4 = np.where(Force_array > Threshold ,df.EMG_radial_4,0)
    EMG_radial_5 = np.where(Force_array > Threshold ,df.EMG_radial_5,0)
    EMG_radial_6 = np.where(Force_array > Threshold ,df.EMG_radial_6,0)
    
    
    EMG_Special_1 = np.where(Force_array > Threshold, df.EMG_special_1,0)
    EMG_Special_2 = np.where(Force_array > Threshold, df.EMG_special_2,0)
    EMG_Special_3 = np.where(Force_array > Threshold, df.EMG_special_3,0)
    EMG_Special_4 = np.where(Force_array > Threshold, df.EMG_special_4,0)
    
    
       
    
    # plot the data 
    
    Sampl_Freq=1000
        
    time=np.array([i/Sampl_Freq for i in range(0,len(Force),1)])    
    
    f,(ax1,ax2,ax3,ax4,ax5,ax6,ax7)=plt.subplots(7,sharex=True)
        
    ax1.set_title('Raw data, Force and EMG configuration 1')
    
    ax1.plot(time, Force,'r')
    ax1.set_ylim(-1,8)
    ax1.set_ylabel('Force(Kg)')
    #ax1.set_yticklabels([])
    
    ax2.plot(time,EMG_radial_1)
    ax2.set_ylim(-0.1,0.1)
    ax2.set_yticklabels([])
    ax2.set_ylabel('Ch1[mV]')    
    
    ax3.plot(time,EMG_radial_2)
    ax3.set_ylim(-0.1,0.1)
    ax3.set_yticklabels([])
    ax3.set_ylabel('Ch2[mV]')   
    
    ax4.plot(time,EMG_radial_3)
    ax4.set_ylim(-0.1,0.1)
    ax4.set_yticklabels([])
    ax4.set_ylabel('Ch3[mV]') 
    
    ax5.plot(time,EMG_radial_4)
    ax5.set_ylim(-0.1,0.1)
    ax5.set_yticklabels([])
    ax5.set_ylabel('Ch4[mV]')  
    
    ax6.plot(time,EMG_radial_5)
    ax6.set_ylim(-0.1,0.3)
    ax6.set_yticklabels([])
    ax6.set_ylabel('Ch5[mV]')    
    
    
    ax7.plot(time,EMG_radial_6)
    ax7.set_ylim(-0.1,0.3)
    ax7.set_yticklabels([])
    ax7.set_ylabel('Ch6[mV]')    
    
    
    plt.show()
    
    # second Configuration 

    f,(ax1,ax2,ax3,ax4,ax5)=plt.subplots(5,sharex=True)

    ax1.plot(time, Force,'r')
    ax1.set_ylim(-1,8)
    ax1.set_ylabel('Force(Kg)')
    #ax1.set_yticklabels([])
    ax1.set_xticklabels([])
    ax1.set_title('Raw data, Force and EMG configuration 2')

    ax2.plot(time, EMG_Special_1,)
    ax2.set_ylim(-0.3,0.3)
    ax2.set_yticklabels([])
    ax2.set_ylabel('Ch1[mV]')
    ax2.set_xticklabels([])

    ax3.plot(time, EMG_Special_2,)
    ax3.set_ylim(-0.3,0.3)
    ax3.set_yticklabels([])
    ax3.set_ylabel('Ch2[mV]')


    ax4.plot(time , EMG_Special_3,)
    ax4.set_ylim(-0.1,0.1)
    ax4.set_yticklabels([])
    ax4.set_ylabel('Ch3[mV]')
    ax4.set_xticklabels([])

    ax5.plot(time, EMG_Special_4,)
    ax5.set_ylim(-0.1,0.1)
    ax5.set_yticklabels([])
    ax5.set_ylabel('Ch4[mV]')
    ax5.set_xlabel('Time[s]')


    plt.show()    

    #Construct the data frame 
    my_dict = {'Force':Force ,'EMG_radial_1':EMG_radial_1,'EMG_radial_2':EMG_radial_2,'EMG_radial_3':EMG_radial_1,'EMG_radial_4':EMG_radial_4,'EMG_radial_5':EMG_radial_5,'EMG_radial_6':EMG_radial_6,
               'EMG_speical_1':EMG_Special_1,'EMG_speical_2':EMG_Special_2,'EMG_speical_3':EMG_Special_3,'EMG_speical_4':EMG_Special_4}
    
    df = pd.DataFrame.from_dict(my_dict)
    
    pd.options.display.float_format = '{:,.2f}'.format
    #df = df.loc[~(df==0).all(axis=1)]
    print(df.shape)
    df.to_csv('inFile.txt', sep =',',index=None)
    
    return(df)
    
EMG_Threshold('03_02_force_10_100.txt')
#EMG_Threshold('06_01_precision_05_200.txt')