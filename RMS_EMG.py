
import numpy as np 
import matplotlib.pylab as plt 
import pandas as pd 
import os 
from numpy import sqrt,nanmean

def RMS_EMG(file_Name):
    
    def window_rms(a, window_size):
        a2 = np.power(a,2)
        window = np.ones(window_size)/float(window_size)
        RMS=np.sqrt(np.convolve(a2, window, 'valid'))
        return RMS
    

    df=pd.read_csv(file_Name,sep=',',skiprows=13,header=None,na_values="null",delimiter=',')
        
    df.columns=['Force_sensor','EMG_radial_1','EMG_radial_2','EMG_radial_3','EMG_radial_4',
                    'EMG_radial_5','EMG_radial_6','EMG_special_1','EMG_special_2','EMG_special_3','EMG_special_4']
    
    # first sensor Configuration  
    
    Force_sensor=df['Force_sensor']
    EMG_radial_1=df['EMG_radial_1']
    EMG_radial_2=df['EMG_radial_2']
    EMG_radial_3=df['EMG_radial_3']
    EMG_radial_4=df['EMG_radial_4']
    EMG_radial_5=df['EMG_radial_5']
    EMG_radial_6=df['EMG_radial_6']
    
    # second sensor Configuration 
    
    EMG_special_1=df['EMG_special_1']
    EMG_special_2=df['EMG_special_2']
    EMG_special_3=df['EMG_special_3']
    EMG_special_4=df['EMG_special_4']
    
    # calling the RMS Function for all the sensors 
    
    # First sensor Configuration
    
    Force_sensor=window_rms(Force_sensor,197)
    EMG_radial_1=window_rms(EMG_radial_1,197)
    EMG_radial_2=window_rms(EMG_radial_2,197)
    EMG_radial_3=window_rms(EMG_radial_3,197)
    EMG_radial_4=window_rms(EMG_radial_4,197)
    EMG_radial_5=window_rms(EMG_radial_5,197)
    EMG_radial_6=window_rms(EMG_radial_6,197)
    
    # second sensor Configuration 
    EMG_special_1=window_rms(EMG_special_1,197)
    EMG_special_2=window_rms(EMG_special_2,197)
    EMG_special_3=window_rms(EMG_special_3,197)
    EMG_special_4=window_rms(EMG_special_4,197)
    
    
    my_dict= dict({'Force_sensor':Force_sensor,'EMG_radial_1':EMG_radial_1,'EMG_radial_2':EMG_radial_2,'EMG_radial_3':EMG_radial_3,
                   'EMG_radial_4':EMG_radial_4,'EMG_radial_5':EMG_radial_5,'EMG_radial_6':EMG_radial_6,
                   'EMG_special_1':EMG_special_1,'EMG_special_2':EMG_special_2,'EMG_special_3':EMG_special_3,'EMG_special_4':EMG_special_4})
    
    #max_len =max(len(x) for x in my_dict.values())
    #df=pd.DataFrame({key:vals+[np.nan]*(max_len-len(vals)) for key,vals in my_dict.items()})
    
    
    df=pd.DataFrame(my_dict,columns=['Force_sensor','EMG_radial_1','EMG_radial_2','EMG_radial_3','EMG_radial_4','EMG_radial_5','EMG_radial_6',
            'EMG_special_1','EMG_special_2','EMG_special_3','EMG_special_4'])
    print(df.head(n=3))
 
    df.to_csv(os.path.join('C:/Users\Hasan_Issa/Downloads/WinPython-64bit-3.4.3.5/python-3.4.3.amd64/MY_Examples/Project_Files',r'{}.txt'.format(file_Name)),sep=',')
    return(df)
    
