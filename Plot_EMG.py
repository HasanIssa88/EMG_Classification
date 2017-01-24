
import os 
import glob
import pandas as pd
import numpy as np
import matplotlib.pylab as plt 


def plot_EMG(file_name):
    
    df=pd.read_csv(file_name,sep=',',skiprows=13,header=None,na_values="null",delimiter=',')
    
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
    
            # time vectors 
    
    
    Sampl_Freq=1000
    Sampl_Time=1/Sampl_Freq
    time_data_force =np.linspace(0,len(Force_sensor),num=len(Force_sensor))
    time_data_radial=np.linspace(0,len(EMG_radial_1),num=len(EMG_radial_1))
    time_data_special=np.linspace(0,len(EMG_special_1),num=len(EMG_special_1))
    
    # plot graphs
    
    
    
    f,(ax1,ax2,ax3,ax4,ax5,ax6,ax7)=plt.subplots(7,sharex=True)
    
    ax1.set_title('Raw data, Force and EMG configuration 1')
    
    ax1.plot(time_data_force, Force_sensor,'r')
    ax1.set_ylim(-10,10)
    ax1.set_ylabel('Force(Kg)')
    #ax1.set_yticklabels([])
    
    ax2.plot(time_data_radial, EMG_radial_1,)
    ax2.set_ylim(-0.1,0.1)
    ax2.set_yticklabels([])
    ax2.set_ylabel('Ch1[mV]')
    
    
    ax3.plot(time_data_radial,EMG_radial_2)
    ax3.set_ylim(-0.5,0.5)
    ax3.set_yticklabels([])
    ax3.set_ylabel('Ch2[mV]')
    
    
    ax4.plot(time_data_radial,EMG_radial_3)
    ax4.set_ylim(-0.5,0.5)
    ax4.set_yticklabels([])
    ax4.set_ylabel('Ch3[mV]')
    
    
    ax5.plot(time_data_radial,EMG_radial_4)
    ax5.set_ylim(-0.5,0.5)
    ax5.set_yticklabels([])
    ax5.set_ylabel('Ch4[mV]')
    
    
    ax6.plot(time_data_radial,EMG_radial_5)
    ax6.set_ylim(-1,1)
    ax6.set_yticklabels([])
    ax6.set_ylabel('Ch5[mV]')
    
    
    ax7.plot(time_data_radial,EMG_radial_6)
    ax7.set_ylim(-1,1)
    ax7.set_yticklabels([])
    ax7.set_ylabel('Ch6[mV]')
    ax7.set_xlabel('Time[ms]')
    
    
    plt.hold()
    
    f,(ax1,ax2,ax3,ax4,ax5)=plt.subplots(5,sharex=True)
    
    ax1.plot(time_data_force, Force_sensor,'r')
    ax1.set_ylim(-10,10)
    ax1.set_ylabel('Force(Kg)')
    #ax1.set_yticklabels([])
    ax1.set_xticklabels([])
    ax1.set_title('Raw data, Force and EMG configuration 2')
    
    ax2.plot(time_data_special, EMG_special_1,)
    ax2.set_ylim(-1,1)
    ax2.set_yticklabels([])
    ax2.set_ylabel('Ch7[mV]')
    ax2.set_xticklabels([])
    
    ax3.plot(time_data_special, EMG_special_2,)
    ax3.set_ylim(-1,1)
    ax3.set_yticklabels([])
    ax3.set_ylabel('Ch8[mV]')
    
    
    ax4.plot(time_data_special, EMG_special_3,)
    ax4.set_ylim(-0.2,0.2)
    ax4.set_yticklabels([])
    ax4.set_ylabel('Ch9[mV]')
    ax4.set_xticklabels([])
    
    ax5.plot(time_data_special, EMG_special_4,)
    ax5.set_ylim(-0.2,0.2)
    ax5.set_yticklabels([])
    ax5.set_ylabel('Ch10[mV]')
    ax5.set_xlabel('Time[ms]')
    
    plt.show()
    
    
    
    
    