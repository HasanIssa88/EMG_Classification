# -*- coding: utf-8 -*-

import os 
import glob
import pandas as pd
import numpy as np
import matplotlib.pylab as plt 

'''
This function will take the path as an argument and it will print the EMG data segregated by either 
movment type, or 

'''

def spilt_Data(path):
    
    
    
    All_Files= glob.glob1(path,'*.txt') # this will read all files in the path which has the txt extenstion 
    
    #file_names=os.listdir("C:/Users\Hasan_Issa/Documents/University/Project's Documents/EMG_data/data_txt")
    
    # defintion of data based on movment type : 
    
    
    Pinch_allsubject=[]
    Force_allsubject=[]
    precision_allsubject=[]
    
    # defintion of data based on subject : 
    
    subject_1=[]
    subject_2=[]
    subject_3=[]
    subject_5=[]
    subject_4=[]
    
    # defintion of data based on Blcok type : 
    
    Block_1=[]
    Block_2=[]
    Block_3=[]
    
    # defintion of data based on Frequency type : 
    Frequency_1=[]
    Frequency_2=[]
    Frequency_3=[]
    Frequency_4=[]
    Frequency_5=[]
    
    
    
    # definition of data based on MVC level 
    MVC_05=[]
    MVC_10=[]
    MVC_20=[]
    MVC_35=[]
    MVC_50=[]
    MVC_66=[]
    MVC_Max=[]
    
    for file_index in All_Files:
    
        if file_index.endswith("02_",0,3):    
            subject_1.append(file_index)
            
        if file_index.endswith("03_",0,3):    
            subject_2.append(file_index)
            
        if file_index.endswith("04_",0,3):    
            subject_3.append(file_index)
            
        if file_index.endswith("05_",0,3):
            subject_4.append(file_index)
            
        if file_index.endswith("06_",0,3):    
            subject_5.append(file_index)
            
    # Segregation based on the movment type 
            
        if file_index.endswith("precision",6,15):    
            precision_allsubject.append(file_index)
            
        if file_index.endswith("pinch",6,11):    
            Pinch_allsubject.append(file_index)   
            
        if file_index.endswith("force",6,11):    
            Force_allsubject.append(file_index)
    # Segregation based on MVC level :
    
        if file_index.endswith("_05_",15,19) or file_index.endswith("_05_",11,15):
            MVC_05.append(file_index)
        if file_index.endswith("_10_",15,19) or file_index.endswith("_10_",11,15):
            MVC_10.append(file_index)        
        if file_index.endswith("_20_",15,19) or file_index.endswith("_20_",11,15):
            MVC_20.append(file_index)            
        if file_index.endswith("_35_",15,19) or file_index.endswith("_35_",11,15):
            MVC_35.append(file_index)        
        if file_index.endswith("_50_",15,19) or file_index.endswith("_50_",11,15):
            MVC_50.append(file_index) 
        if file_index.endswith("_66_",15,19) or file_index.endswith("_66_",11,15):
            MVC_66.append(file_index)    
        if file_index.endswith("_max_",15,20) or file_index.endswith("_max_",11,16) or file_index.endswith("_max_",9,14):
            MVC_Max.append(file_index)         
            
            
    # Segregation of data based on the Frequency 
        
        
        if file_index.endswith('010',14,18) or file_index.endswith('010',18,22):
            Frequency_1.append(file_index)        
        if file_index.endswith('025',14,18) or file_index.endswith('025',18,22):
            Frequency_2.append(file_index)            
        if file_index.endswith('050',14,18) or file_index.endswith('050',18,22):
            Frequency_3.append(file_index)        
        if file_index.endswith('100',14,18) or file_index.endswith('100',18,22):
            Frequency_4.append(file_index) 
        if file_index.endswith('200',14,18) or file_index.endswith('200',18,22):   
            Frequency_5.append(file_index)
            
    
    my_dict= dict({'First Subject':subject_1,'Second Subject':subject_2,'Third Subject':subject_3,'Forth Subject':subject_4,'Fifth Subject':subject_5,
    'Pinch Movment':Pinch_allsubject,'Force Movment':Force_allsubject,'Precesion Movment':precision_allsubject,
    'MVC_5':MVC_05,'MVC_10':MVC_10,'MVC_20':MVC_20,'MVC_35':MVC_35,'MVC_50':MVC_50,'MVC_66':MVC_66,'MVC_Max':MVC_Max,
    'Freq_10':Frequency_1,'Freq_25':Frequency_2,'Freq_50':Frequency_3,'Freq_100':Frequency_4,'Freq_200':Frequency_5})
     
    max_len =max(len(x) for x in my_dict.values())
    
    
    df=pd.DataFrame({key:vals+[np.nan]*(max_len-len(vals)) for key,vals in my_dict.items()})

    
    data=df[['First Subject','Second Subject','Third Subject','Forth Subject','Fifth Subject', 'Pinch Movment','Precesion Movment'
                     ,'Force Movment','MVC_5','MVC_10', 'MVC_20', 'MVC_35','MVC_50', 'MVC_66', 'MVC_Max','Freq_10','Freq_25', 'Freq_50',
                     'Freq_100', 'Freq_200']]    
    
    print(data.head())
    
    df.to_csv(os.path.join('C:/Users\Hasan_Issa/Downloads/WinPython-64bit-3.4.3.5/python-3.4.3.amd64/MY_Examples/Project_Files',r'Name_Index'),sep='\t')
    
    return(data)





