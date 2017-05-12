'''
This function will combine three feature matrix for three different movment and perfrom PCA 
and LDA to validate the accuracy of the Features 

'''

# import the required Libraries 

import os 
import numpy as np
import pandas as pd
from statistics import median
from itertools import groupby
import matplotlib.pylab as plt 
from Feature_Calculation import EMG_Feature
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from Feature_Calculation_threshold import EMG_Feature_Threshold

# call the Feature caluclation function for three different movment 

df1 = EMG_Feature_Threshold('02_02_force_05_010.txt')
df2 = EMG_Feature_Threshold('02_03_pinch_05_010.txt')
df3 = EMG_Feature_Threshold('03_01_precision_05_010.txt')
df4 = EMG_Feature_Threshold('03_01_precision_05_025.txt')
df5 = EMG_Feature_Threshold('02_03_pinch_50_050.txt')
df6 = EMG_Feature_Threshold('05_02_force_20_100.txt')
df7 = EMG_Feature_Threshold('03_02_force_05_050.txt')
df8 = EMG_Feature_Threshold('04_03_pinch_05_025.txt')
df9 = EMG_Feature_Threshold('05_01_precision_10_100.txt')

frame  = [df1,df2,df3,df4,df5,df6,df7,df8,df9]
df = pd.concat(frame)
df.drop('Channel label',axis=1,inplace=True)

# Label encoder 

from sklearn.preprocessing import LabelEncoder

X_1 = df[[1,2,3,4,5,6,7,8]].values

y = df['File Name'].values

enc = LabelEncoder()
label_encoder = enc.fit(y)
y = label_encoder.transform(y) + 1

# Normalization of the data between 1 and zero 
min_max_scaler = preprocessing.MinMaxScaler()
X = min_max_scaler.fit_transform(X_1)

#X = StandardScaler().fit_transform(X)

label_dict = {1: 'Force', 2: 'Pinch', 3:'precision'}

# For Label Encoding : Force is 1 , Pinch is 2 , Prescison is 3 


from sklearn.decomposition import PCA as sklearnPCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

sklearn_pca = sklearnPCA()
X_pca = sklearn_pca.fit_transform(X)



def plot_pca():

    ax = plt.subplot(111)

    for label,marker,color in zip(
        range(1,4),('^', 's', 'o'),('blue', 'red', 'green')):

        plt.scatter(x=X_pca[:,0][y == label],
                y=X_pca[:,1][y == label],
                marker=marker,
                color=color,
                alpha=0.5,
                label=label_dict[label]
                )

    plt.xlabel('PC1')
    plt.ylabel('PC2')

    leg = plt.legend(loc='upper right', fancybox=True)
    leg.get_frame().set_alpha(0.5)
    plt.title('PCA: Movment projection onto the first 2 principal components')

    # hide axis ticks
    plt.tick_params(axis="both", which="both", bottom="off", top="off",  
            labelbottom="on", left="off", right="off", labelleft="on")

    # remove axis spines
    ax.spines["top"].set_visible(False)  
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)    

    plt.tight_layout
    plt.grid()
    plt.show()

plot_pca()

sklearn_lda = LDA()
X_lda_sklearn = sklearn_lda.fit_transform(X_pca, y)

def plot_scikit_lda(X, title):

    ax = plt.subplot(111)
    for label,marker,color in zip(
        range(1,4),('^', 's', 'o'),('blue', 'red', 'green')):

        plt.scatter(x=X[:,0][y == label],
                    y=X[:,1][y == label] * -1, # flip the figure
                    marker=marker,
                    color=color,
                    alpha=0.5,
                    label=label_dict[label])

    plt.xlabel('LD1')
    plt.ylabel('LD2')

    leg = plt.legend(loc='upper right', fancybox=True)
    leg.get_frame().set_alpha(0.5)
    plt.title(title)

    # hide axis ticks
    plt.tick_params(axis="both", which="both", bottom="off", top="off",  
            labelbottom="on", left="off", right="off", labelleft="on")

    # remove axis spines
    ax.spines["top"].set_visible(False)  
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)    

    plt.grid()
    plt.tight_layout
    plt.show()
    
plot_scikit_lda(X_lda_sklearn, title='Default LDA via scikit-learn')

# try the classification method with one file   

df2 = EMG_Feature_Threshold('06_02_force_10_200.txt')
df2.drop('Channel label',axis=1,inplace=True)
df2.drop('File Name',axis=1,inplace=True)

new = df2[[0,1,2,3,4,5,6,7]].values

#min_max_scaler = preprocessing.MinMaxScaler()
N = min_max_scaler.fit_transform(new)
#N = StandardScaler().fit_transform(N)


Z = sklearn_lda.transform(N) #using the model to project Z
#z_labels = sklearn_lda.predict(Z) #gives you the predicted label for each sample
#z_prob = sklearn_lda.predict_proba(N) #the probability of each sample to belong to each class

print(Z)
#print(z_labels)
#print(z_prob)