# -*- coding: utf-8 -*-
"""
Created on Wed Oct 10 10:52:57 2018

@author: zee
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import seaborn as sns
from sklearn import svm
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import roc_curve, auc

from sklearn.linear_model import LogisticRegression
from sklearn.externals import joblib

from sklearn.model_selection import StratifiedKFold

from mpl_toolkits.mplot3d import Axes3D

from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import mpl_toolkits.mplot3d as mp3d




class Classifier():
    def __init__(self,df,features,which):
        self.features=features
        self.which=which
        self.df_sub=df.loc[:,features+[which]]
        
        self.data=np.array(self.df_sub)[:,:-1]
        self.label=np.array(self.df_sub)[:,-1]

#        
        index0=np.where(self.label==0)
        index1=np.where(self.label==1)
        
        self.data0=self.data[index0]
        self.data1=self.data[index1]
        
            
    def lr(self,save_path=None):
        self.model=model =LogisticRegression(penalty='l2',C=1500,random_state=0, solver='lbfgs',
                        multi_class='multinomial')
        
        model.fit(self.data,self.label)
        if save_path:
            joblib.dump(model, save_path) 
            print ('The model has been saved in %s'%save_path)
        
    def validation(self,df):
        serial=np.array(df.loc[:,'Statue_monitor'])

        if self.which=='Battery':
            for i in range(serial.shape[0]):
                serial[i]=serial[i]%2
#            print (serial)
        if self.which=='Meachine':
            for i in range(serial.shape[0]):
                serial[i]=serial[i]//2
#            print (serial)
        df_sub=df.loc[:,self.features+[self.which]]
        data=np.array(df_sub)[:,:-1]
        label=np.array(df_sub)[:,-1]
        p=self.model.predict(data)

        scores=self.model.predict_proba(data)[:,1]
        fpr, tpr, thresholds = roc_curve(label, scores, pos_label=1,drop_intermediate =False)
        recall=recall_score(label,p)
        precision=precision_score(label,p)
    
        recall_serial=recall_score(label,serial)
        precision_serial=precision_score(label,serial)        

        return recall,precision,recall_serial,precision_serial,fpr, tpr, thresholds
#        return recall,precision,1,1
   
def delta_ln(df,select_feature,cali_feature):
    select=df[select_feature]
    ln_select=np.log(select)
    cali=df[cali_feature]
    ln_cali=np.log(cali)
    delta=ln_select-ln_cali
    return delta
     
if __name__=='__main__':
    split=0.999
    model_battery_path='model/lr_battery.pkl'
    model_meachine_path='model/lr_meachine.pkl'
    excel_path='excels/data_all.xlsx'
    df=pd.read_excel(excel_path)
    
    df_test=pd.read_excel('excels/data4-5.xlsx')
    
    df=pd.concat([df,df_test])

    df['R1_delta_ln']=delta_ln(df,'R1','R1_')
    df['G1_delta_ln']=delta_ln(df,'G1','G1_')
    df['B1_delta_ln']=delta_ln(df,'B1','B1_')
    
    df['R2_delta_ln']=delta_ln(df,'R2','R2_')
    df['G2_delta_ln']=delta_ln(df,'G2','G2_')
    df['B2_delta_ln']=delta_ln(df,'B2','B2_')
    
    
    shuffle=np.array(range(df.shape[0]))
    np.random.shuffle(shuffle)
    df=df.iloc[shuffle]
    df_train=df[:int(df.shape[0]*split)]
    df_val=df[int(df.shape[0]*split):]

    
    

    battery=Classifier(df_train,features=['R1_delta_ln','G1_delta_ln','B1_delta_ln'],
                       which='Battery')
    battery.lr(save_path=model_battery_path)
    recall_b,precision_b,recall_b_ser,precision_b_ser,fpr_b,tpr_b,thred_b=battery.validation(df_train)
    
    
    
    
    meachine=Classifier(df_train,features=['R2_delta_ln','G2_delta_ln','B2_delta_ln'],
                       which='Meachine')
    meachine.lr(save_path=model_meachine_path)
    recall_m,precision_m,recall_m_ser,precision_m_ser,fpr_m,tpr_m,thred_m=meachine.validation(df_train)
    
    
    print ('============battery===========')
    print ('new recall:%.4f , new precision:%.4f'%(recall_b,precision_b))
    print ('old recall:%.4f , old precision:%.4f'%(recall_b_ser,precision_b_ser))
    print ('============battery===========\n')
    
    
    print ('============meachine===========')
    print ('new recall:%.4f , new precision:%.4f'%(recall_m,precision_m))
    print ('old recall:%.4f , old precision:%.4f'%(recall_m_ser,precision_m_ser))
    print ('============meachine===========\n')



plt.figure()
lw = 2
plt.plot(fpr_m, tpr_m, color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % 2)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()