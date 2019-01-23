# -*- coding: utf-8 -*-
"""
Created on Tue Sep 11 14:00:18 2018

@author: zee
"""


import pandas as pd
import numpy as np

from django.conf import settings
import os

basedir=settings.BASE_DIR
print ('=====================================',basedir,'=============================================')
model_battery_path=os.path.join(basedir,'data/model/coef_Battery.txt')
model_meachine_path=os.path.join(basedir,'data/model/coef_Meachine.txt')

with open(model_battery_path) as f:
    coef_b=f.readlines()
for i,v in enumerate(coef_b):
    coef_b[i]=float(v)
    
with open(model_meachine_path) as f:
    coef_m=f.readlines()
for i,v in enumerate(coef_m):
    coef_m[i]=float(v)
    

def pred(x,y,z,coef):
    score=x*coef[0]+y*coef[1]+z*coef[2]+coef[-1]
    if score>0:
        return 1
    else:
        return 0
    
    
def statue_judge(zip_rgb):
    r1,r1_,r2,r2_,g1,g1_,g2,g2_,b1,b1_,b2,b2_=zip_rgb
    battery=pred(r1-r1_,g1-g1_,b1-b1_,coef_b)
    meachine=pred(r2-r2_,g2-g2_,b2-b2_,coef_m)

    confidence_b=0.9
    confidence_m=0.9
    
    if battery==0 and meachine==0:
        total_statue=0
    if battery==1 and meachine==0:
        total_statue=1
    if battery==0 and meachine==1:
        total_statue=2
    if battery==1 and meachine==1:
        total_statue=3
    
    return total_statue,battery,meachine,confidence_b,confidence_m

    


if __name__=='__main__':
  
    from lr_pred import statue_judge as statue_judge_lr
    
    df_data=pd.read_excel('data2.xlsx')
    #df_data=df_data[['R1','R1_','R2','R2_','G1','G1_','G2','G2_','B1','B1_','B2','B2_']]
    nrows,_=df_data.shape
    df_data['judge']=999
    for i in range(nrows):
        true_statue=df_data.loc[i,['Statue_monitor']]
        val=df_data.loc[i,['R1','R1_','R2','R2_','G1','G1_','G2','G2_','B1','B1_','B2','B2_']]
        
        val=np.array(val)
        df_data.loc[i,['judge']]=statue_judge_lr(val)[0]
    
    
    test=statue_judge([14094,11861,25465,24424,10093,8303,16599,15847,13099,10652,20246,19091])
    
    df_myJudge_failed=df_data[df_data['judge']!=df_data['Display']]   
    df_serial_failed=df_data[df_data['Statue_monitor']!=df_data['Display']] 
    
