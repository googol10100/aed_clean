#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 12 10:23:45 2018

@author: ly
"""

import numpy as np
from sklearn.externals import joblib



model_battery_path='model/lr_battery.pkl'
model_meachine_path='model/lr_meachine.pkl'

test_data=[14094,11861,25465,24424,10093,8303,16599,15847,13099,10652,20246,19091]

model_b = joblib.load(model_battery_path) 
model_m = joblib.load(model_meachine_path) 


def statue_judge(zip_rgb):
    r1,r1_,r2,r2_,g1,g1_,g2,g2_,b1,b1_,b2,b2_=zip_rgb
    battery_data=np.array([r1-r1_,g1-g1_,b1-b1_]).reshape((1,-1))
    meachine_data=np.array([r2-r2_,g2-g2_,b2-b2_]).reshape((1,-1))
    
    battery=model_b.predict(battery_data)[0]
    meachine=model_m.predict(meachine_data)[0]
    
    proba_b=model_b.predict_proba(battery_data)
    proba_m=model_m.predict_proba(meachine_data)

    
    confidence_b=battery*proba_b[0][1]+(1-battery)*proba_b[0][0]
    confidence_m=meachine*proba_m[0][1]+(1-meachine)*proba_m[0][0]
    print(confidence_b,confidence_m)
    
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
    p=statue_judge(test_data)