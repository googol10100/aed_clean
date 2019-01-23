# -*- coding: utf-8 -*-
"""
Created on Thu Oct 11 01:04:22 2018

@author: zee
"""

with open('coef_Battery.txt') as f:
    coef_b=f.readlines()
for i,v in enumerate(coef_b):
    coef_b[i]=float(v)
    
with open('coef_Meachine.txt') as f:
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
    """
    input: array [r1,r1_,r2,r2_,g1,g1_,g2,g2_,b1,b1_,b2,b2_]
    return : int 
    """
    
    r1,r1_,r2,r2_,g1,g1_,g2,g2_,b1,b1_,b2,b2_=zip_rgb
    battery=pred(r1-r1_,g1-g1_,b1-b1_,coef_b)
    meachine=pred(r2-r2_,g2-g2_,b2-b2_,coef_m)
    if battery==0 and meachine==0:
        return 0  
    if battery==1 and meachine==0:
        return 1
    if battery==0 and meachine==1:
        return 2
    if battery==1 and meachine==1:
        return 3