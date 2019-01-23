# -*- coding: utf-8 -*-
"""
Created on Fri Aug 24 08:53:19 2018

@author: zee
"""
import re
import os
import numpy as np
import pandas as pd
import collections


base_dir='data/'

#df =pd.DataFrame(columns=['AED_ID','Monitor_ID','Display','Statue_monitor','R1','R1_','R2','R2_','G1','G1_','G2','G2_','B1','B1_','B2','B2_','C1','C1_','C2','C2_'],index=None)
features=['AED_ID','Monitor_ID','Display','Statue_monitor','R1','R1_','R2','R2_','G1','G1_','G2','G2_','B1','B1_','B2','B2_','C1','C1_','C2','C2_']
mydic=dict(zip(features,[[] for _ in range(len(features))]))


def appendDic(path):
    print (path)

#    path='data/7315-m1-d0.txt'
    aed_id=int(path.split('/')[-1].split('-')[0])
    m_id=int(path.split('m')[-1].split('-')[0])
    display=int(path.split('d')[-1].split('.')[0])
    
    with open(path,'r') as f:
        txt=f.readlines()
    
    
    for entry in txt:
        if 'ART Data' in entry:
            xx=entry.split('*B,')[-1].split('&')[0]
#            xx=[val for val in xx if val not in ['R','G','B','C']]
    #        x=x.split(',')
    #        x=x.split(',')
            xx=re.split('[,: ]',xx)
            x=[val for val in xx if val not in ['R','G','B','C','']]
            
            if len(x)<17:
                continue
            mydic['AED_ID'].append(aed_id)
            mydic['Monitor_ID'].append(m_id)
            mydic['Display'].append(display)
    #        mydic['Statue_monitor'].append(x[0])
            for i in range(len(x)):
                mydic[features[i+3]].append(x[i])
       
        
file_list=os.listdir(base_dir)  
for name in file_list:
    path=os.path.join(base_dir,name)
    appendDic(path)

df=pd.DataFrame(mydic,columns=features)  
c1_=df['C1_']
c2_=df['C2_']
c12_=zip(c1_,c2_)
counter=collections.Counter(c12_)
mapper={}
for key in counter:
    if counter[key]!=1:
        mapper[key[0]]=key[1]
    

for i in range(df.shape[0]):
    df.loc[i,'C2_']=mapper[df.loc[i,'C1_']]

df=df.astype('int64')
df.to_excel('data.xlsx',index=False)       
