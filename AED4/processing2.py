# -*- coding: utf-8 -*-
"""
Created on Sat Sep 15 13:58:18 2018

@author: zee
"""

import re
import os
import numpy as np
import pandas as pd
import collections


base_dir='data4-5/'

save_excel_path='data4-5.xlsx'

#df =pd.DataFrame(columns=['AED_ID','Monitor_ID','Display','Statue_monitor','R1','R1_','R2','R2_','G1','G1_','G2','G2_','B1','B1_','B2','B2_','C1','C1_','C2','C2_'],index=None)
features=['AED_ID','Monitor_ID','Display','Statue_monitor','R1','R1_','R2','R2_','G1','G1_','G2','G2_','B1','B1_','B2','B2_','C1','C1_','C2','C2_']
mydic=dict(zip(features,[[] for _ in range(len(features))]))

file_list=os.listdir(base_dir)  
for name in file_list:
    path=os.path.join(base_dir,name)
    

with open(path,'r') as f:
    txt=f.readlines()
s=''
for entry in txt:
    s+=entry
s=s.replace('\n','')
    
def f(s,i,pattern):
    len_s=len(s)
    len_p=len(pattern)
    if i+len_p >= len_s:
        return False
    for k in range(len_p):
        if s[i+k]!=pattern[k]:
            return False
    return True


def match(a,start,end):
    box=[]
    length=len(a)
    s_len=len(start)
    e_len=len(end)
    ss=-1

            
            
    for i,v in enumerate(a):
        if ss>0 :
            if f(a,i,end):
               box.append(a[ss:i])
               i=i+e_len
               ss=-1
            else:
                if f(a,i,start):
                    ss=i+s_len
                    i=ss
        else:
            if f(a,i,start):
                ss=i+s_len
                i=ss
    return box
    

nn=match(s,'[',']')
for item in nn:
    s=s.replace('['+item+']','')
hh=match(s,'*B,','&')  











def appendDic(path):
    print (path)

#    path='data/7315-m1-d0.txt'
    aed_id=path.split('/')[-1].split('-')[0]
    m_id=int(path.split('m')[-1].split('-')[0])
    display=int(path.split('d')[-1].split('.')[0])
    
    with open(path,'r') as f:
        txt=f.readlines()
    s=''
    for entry in txt:
        s+=entry
    s=s.replace('\n','')
    
    nn=match(s,'[',']')
    for item in nn:
        s=s.replace('['+item+']','')
    hh=match(s,'*B,','&')      
    
    
    for xx in hh:
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


df['AED_ID']=df['AED_ID'].astype('str')
df[features[1:]]=df[features[1:]].astype('int')
df.to_excel(save_excel_path,index=False)   


dark=df[(df['AED_ID']=='73654011061') | (df['AED_ID']=='73654011061d')]
