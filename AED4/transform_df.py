#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 19 11:52:32 2018

@author: dirac
"""

import os
import pandas as pd
import numpy as np

base_dir='excels'
file_list=os.listdir(base_dir)
file_list=filter(lambda x:x.endswith('xlsx'),file_list)

for name in file_list:
    excel_path=os.path.join(base_dir,name)
    df=pd.read_excel(excel_path)
    
    df['R1_delta']=df['R1']-df['R1_']
    df['R2_delta']=df['R2']-df['R2_']
    
    df['G1_delta']=df['G1']-df['G1_']
    df['G2_delta']=df['G2']-df['G2_']
    
    df['B1_delta']=df['B1']-df['B1_']
    df['B2_delta']=df['B2']-df['B2_']
    
    df['C1_delta']=df['C1']-df['C1_']
    df['C2_delta']=df['C2']-df['C2_']
    
    
    df['Battery']=df['Display']%2
    df['Meachine']=df['Display']//2
     
    df['AED_ID']=df['AED_ID'].astype('str')
#    df[features[1:]]=df[features[1:]].astype('int')
    df.to_excel(excel_path,index=False)
    