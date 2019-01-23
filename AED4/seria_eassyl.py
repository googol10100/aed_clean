# -*- coding: utf-8 -*-
"""
Created on Thu Sep 20 10:16:40 2018

@author: zee
"""

import serial  
import os
import time
  
  
#n = t.write('you are my world')  
#print t.portstr  
#print n  
#str = t.read(n)  
#print str  

N=4
aed_id='73654010180'
monitor='6103'
display='0'

base_dir='data5'
if not os.path.exists(base_dir):
    os.mkdir(base_dir)


def oneSet(aed_id,monitor,display):
    ser = serial.Serial('com3',115200)
    save_name=os.path.join(base_dir,'%s-m%s-d%s.txt'%(aed_id,monitor,display)) 
    already_file=0
    while os.path.exists(save_name):
        already_file+=1
        save_name=os.path.join(base_dir,'%s-m%s-d%s.txt'%(aed_id+'~'+str(already_file),monitor,display)) 
        
    print ('start collect data,it will be saved in %s'%save_name)
    
    if display==0 or display=='0':
        print ('you should calibration in 90 s firstly!!!')
        calibration_s=time.time()
        while True:
            line=ser.readline() 
            if '*C,66&' in line:
                print (line)
                print ('calibration done!!!\nsleep 90s')
                time.sleep(90)
                
                break
            if (time.time()-calibration_s)>90:
                print ('No calibration!!!!\nWe had waited 90 s')
                break
    
    box=[]
    count=0
    while True:
        
        line = ser.readline() 
        box.append(line)
#        print (line)        
        if 'UART Data:*B' in line:
            count+=1
            print (line)

        if count==N:
            txt=''
            for row in box:
                txt+=row
            with open(save_name,'w') as f:
                f.write(txt)
            break
    ser.close()


while True:
    print ('======================================================================')
    input_N=raw_input('sample number(default %s):'%str(N))
    if input_N!='':
        N=int(input_N)



    input_aed=raw_input('AED_ID(default %s):'%aed_id)
    if input_aed!='':
        aed_id=input_aed
        
        
    input_monitor=raw_input('Monitor(default %s):'%monitor)
    if input_monitor!='':
        monitor=input_monitor
        
    input_display=raw_input('True display statue(default %s):'%display)
    if input_display!='':
        display=input_display
    
    oneSet(aed_id,monitor,display)
    
    
