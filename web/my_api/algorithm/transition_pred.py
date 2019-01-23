#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 25 09:55:05 2018

@author: ly
"""

import numpy as np


Thred_b=np.array([1.05,1.05,1.05])
Thred_m=np.array([1.05,1.05,1.05])


def transition(cur,pre_good,pre_bad,threds):
    """
    cur:npArray [1003,3423,5435]
    
    pre_good:  [[432,5435,3243],
                [3353,564,4234],
                ...............
                [4534,2423,4324]]
    """
    mean_good=np.mean(pre_good,axis=0)
    mean_bad=np.mean(pre_bad,axis=0)
    if len(pre_good) and len(pre_bad):
        dist_good=np.linalg.norm(cur-mean_good)
        dist_bad=np.linalg.norm(cur-mean_bad)
        if dist_good<=dist_bad:
            return 0
        else:
            return 1
    elif len(pre_good):
        concat=np.concatenate((cur.reshape((1,-1)),mean_good.reshape((1,-1)))).astype('float')
        ratio=concat.max(axis=0)/concat.min(axis=0)
        ratio_great_thred=np.greater(ratio,threds)
        great=np.greater(cur,mean_good)
        cc=np.array([great,ratio_great_thred])
        boolean=cc.all(axis=0)
        sum_bool=boolean.sum()
        if sum_bool>1.5:
            return 1
        else:
            return 0
    elif len(pre_bad):
        concat=np.concatenate((cur.reshape((1,-1)),mean_bad.reshape((1,-1)))).astype('float')
        ratio=concat.max(axis=0)/concat.min(axis=0)
        ratio_great_thred=np.greater(ratio,threds)
        great=np.greater(mean_bad,cur)
        cc=np.array([great,ratio_great_thred])
        boolean=cc.all(axis=0)
        sum_bool=boolean.sum()
        if sum_bool>1.5:
            return 0
        else:
            return 1


def total_judge(observe,dic):
    """
    observe: npArray[5290-4869-6675-6447-4657-4259-5399-5276-6430-5885-7002-6974]
    dic:   {'b_good':b_good,
            'b_bad':b_bad,
            'm_good':m_good,
            'm_bad':m_bad}
    """
    
    R1,_,R2,_,G1,_,G2,_,B1,_,B2,_=observe
    cur_b=np.array([R1,G1,B1])
    cur_m=np.array([R2,G2,B2])
    b_good=np.array(dic['b_good'])
    b_bad=np.array(dic['b_bad'])
    m_good=np.array(dic['m_good'])
    m_bad=np.array(dic['m_bad'])
    
    battery=transition(cur_b,b_good,b_bad,Thred_b)
    meachine=transition(cur_m,m_good,m_bad,Thred_m)
    if battery==0 and meachine==0:
        total_statue=0
    if battery==1 and meachine==0:
        total_statue=1
    if battery==0 and meachine==1:
        total_statue=2
    if battery==1 and meachine==1:
        total_statue=3
    return total_statue,battery,meachine,88,88


def total_judge_formularTJ(observe,dic):
    """
    observe: npArray[5290-4869-6675-6447-4657-4259-5399-5276-6430-5885-7002-6974]
    dic:   {'b_good':b_good,
            'b_bad':b_bad,
            'm_good':m_good,
            'm_bad':m_bad}
    """
    coef_r=-0.68202
    coef_g=0.77073
    coef_b=0.56332
    coef=np.array([[coef_r]*3,[coef_g]*3,[coef_b]*3])
    
    
    R1,_,R2,_,G1,_,G2,_,B1,_,B2,_=observe
#    z_b=coef_r*R1+coef_g*G1+coef_b*B1
#    z_m=coef_r*R2+coef_g*G2+coef_b*B2
    
    
    cur_b=np.array([R1,G1,B1]).reshape((-1,3))
    cur_m=np.array([R2,G2,B2]).reshape((-1,3))
    b_good=np.array(dic['b_good']).reshape((-1,3))
    b_bad=np.array(dic['b_bad']).reshape((-1,3))
    m_good=np.array(dic['m_good']).reshape((-1,3))
    m_bad=np.array(dic['m_bad']).reshape((-1,3))
    
    cur_b=np.dot(cur_b,coef).squeeze()
    cur_m=np.dot(cur_m,coef).squeeze()
    b_good=np.dot(b_good,coef)
    b_bad=np.dot(b_bad,coef)
    m_good=np.dot(m_good,coef)
    m_bad=np.dot(m_bad,coef)
    
    battery=transition(cur_b,b_good,b_bad,Thred_b)
    meachine=transition(cur_m,m_good,m_bad,Thred_m)
    if battery==0 and meachine==0:
        total_statue=0
    if battery==1 and meachine==0:
        total_statue=1
    if battery==0 and meachine==1:
        total_statue=2
    if battery==1 and meachine==1:
        total_statue=3
    return total_statue,battery,meachine,88,88
  

if __name__=='__main__':
    cur=np.array([1003,3423,5435])
    pre_good=np.random.randint(0,1000,(0,0))
    pre_bad=np.random.randint(0,1000,(6,3))
#    threds=np.array([1.1,1.08,1.14])
    a=transition(cur,pre_good,pre_bad,Thred_b)