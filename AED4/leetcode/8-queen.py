# -*- coding: utf-8 -*-
"""
Created on Mon Sep 17 16:09:23 2018

@author: zee
"""

import time
from  multiprocessing import Pool
from sklearn.linear_model import LogisticRegression

import time

base_time=time.time()

def queen(L=6):
    s=time.time()-base_time
    box=[]
    def dfs(a,L):
        if len(a)==L:
            box.append(a)
        else:
            tt=[x for x in range(L) if x not in a]
            for i in tt:
                count=len(a)
                for k in range(len(a)):
                    if abs(i-a[k])!=abs(len(a)-k):
                        count-=1
                if count==0:
                    
                    dfs(a+[i],L)
    
    dfs([],L)
    e=time.time()-base_time
    print ('%.3f--%.3f : %.3f'%(s,e,(e-s)))
    return s,e,(e-s)

#aa=queen(12)


def one_test(process_num,L=9,total=32):
    p=Pool(process_num)
    log=p.map(queen,[L]*total)
    #    p.apply_async(queen, args=(12,))
#    print('Waiting for all subprocesses done...')
    p.close()
    p.join()
    time_stamp=[x[0] for x in log]+[x[1] for x in log]
    s=min(time_stamp)
    e=max(time_stamp)
    return (e-s)/total

a=queen(10)

#for num in [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,32]:
#    av_time=one_test(num,L=12)
#    print('average time is %.3f when %d process'%(av_time,num))
time.sleep(10)