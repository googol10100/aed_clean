#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 11 16:10:21 2018

@author: ly
"""

from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt

import time
import os
import math

import lr_pred
import svm_pred
import neural_pred


@csrf_exempt
def predict(request,param):
    zipdata=param.split('-')
    uids=zipdata[12:]
    zipdata=zipdata[:12]
    zipdata=map(lambda x:int(x),zipdata)
    zipdata=map(lambda x:math.log(x),zipdata)
    statue1,statue_battery1,statue_meachine1,confidence_battery1,confidence_meachine1=lr_pred.statue_judge(zipdata)
    statue2,statue_battery2,statue_meachine2,confidence_battery2,confidence_meachine2=svm_pred.statue_judge(zipdata)
    statue3,confidence3=neural_pred.statue_judge(zipdata)
    return JsonResponse({'input': param, 
                         'statue1':statue1,
                         'confidence1':float(confidence_battery1*confidence_meachine1),
                         'statue_battery1':statue_battery1,
                         'statue_meachine1':statue_meachine1,
                         #'confidence_battery':confidence_battery,
                         #'confidence_meachine':confidence_meachine,
                         'statue2':statue2,
                         'confidence2':float(confidence_battery2*confidence_meachine2),
                         'statue_batterys2':statue_battery2,
                         'statue_meachine2':statue_meachine2,
                         #'confidence_battery':confidence_battery,
                         #'confidence_meachine':confidence_meachine,
                         'statue3':statue3,
                         'confidence3':float(confidence3),
                         })



@csrf_exempt
def lr(request,param):
    zipdata=param.split('-')
    zipdata=map(lambda x:int(x),zipdata)
    statue,statue_battery,statue_meachine,confidence_battery,confidence_meachine=lr_pred.statue_judge(zipdata)
    
    return JsonResponse({'input': param, 
                         'statue':statue,
                         'statue_battery':statue_battery,
                         'statue_meachine':statue_meachine,
                         #'confidence_battery':confidence_battery,
                         #'confidence_meachine':confidence_meachine
                         })


    
    
@csrf_exempt
def svm(request,param):
    zipdata=param.split('-')
    zipdata=map(lambda x:int(x),zipdata)
    statue,statue_battery,statue_meachine,confidence_battery,confidence_meachine=svm_pred.statue_judge(zipdata)
    
    return JsonResponse({"input": param, 
                         "statue":statue,
                         'statue_battery':statue_battery,
                         'statue_meachine':statue_meachine,
                         #'confidence_battery':confidence_battery,
                         #'confidence_meachine':confidence_meachine
                         })