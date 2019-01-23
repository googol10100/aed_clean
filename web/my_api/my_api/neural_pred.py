#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  1 16:42:57 2018

@author: ly
"""

import numpy as np

from keras.layers import Input, Dense,Dropout,Multiply,Lambda
from keras.models import Model,load_model
from keras.utils import to_categorical

from django.conf import settings
import os

basedir=settings.BASE_DIR
#basedir='../'
print ('=====================================',basedir,'=============================================')
model_path=os.path.join(basedir,'data/model/neural.h5')

model_neural=load_model(model_path)
#box=[]
#for layer in model.layers:
#    box.append(layer.get_weights())

test_data=[6137,3972,6021,4360,4745,3208,5528,4009,6771,4716,7974,5932]
test_data=np.array(test_data).reshape([1,-1])
print (model_neural.predict(test_data))


def statue_judge(zip_rgb):
    zip_rgb=np.array(zip_rgb).reshape([1,-1])
    zip_rgb=zip_rgb/10000.
    pred=model_neural.predict(zip_rgb)
#    pred=np.array([[12,14,15,16]])
    max_index=pred.argmax()
    confidence=pred[0][max_index]  
    return max_index,confidence

if __name__=='__main__':
    test_data=[6137,3972,6021,4360,4745,3208,5528,4009,6771,4716,7974,5932]
    test_data=np.array(test_data).reshape([1,-1])

    pred=model_neural.predict(test_data/10000.)
    s=pred.argmax()
    
    ooxx=statue_judge(test_data)