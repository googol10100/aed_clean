#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 31 14:41:44 2018

@author: ly
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import keras
from keras.layers import Input, Dense,Dropout,Multiply,Lambda
from keras.models import Model,load_model

from keras.utils import to_categorical




##checkpoint for callback
#checkpoint=ModelCheckpoint(filepath=os.path.join(model_dir,'epoch:{epoch:03d}-loss-cls-recall:{loss:.3f}-{loss_cls:.3f}-{recall:.3f}-{nohard:.3f}-{cls_nohard:.3f}-valloss_cls_recall:{val_loss:.3f}-{val_loss_cls:.3f}-{val_recall:.3f}-{val_nohard:.3f}-{val_cls_nohard:.3f}.h5'), 
#                                monitor='val_loss', 
#                                verbose=0, 
#                                save_best_only=False, 
#                                save_weights_only=False, 
#                                period=1)
def lr_decay(epoch):
    lr=0.003
    if epoch>2:
        lr=0.001
    if epoch>5:
        lr=0.0003
    if epoch>10:
        lr=0.0001
    if epoch>20:
        lr=0.00003
    if epoch>40:
        lr=0.00001
    if epoch>60:
        lr=0.000003
    return lr

split=0.99
n_epochs=2000
excel_path='excels/data_all.xlsx'
df=pd.read_excel(excel_path)

df_test=pd.read_excel('excels/data4-5.xlsx')

df=pd.concat([df,df_test])


shuffle=np.array(range(df.shape[0]))
np.random.shuffle(shuffle)
df=df.iloc[shuffle]
df_train=df[:int(df.shape[0]*split)]
df_val=df[int(df.shape[0]*split):]

features=['R1','R1_','R2','R2_','G1','G1_','G2','G2_','B1','B1_','B2','B2_',]
#features=['R1_delta','R2_delta','G1_delta','G2_delta','B1_delta','B2_delta',]
label=['Display']
n_input=len(features)

def data_augument(data,label,n_times=2):
    box=[data]
    for _ in range(n_times):
        m,n=data.shape
        dummy=data+np.random.randint(-30,30,[m,n])
        box.append(dummy)
    return np.concatenate(box),np.concatenate([label]*(n_times+1))



data_train=np.array(df_train.loc[:,features])
data_train=data_train/10000.
label_train=np.array(df_train.loc[:,label])
onehot_train=to_categorical(label_train)


data_train,onehot_train=data_augument(data_train,onehot_train)

data_val=np.array(df_val.loc[:,features])
data_val=data_val/10000.
label_val=np.array(df_val.loc[:,label])
onehot_val=to_categorical(label_val)



ooxx=data_augument(data_train,label_train)


multi=Lambda(lambda x:x*0.0001)
# This returns a tensor
inputs = Input(shape=(n_input,))
# a layer instance is callable on a tensor, and returns a tensor
x = Dense(32, activation='relu')(inputs)
x = Dense(64, activation='relu')(x)
x = Dense(32, activation='relu')(x)
x = Dense(16, activation='relu')(x)
#x=Dropout(0.1)(x)
predictions = Dense(4, activation='softmax')(x)

# This creates a model that includes
# the Input layer and three Dense layers
model = Model(inputs=inputs, outputs=predictions)
adam=keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
model.compile(optimizer=adam,
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.fit(data_train, onehot_train,epochs=n_epochs,batch_size=128,validation_data=[data_val,onehot_val])  # starts training
model.save('model/neural.h5')