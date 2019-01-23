# -*- coding: utf-8 -*-
"""
Created on Tue Sep 11 09:19:10 2018

@author: zee
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import seaborn as sns
from sklearn import svm
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.externals import joblib


from mpl_toolkits.mplot3d import Axes3D

from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import mpl_toolkits.mplot3d as mp3d




class Classifier(object):
    def __init__(self,df,features,which):
        self.features=features
        self.which=which
        self.df_sub=df.loc[:,features+[which]]
        
        self.data=np.array(self.df_sub)[:,:-1]
        self.label=np.array(self.df_sub)[:,-1]

#        
        index0=np.where(self.label==0)
        index1=np.where(self.label==1)
        
        self.data0=self.data[index0]
        self.data1=self.data[index1]
        
        
    def svm(self,save_path,C=0.00000003,class_weight={0:1,1:8}):
        self.model=model = svm.SVC(kernel='linear',C=C,class_weight=class_weight)
        model.fit(self.data,self.label)
        joblib.dump(model, save_path) 
        print ('The model has been saved in %s'%save_path)
        w,b=model.coef_[0],model.intercept_
        self.w=w
        self.b=b
        self.support=support=model.support_vectors_
        norm=np.linalg.norm(w)
        dist=(np.dot(support,w)+b)/norm
        self.margin_dist=abs(dist[0])
#        print ('support vector :')
#        print (support)
        print ('margin distance : %d'%abs(dist[0]))
        with open('model/coef_%s.txt'%self.which,'w') as f:
            for i in w:
                f.write('%.9f\n'%i)
            f.write('%.9f\n'%b[0])
            
            
    def plot(self,isSurface=True):
        data0=self.data0
        data1=self.data1
        ax = plt.figure(figsize=[15,15]).add_subplot(111, projection = '3d')
        
        f1=ax.scatter(data0[:,0], data0[:,1], data0[:,2], s=50,facecolor='w',edgecolors='g',marker='o')
        f2=ax.scatter(data1[:,0], data1[:,1], data1[:,2], s=50,facecolor='w',edgecolors='r',marker='o')
        ax.scatter(self.support[:,0], self.support[:,1], self.support[:,2], s=10,facecolor='k',edgecolors='k',marker='o')
        
        
        ax.legend((f1,f2),
                      ('good','bad'),
                      scatterpoints=1,fontsize=20,loc='Best')
        ax.set_xlabel('R')
        ax.set_ylabel('G')
        ax.set_zlabel('B')
    
        
        
        xy=np.array([[-0.1,-0.1],
               [-0.1,3],
               [3,3],
               [3,-1],
               ])
        w,b=self.model.coef_[0],self.model.intercept_
        
        z=-(b[0]+np.dot(xy,w[:2]))/w[2]
        xyz=np.concatenate([xy,z.reshape([4,1])],axis=1)     
        face1 = mp3d.art3d.Poly3DCollection([xyz], alpha=0.4, linewidth=1)
        # This is the key step to get transparency working
        alpha = 0.5
        face1.set_facecolor((0, 0, 1, alpha))
        ax.add_collection3d(face1)


        z1=z+1/w[2]
        xyz1=np.concatenate([xy,z1.reshape([4,1])],axis=1)     
        face1 = mp3d.art3d.Poly3DCollection([xyz1], alpha=0.1, linewidth=1)
        # This is the key step to get transparency working
        alpha = 0.5
        face1.set_facecolor((0, 0, 1, alpha))
        ax.add_collection3d(face1)
        
        
        z2=z-1/w[2]
        xyz2=np.concatenate([xy,z2.reshape([4,1])],axis=1)     
        face1 = mp3d.art3d.Poly3DCollection([xyz2], alpha=0.1, linewidth=1)
        # This is the key step to get transparency working
        alpha = 0.5
        face1.set_facecolor((0, 0, 1, alpha))
        ax.add_collection3d(face1)


        plt.title(self.which,fontsize=20)
        plt.show()

    def distance(self,xyz):
        norm=np.linalg.norm(self.w)
        dist=(np.dot(np.array(xyz),self.w)+self.b)/norm
        return dist
    def validation(self,df):
        serial=np.array(df.loc[:,'Statue_monitor'])

        if self.which=='Battery':
            for i in range(serial.shape[0]):
                serial[i]=serial[i]%2
#            print (serial)
        if self.which=='Meachine':
            for i in range(serial.shape[0]):
                serial[i]=serial[i]//2
#            print (serial)
        df_sub=df.loc[:,self.features+[self.which]]
        data=np.array(df_sub)[:,:-1]
        label=np.array(df_sub)[:,-1]
        p=self.model.predict(data)
        recall=recall_score(label,p)
        precision=precision_score(label,p)
#        print (serial)
#        print ('==================================')
#        print (label)        
        recall_serial=recall_score(label,serial)
        precision_serial=precision_score(label,serial)        

        return recall,precision,recall_serial,precision_serial
#        return recall,precision,1,1
        


class SVM(Classifier):
    def __init__(self,df,features,which,model_path):
        super(SVM,self).__init__(df,features,which)
        self.model=joblib.load(model_path)

        self.w,self.b=self.model.coef_[0],self.model.intercept_

        self.support=self.model.support_vectors_   

    def plot(self,df,isShowAll=True,isShowSupport=True):
        data0=self.data0
        data1=self.data1
        
        
        df_sub=df.loc[:,self.features+[self.which]]
        data_test=np.array(df_sub)[:,:-1]
        label_test=np.array(df_sub)[:,-1]

        index_test0=np.where(label_test==0)
        index_test1=np.where(label_test==1)
        
        data_test0=data_test[index_test0]
        data_test1=data_test[index_test1]
        
        
        
        ax = plt.figure(figsize=[15,15]).add_subplot(111, projection = '3d')
        
        f1=ax.scatter(data_test0[:,0], data_test0[:,1], data_test0[:,2], s=80,facecolor='g',edgecolors='g',marker='s')
        f2=ax.scatter(data_test1[:,0], data_test1[:,1], data_test1[:,2], s=80,facecolor='r',edgecolors='r',marker='s')
        if isShowAll:
            ax.scatter(data0[:,0], data0[:,1], data0[:,2], s=20,facecolor='w',edgecolors='g',marker='o')
            ax.scatter(data1[:,0], data1[:,1], data1[:,2], s=20,facecolor='w',edgecolors='r',marker='o')            
        
        if isShowSupport:
            ax.scatter(self.support[:,0], self.support[:,1], self.support[:,2], s=10,facecolor='k',edgecolors='k',marker='o')
        
        
        ax.legend((f1,f2),
                      ('good','bad'),
                      scatterpoints=1,fontsize=20,loc='Best')
        ax.set_xlabel('R')
        ax.set_ylabel('G')
        ax.set_zlabel('B')
    
        
        
        xy=np.array([[-100,-100],
               [-100,3000],
               [3000,3000],
               [3000,-100],
               ])
        w,b=self.model.coef_[0],self.model.intercept_
        
        z=-(b[0]+np.dot(xy,w[:2]))/w[2]
        xyz=np.concatenate([xy,z.reshape([4,1])],axis=1)     
        face1 = mp3d.art3d.Poly3DCollection([xyz], alpha=0.4, linewidth=1)
        # This is the key step to get transparency working
        alpha = 0.5
        face1.set_facecolor((0, 0, 1, alpha))
        ax.add_collection3d(face1)


        z1=z+1/w[2]
        xyz1=np.concatenate([xy,z1.reshape([4,1])],axis=1)     
        face1 = mp3d.art3d.Poly3DCollection([xyz1], alpha=0.1, linewidth=1)
        # This is the key step to get transparency working
        alpha = 0.5
        face1.set_facecolor((0, 0, 1, alpha))
        ax.add_collection3d(face1)
        
        
        z2=z-1/w[2]
        xyz2=np.concatenate([xy,z2.reshape([4,1])],axis=1)     
        face1 = mp3d.art3d.Poly3DCollection([xyz2], alpha=0.1, linewidth=1)
        # This is the key step to get transparency working
        alpha = 0.5
        face1.set_facecolor((0, 0, 1, alpha))
        ax.add_collection3d(face1)


        plt.title(self.which,fontsize=20)
        plt.show()        


def ration_one_over_total(df,select_feature,total_feature):
    select=df[select_feature]
    
    dff=df[total_feature]
    total=dff.sum(axis=1)
    total=total.astype('float')
    
    ratio=select/total
    return ratio

if __name__=='__main__':
    split=0.99
    model_battery_path='model/svm_battery.pkl'
    model_meachine_path='model/svm_meachine.pkl'
    excel_path='excels/data_all.xlsx'
    df=pd.read_excel(excel_path)
    
    df_test=pd.read_excel('excels/data4-5.xlsx')
    
    df=pd.concat([df,df_test])
    df['R1_ratio']=ration_one_over_total(df,'R1',['R1','G1','B1'])
    df['G1_ratio']=ration_one_over_total(df,'G1',['R1','G1','B1'])
    df['B1_ratio']=ration_one_over_total(df,'B1',['R1','G1','B1'])
    
    df['R2_ratio']=ration_one_over_total(df,'R2',['R2','G2','B2'])
    df['G2_ratio']=ration_one_over_total(df,'G2',['R2','G2','B2'])
    df['B2_ratio']=ration_one_over_total(df,'B2',['R2','G2','B2'])
    
    
    
    shuffle=np.array(range(df.shape[0]))
    np.random.shuffle(shuffle)
    df=df.iloc[shuffle]
    df_train=df[:int(df.shape[0]*split)]
    df_val=df[int(df.shape[0]*split):]
            
    
    battery=Classifier(df_train,features=['R1_ratio','G1_ratio','B1_ratio'],
                       which='Battery')
    battery.svm(save_path=model_battery_path,C=1,class_weight={0:1,1:2})
    battery.plot()
    recall_b,precision_b,recall_b_ser,precision_b_ser=battery.validation(df_val)
    
    
    #df_train['B_dist']=0
    #df_train['B_dist']=df_train.apply(lambda dfx:int(battery.distance([dfx['R1_delta'],dfx['G1_delta'],dfx['B1_delta']])),axis=1)
    
    
    
    meachine=Classifier(df_train,features=['R2_ratio','G2_ratio','B2_ratio'],
                       which='Meachine')
    meachine.svm(save_path=model_meachine_path,C=1,class_weight={0:1,1:4})
    meachine.plot()
    recall_m,precision_m,recall_m_ser,precision_m_ser=meachine.validation(df_val)
    
    print ('============battery===========')
    print ('new recall:%.4f , new precision:%.4f'%(recall_b,precision_b))
    print ('old recall:%.4f , old precision:%.4f'%(recall_b_ser,precision_b_ser))
    print ('============battery===========\n')
    
    
    print ('============meachine===========')
    print ('new recall:%.4f , new precision:%.4f'%(recall_m,precision_m))
    print ('old recall:%.4f , old precision:%.4f'%(recall_m_ser,precision_m_ser))
    print ('============meachine===========\n')



    restore_battery=SVM(df_train,features=['R1_ratio','G1_ratio','B1_ratio'],
                       which='Battery',model_path=model_battery_path)

    restore_battery.plot(df_test,isShowAll=True)
    
    
    
    restore_meachine=SVM(df_train,features=['R2_ratio','G2_ratio','B2_ratio'],
                       which='Meachine',model_path=model_meachine_path)

    restore_meachine.plot(df_test,isShowAll=True)