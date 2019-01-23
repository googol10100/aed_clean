# -*- coding: utf-8 -*-
"""
Created on Fri Sep 07 18:07:29 2018

@author: zee
"""



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import seaborn as sns
from sklearn import svm

from mpl_toolkits.mplot3d import Axes3D

from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter


excel_path='data2.xlsx'
df=pd.read_excel(excel_path)
df=df[df['Monitor_ID']!=8]

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

df1_delta=df.loc[:,['R1_delta','G1_delta','B1_delta','Battery']]
data1=np.array(df1_delta)
x1=data1[:,0]
y1=data1[:,1]
z1=data1[:,2]
X1=data1[:,:3]
label1=data1[:,3]

index1_n=np.where(label1==0)
index1_p=np.where(label1==1)

x1_n=data1[index1_n,0]
y1_n=data1[index1_n,1]
z1_n=data1[index1_n,2]

x1_p=data1[index1_p,0]
y1_p=data1[index1_p,1]
z1_p=data1[index1_p,2]




ax = plt.figure(figsize=[15,15]).add_subplot(111, projection = '3d')

f1=ax.scatter(x1_n, y1_n, z1_n, s=50,facecolor='w',edgecolors='g',marker='o')
f2=ax.scatter(x1_p, y1_p, z1_p, s=50,facecolor='w',edgecolors='r',marker='o')

ax.legend((f1,f2),
              ('good','bad'),
              scatterpoints=1,fontsize=20,loc='Best')


ax.set_xlabel('R')

ax.set_ylabel('G')

ax.set_zlabel('B')

 


model = svm.SVC(C=1,kernel='linear')
model.fit(X1[:,:3],label1)
w,b=model.coef_[0],model.intercept_



X = np.linspace(-1000, 3000, 10)
Y = np.linspace(-1000, 3000, 10)
X,Y = np.meshgrid(X, Y)  # 将坐标向量变为坐标矩阵，列为x的长度，行为y的长度
Z = -(b[0]+w[0]*X+w[1]*Y)/w[2]

#surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1,  linewidth=0, antialiased=False,shade=0.9,facecolor='y')
surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)

surf1 = ax.plot_surface(X, Y, Z+1/w[2], cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
surf2 = ax.plot_surface(X, Y, Z-1/w[2], cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)

support=model.support_vectors_
norm=np.linalg.norm(w)
dist=(np.dot(support,w)+b)/norm




def plot_contours(ax, clf, xx, yy, **params):
    """Plot the decision boundaries for a classifier.

    Parameters
    ----------
    ax: matplotlib axes object
    clf: a classifier
    xx: meshgrid ndarray
    yy: meshgrid ndarray
    params: dictionary of params to pass to contourf, optional
    """
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    out = ax.contourf(xx, yy, Z, **params)
    return out



plt.show()