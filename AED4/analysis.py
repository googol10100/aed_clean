# -*- coding: utf-8 -*-
"""
Created on Mon Aug 27 02:34:12 2018

@author: zee
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import seaborn as sns


excel_path='data4.xlsx'
df=pd.read_excel(excel_path)
df['CC1']=df['R1']+df['G1']+df['B1']
df['CC1_']=df['R1_']+df['G1_']+df['B1_']
df['CC2']=df['R2']+df['G2']+df['B2']
df['CC2_']=df['R2_']+df['G2_']+df['B2_']

df['R1_delta']=df['R1']-df['R1_']
df['R2_delta']=df['R2']-df['R2_']

df['G1_delta']=df['G1']-df['G1_']
df['G2_delta']=df['G2']-df['G2_']

df['B1_delta']=df['B1']-df['B1_']
df['B2_delta']=df['B2']-df['B2_']

df['C1_delta']=df['C1']-df['C1_']
df['C2_delta']=df['C2']-df['C2_']

cor=pearsonr(df['G2_delta'],df['B2_delta'])

df_delta=df.loc[:,['R1_delta','R2_delta','G1_delta','G2_delta','B1_delta','B2_delta']]
statue_monitor=df['Statue_monitor']
statue_monitor=statue_monitor.T

matrix_delta=np.array(df_delta)
cov=np.cov(matrix_delta.T)

df_anomalous=df[(df['R1_delta']>1500) & (df['Display']==2)]
df_anomalous2=df[(df['R2_delta']<900) & (df['Display']==2)]

def jiaozhun(df,col1,col2,monitor,color,statue=0):
    df_=df[(df['Monitor_ID']==monitor) & (df['AED_color']==color) & (df['Display']==statue) ]
    
    arr= np.array(df_[[col1,col2]])
    return np.unique(arr,axis=0)


class Utils():
    def __init__(self,df,col1,col2,labels):
        self.df=df
        self.col1=col1
        self.col2=col2
        self.x=np.array(df[col1])
        self.y=np.array(df[col2])
        self.labels=np.array(df[labels])
        
        self.boxes={}
        for label in [0,1,2,3]:
            xy=df[df[labels]==label]
            xy=xy.loc[:,[col1,col2]]
            xy=np.array(xy)
            self.boxes[label]=xy
        self.statistic=[]
        for label in [0,1,2,3]:
            xy=self.boxes[label] 
            xy_min=xy.min(axis=0)
            xy_max=xy.max(axis=0)
            xy_center=xy.mean(axis=0)
            self.statistic.append(np.concatenate([xy_min,xy_max,xy_center]))
        self.statistic=np.array(self.statistic)
        
        ss=self.statistic
        
        self.xmid=max((ss[0,2],ss[2,2])+min(ss[1,0],ss[3,0]))/2
        self.ymid=max((ss[0,3],ss[1,3])+min(ss[2,1],ss[3,1]))/2
        self.xmid=int(self.xmid)
        self.ymid=int(self.ymid)
        self.lim=np.array([ss.min(axis=0)[0]-100,ss.min(axis=0)[1]-100,ss.max(axis=0)[2]+100,ss.max(axis=0)[3]+100])
        
    def plot(self,fig):
        box=self.boxes      
        f0=plt.scatter(box[0][:,0],box[0][:,1],s=55,facecolor='w',edgecolors='r',marker='o')
        f1=plt.scatter(box[1][:,0],box[1][:,1],s=55,facecolor='w',edgecolors='g',marker='o')
        f2=plt.scatter(box[2][:,0],box[2][:,1],s=55,facecolor='w',edgecolors='b',marker='o')
        f3=plt.scatter(box[3][:,0],box[3][:,1],s=55,facecolor='w',edgecolors='y',marker='o')
        plt.plot([self.xmid,self.xmid],[-10000,10000],'c--',linewidth=5)
        plt.plot([-10000,10000],[self.ymid,self.ymid],'m--',linewidth=5)
        plt.annotate('(%d,%d)'%(self.xmid,self.ymid),np.array([self.xmid,self.ymid]),np.array([self.xmid,self.ymid])+[300,200],fontsize=20,arrowprops=dict(facecolor='black',shrink=0.05))
        plt.title(self.col1[0],fontsize=20)
        plt.xlim([self.lim[0],self.lim[2]])
        plt.ylim([self.lim[1],self.lim[3]])
        

color_list=['r','g','b','y']
marker_list=['o','v','^','s']


fig=plt.figure(figsize=(12,12))

R=Utils(df,'R1_delta','R2_delta','Display')
plt.subplot(221)
R.plot(fig)

G=Utils(df,'G1_delta','G2_delta','Display')
plt.subplot(222)
G.plot(fig)

B=Utils(df,'B1_delta','B2_delta','Display')
plt.subplot(223)
B.plot(fig)

plt.subplot(224)

colormap = sns.diverging_palette(220, 10, as_cmap = True)
_ = sns.heatmap(
        df_delta.corr(), 
        cmap = colormap,
        square=True, 
        cbar_kws={'shrink':.9 }, 
#        ax=ax,
        annot=True, 
        linewidths=0.1,vmax=1.0, linecolor='white',
        annot_kws={'fontsize':12 }
    )
plt.title('Correlation Matrix', size=18)


fig.savefig('scatter4.jpg')



  


"""     

jj=jiaozhun(df,'R1_','R2_',2,'red')

fig=plt.figure(figsize=(12,12))

color_map_color={'red':'r','black':'k'}

monitor_map_shape={2:'o',6:'s',8:'^'}



def one_channel(c='R'):
    for color in ['red','black']:
        for monitor in [2,6,8]:
            jj=jiaozhun(df,c+'1_',c+'2_',monitor,color)
            gg=jiaozhun(df,c+'1',c+'2',monitor,color,statue=0)
            jj=np.concatenate([jj,gg],axis=0)
            xx=jj[:,0]
            yy=jj[:,1]
            plt.scatter(xx,yy,s=150,facecolor='w'
                        ,edgecolors=color_map_color[color],
                        marker=monitor_map_shape[monitor],linewidths=2)
            plt.title(c,fontsize=20)

plt.subplot(221)
one_channel('R')
plt.subplot(222)
one_channel('G')
plt.subplot(223)
one_channel('B')
"""