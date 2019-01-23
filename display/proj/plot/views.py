from django.shortcuts import render

# Create your views here.
from django.http import JsonResponse
from django.http import HttpResponseRedirect,HttpResponse
from django.views.decorators.csrf import csrf_exempt

import time
import datetime
from datetime import timedelta
import django.utils.timezone as timezone
import os
import pandas as pd
import numpy as np
#from sklearn import linear_model
#import matplotlib.pyplot as plt
#plt.switch_backend('agg')

import sqlite3

from django.conf import settings
from models import AlgorithmRgb
try:
    basedir=settings.BASE_DIR
except:
    basedir='..'
    
    
db_path=os.path.join(basedir,'static/test.sqlite3')
media_dir=os.path.join(basedir,'static/media')

table_name='algorithm_rgb'
id1='868994037706145'
conn = sqlite3.connect(db_path)
#cur = conn.cursor()
#sql_command='SELECT * FROM '+table_name+' WHERE Uid='+id1
#df = pd.read_sql_query(sql_command, conn)


@csrf_exempt
def p(request,param):
    
    return JsonResponse({'input': param, 
                         'statue3':4,
                         })
    
def pp(request):
    
    return JsonResponse({'input': 'fghjk', 
                         'statue3':4,
                         })


def pull_data(uid):
    sql_command='SELECT * FROM '+table_name+' WHERE Uid='+uid
    df = pd.read_sql_query(sql_command, conn)
    return df    
    
def index(request):
    return render(request, 'index.html')

def index2(request):
    return render(request, 'index2.html')

def index3(request):
    return render(request, 'index3.html')

def index4(request):
    return render(request, 'index4.html')

def index5(request):
    return render(request, 'index5.html')
     
def add(request):

    color= request.GET['color']
    uid=request.GET['uid']
    uid=str(uid)
  
    select=AlgorithmRgb.objects.filter(Uid=uid)
    select.all().order_by("Datetime")
    t=map(lambda x:x.Datetime,select)
    y=select.values(color)
    y=map(lambda x : x[color],y)
#    fig=plt.figure(figsize=[18,9])
#    plt.subplot(111)
#    plt.plot(t,y,'r',linewidth=1)
##    plt.legend(fontsize=20)
#    plt.xlabel('Time',fontsize=15)
#    img_path=os.path.join(media_dir,uid+color+'.jpg')
#    plt.savefig(img_path)
#    time.sleep(1)
#    target_name=os.path.join('/static/media/',uid+color+'.jpg')
#    return HttpResponse('<img src="%s" height="1800" width="3600" />  <br/>'%(target_name))
    return HttpResponse('hhhhhhhhhhhhhhhhhhhhhhhhhh')


#def future_pred(past_t,past_y,future_t):
#    past_t=np.array(past_t).reshape(-1,1)
#    past_y=np.array(past_y).reshape(-1,1)
#    past_ln=np.log(past_t)
#    reg = linear_model.LinearRegression()
#    reg.fit(past_t,past_ln)
#    k=reg.coef_[0]
#    b=reg.intercept_
#    A=np.exp(b)
#    print (A,k)
#    def fit(t):
#        return int(A*np.exp(k*t))
#    ret=[fit(x) for x in future_t]
#    return ret




def getData(request):
    max_points=500000
    beginDate = request.GET.get("beginDate", "2018-01-22")
    endDate = request.GET.get("endDate")
    
    endDate_=datetime.datetime.strptime(endDate, "%Y-%m-%d")
    endDate_=endDate_+timedelta(days=1)
    endDate=endDate_
    
    color=request.GET.get("color")
    uid=request.GET.get("uid")
    if uid=='':
        uid='fsld4fijo89f45trjesouijwire984309cddsifdscsfsf'
    cali=request.GET.get("cali")
    
    print(endDate,color,uid,cali)

    select=AlgorithmRgb.objects.filter(Uid=uid)
    select=select.filter(Datetime__range=[beginDate, endDate])
    select.all().order_by("Datetime")
#    t=map(lambda x:x.Datetime,select)
    t=map(lambda x:int(1000*x.Timestamp),select)
    
    if color != '====':
        y=select.values(color)
        y=map(lambda x : x[color],y)
    
    if cali != '====':
        color_cali=cali+'C'
        z=select.values(color_cali)
        z=map(lambda x : x[color_cali],z)
    
    n=len(t)
    if n==0:
        return []
    interval=int(np.ceil(n/float(max_points)))
    indexs=range(0,n,interval)

    

    tt=[[1370131200000, 0.7695],
        [1370908800000, 0.751],
        [1370995200000, 0.7498]]
    appRank={}
    if color != '====':
        ty=[[t[i],y[i]] for i in indexs]
        appRank['value']=ty
    if cali != '====':
        tz=[[t[i],z[i]] for i in indexs]
        appRank['calibration']=tz
#    appRank = {'value':ty,'calibration':tz}
#    appRank = {'value':ty}
    return JsonResponse(appRank)













def getRatioData(request):
    max_points=500000
    beginDate = request.GET.get("beginDate", "2018-01-22")
    endDate = request.GET.get("endDate")
    
    endDate_=datetime.datetime.strptime(endDate, "%Y-%m-%d")
    endDate_=endDate_+timedelta(days=1)
    endDate=endDate_
    
    color=request.GET.get("color")
    uid=request.GET.get("uid")

    select=AlgorithmRgb.objects.filter(Uid=uid)
    select=select.filter(Datetime__range=[beginDate, endDate])
    select.all().order_by("Datetime")
#    t=map(lambda x:x.Datetime,select)
    t=map(lambda x:int(1000*x.Timestamp),select)
    y=select.values(color)
    y=map(lambda x : x[color],y)
    if color in ['R1','G1','B1']:
        RR='R1'
        GG='G1'
        BB='B1'
        
    if color in ['R2','G2','B2']:
        RR='R2'
        GG='G2'
        BB='B2'
    
    y_r=select.values(RR)
    y_r=map(lambda x : x[RR],y_r)
    
    y_g=select.values(GG)
    y_g=map(lambda x : x[GG],y_g)
    
    y_b=select.values(BB)
    y_b=map(lambda x : x[BB],y_b)
    
    

    y_total=[y_r[i]+y_g[i]+y_b[i] for i in range(len(y))]
    y_ratio=[y[i]/float(y_total[i]) for i in range(len(y))]
    y=y_ratio
    
    color_cali=color+'C'
    z=select.values(color_cali)
    z=map(lambda x : x[color_cali],z)
    
    n=len(y)
    if n==0:
        return []
    interval=int(np.ceil(n/float(max_points)))
    indexs=range(0,n,interval)

    print(endDate,color,uid,interval)
    
#    today=t[-1]
#    end=int(1000*time.mktime(endDate_.timetuple()))
#    future_time=range(today,end,24*3600*1000)
#    ret=future_pred(t,y,future_time)
#    tz=[[future_time[i],ret[i]] for i in range(len(future_time))]
    
    ty=[[t[i],y[i]] for i in indexs]
    

#    appRank = {'value':ty,'calibration':tz}
    appRank = {'value':ty}
    return JsonResponse(appRank)














def getBigData(request):
    max_points=500000
    beginDate = request.GET.get("beginDate", "2018-01-22")
    endDate = request.GET.get("endDate")
    
    endDate_=datetime.datetime.strptime(endDate, "%Y-%m-%d")
    endDate_=endDate_+timedelta(days=1)
    endDate=endDate_
    
    color=request.GET.get("color")
    uid=request.GET.get("uid")

    select=AlgorithmRgb.objects.filter(Uid=uid)
    select=select.filter(Datetime__range=[beginDate, endDate])
    select.all().order_by("Datetime")
#    t=map(lambda x:x.Datetime,select)
    t=map(lambda x:int(1000*x.Timestamp),select)
    y=select.values(color)
    y=map(lambda x : x[color],y)
    
    color_cali=color+'C'
    z=select.values(color_cali)
    z=map(lambda x : x[color_cali],z)
    
    n=len(y)
    if n==0:
        return []
    interval=int(np.ceil(n/float(max_points)))
    indexs=range(0,n,interval)

    print(endDate,color,uid,interval)
    
#    today=t[-1]
#    end=int(1000*time.mktime(endDate_.timetuple()))
#    future_time=range(today,end,24*3600*1000)
#    ret=future_pred(t,y,future_time)
#    tz=[[future_time[i],ret[i]] for i in range(len(future_time))]
    
    ty=[[t[i],y[i]] for i in indexs]
    

#    appRank = {'value':ty,'calibration':tz}
    appRank = {'value':ty}
    return JsonResponse(appRank)



def formulaTJ(request):
    max_points=500000
    beginDate = request.GET.get("beginDate", "2018-01-22")
    endDate = request.GET.get("endDate")
    
    endDate_=datetime.datetime.strptime(endDate, "%Y-%m-%d")
    endDate_=endDate_+timedelta(days=1)
    endDate=endDate_
    
    color=request.GET.get("color")
    uid=request.GET.get("uid")

    select=AlgorithmRgb.objects.filter(Uid=uid)
    select=select.filter(Datetime__range=[beginDate, endDate])
    select.all().order_by("Datetime")
#    t=map(lambda x:x.Datetime,select)
    t=map(lambda x:int(1000*x.Timestamp),select)
    if color=='Z1':
        yy=select.values('R1','G1','B1')
        y=map(lambda x:-0.68202*x['R1']+0.77073*x['G1']+0.56332*x['B1'],yy)
        y_raw=map(lambda x:x['R1'],yy)
    n=len(y)
    if n==0:
        return []
    interval=int(np.ceil(n/float(max_points)))
    indexs=range(0,n,interval)

    print(endDate,color,uid,interval)
    
#    today=t[-1]
#    end=int(1000*time.mktime(endDate_.timetuple()))
#    future_time=range(today,end,24*3600*1000)
#    ret=future_pred(t,y,future_time)
#    tz=[[future_time[i],ret[i]] for i in range(len(future_time))]
    
    ty=[[t[i],y[i]] for i in indexs]
    ty_raw=[[t[i],y_raw[i]] for i in indexs]

    appRank = {'transform':ty,'origin':ty_raw}
#    appRank = {'value':ty}
    return JsonResponse(appRank)

if __name__=='__main__':
    pass
#    df=pull_data(id1)
#    df2=pull_data(id1)