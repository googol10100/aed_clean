#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  5 17:10:39 2018

@author: ly
"""


import time
from datetime import datetime
from datetime import timedelta

import sqlite3

def timestamp2beijing(t):
    time2 = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(t))
    return time2


db_path='my_api/db.sqlite3'
table_name='algorithm_rgb'
id1='868994037706145'
conn = sqlite3.connect(db_path)
conn2 = sqlite3.connect(db_path)

cur = conn.cursor()
cur2 = conn.cursor()
#sql_command='SELECT * FROM '+table_name+' WHERE Uid='+id1
#sql_command2='SELECT * FROM '+table_name

for i in range(100000):
    search_command="SELECT Timestamp FROM %s WHERE Id=%d" % (table_name,i)
    cur2.execute(search_command)
    t=cur2.fetchall()
    if not t:
        continue
    t=t[0][0]
    date=timestamp2beijing(t)
    date = datetime.strptime(date,"%Y-%m-%d %H:%M:%S")
#    sql_command='UPDATE %s SET R1=%d WHERE Id=%d'%(table_name,1000000+i,i)
    sql_command='UPDATE %s SET Datetime="%s" WHERE Id=%d'%(table_name,date,i)
    cur.execute(sql_command)
conn.commit()


