#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 29 10:03:33 2018

@author: ly
"""

import numpy as np
import sqlite3
import pandas as pd

db_path='data/db.sqlite3'
table_name='algorithm_rgb'
id1='868994037706145'
conn = sqlite3.connect(db_path)

cur = conn.cursor()
sql_command='SELECT * FROM '+table_name+' WHERE Uid='+id1
#cur.execute(sql_command)
#res = cur.fetchall()

resp = pd.read_sql_query(sql_command, conn)