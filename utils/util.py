from __future__ import print_function

import torch
import numpy as np
import pymssql
import sqlite3
import os
from typing import Tuple


def local_db_backup_forgetting(insert_sql):
    conn = sqlite3.connect('./Quantization.db')
    cursor_obj = conn.cursor()

    cursor_obj.execute('''CREATE TABLE IF NOT EXISTS forgetting (dataset, quant, example, forgetting, param1, param2, param3)''')
    conn.commit()

    cursor_obj.execute(insert_sql)
    conn.commit()
    conn.close()
    
    
def insert_forgetting(dataset, quant, example, forgetting, param1, param2, param3):
    insert_sql = """INSERT into forgetting (dataset, quant, example, forgetting, param1, param2, param3) VALUES ('{}', '{}', '{}', '{}', '{}', '{}', '{}')""".format(dataset, quant, example, forgetting, param1, param2, param3)
    
    try:
        lines = []

        with open('./MSSQL.txt') as f:
            lines = f.read().splitlines()

        connSQL = pymssql.connect(server=lines[0], user=lines[1], password=lines[2], database=lines[3])
        cursorSQL = connSQL.cursor()
        cursorSQL.execute(insert_sql)
        connSQL.commit()
        connSQL.close()
    except Exception as e:
        # Local backup
        local_db_backup_forgetting(insert_sql)
        print("Results inserted in a local DB")
        

def local_db_backup_coreset(insert_sql):
    conn = sqlite3.connect('./Quantization.db')
    cursor_obj = conn.cursor()

    cursor_obj.execute('''CREATE TABLE IF NOT EXISTS coreset (dataset, pid, seed, core_size, core_used, quantization, accuracy, stream, param1, param2, param3)''')
    conn.commit()

    cursor_obj.execute(insert_sql)
    conn.commit()
    conn.close()
    
    
def insert_coreset(dataset, pid, seed, core_size, core_used, quantization, accuracy, stream, param1, param2, param3):
    insert_sql = """INSERT into coreset (dataset, pid, seed, core_size, core_used, quantization, accuracy, stream, param1, param2, param3) VALUES ('{}', '{}', '{}', '{}', '{}', '{}', '{}', '{}', '{}', '{}', '{}')""".format(dataset, pid, seed, core_size, core_used, quantization, accuracy, stream, param1, param2, param3)
    
    try:
        lines = []

        with open('./utils/MSSQL.txt') as f:
            lines = f.read().splitlines()

        connSQL = pymssql.connect(server=lines[0], user=lines[1], password=lines[2], database=lines[3])
        cursorSQL = connSQL.cursor()
        cursorSQL.execute(insert_sql)
        connSQL.commit()
        connSQL.close()
    except Exception as e:
        # Local backup
        local_db_backup_coreset(insert_sql)
        print("Results inserted in a local DB")
        print("Task: "+ str(seed) +", Accuracy: " + str(accuracy))