# -*- coding: utf-8 -*-
"""
Created on Mon Jul 29 15:50:42 2024

@author: piercetf
"""

import sqlite3
import pandas
import os

candidate_table = pandas.read_csv(, sep='\t')
data_table = pandas.read_csv(, sep='\t')

userprofile = os.environ['USERPROFILE']

conn = sqlite3.connect()