# -*- coding: utf-8 -*-
"""
Created on Mon May  6 19:55:23 2019

@author: Aditya Chondke
"""


from siftmatch import match_template1
import time

start=time.time()
match_template1("indiagater40.jpg","indiagate.png" , 5)
end=time.time()

print(end-start)