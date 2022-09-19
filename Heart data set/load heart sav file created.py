# -*- coding: utf-8 -*-
"""
Created on Fri Jul 29 19:46:49 2022

@author: Subham
"""
import pickle
model_load=pickle.load(open("C:/Users/Subham/Desktop/Learn N Build/Machine learning algorithm by kris sir/Assignments to do/Heart data set/heart.sav",'rb'))
output_custom=model_load.predict([[34,0,0,0,1,1,1,1,1,23,1,1,0]])
