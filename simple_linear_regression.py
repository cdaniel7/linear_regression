#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 25 23:27:20 2019

@author: christopherdevairakkam
"""

# Data Preprocessing
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os 
os.chdir('/Users/christopherdevairakkam/Desktop/')

# Importing the dataset
dataset = pd.read_csv('Salary_Data.csv')

X=dataset.iloc[:,:1]
y=dataset.iloc[:,1]

# train test split  
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test =train_test_split(X,y,test_size=1/3,random_state=0)

# Simple Linear Regression will take care of feature scaling therefore not implementing it 
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,y_train)

# predict the Test set results 

y_pred = regressor.predict(X_test)

# visualization of training set
plt.scatter(X_train,y_train,color='red')
plt.plot(X_train,regressor.predict(X_train))
plt.title("Salary Prediction")
plt.xlabel("Years of Exp")
plt.ylabel("Salary")
plt.show()


# visualization of test set
plt.scatter(X_test,y_test,color='red')
plt.plot(X_train,regressor.predict(X_train))
plt.title("Salary Prediction")
plt.xlabel("Years of Exp")
plt.ylabel("Salary")
plt.show()


