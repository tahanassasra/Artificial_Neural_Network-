# -*- coding: utf-8 -*-
"""
Created on Fri May  6 13:44:51 2022

@author: t90na
"""

import numpy as np
import pandas as pd
import tensorflow as tf

# READING THE DATASET FILE

dataset = pd.read_csv('Churn_Modelling.csv')
x = dataset.iloc[:,3:-1]   #eliminate the first 3 columns as they are irrelevent and the last column as it is the target column
y = dataset.iloc[:,-1]   #assign target column for y

#Label Encoder for Gender column

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
x.iloc[:,2] = le.fit_transform(x.iloc[:,2])

#OneHotEncoder for the country column

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers = [('encoder', OneHotEncoder(),[1])], remainder='passthrough')
x = ct.fit_transform(x)

#Split training and testing sets

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2, random_state=0)

#FeatureScaling

from sklearn.preprocessing import StandardScaler
sc =StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.fit_transform(x_test)

# Building the ANN

ann = tf.keras.models.Sequential()    #initializing the neural network
ann.add(tf.keras.layers.Dense(units=6, activation='relu'))  #adding input layer
ann.add(tf.keras.layers.Dense(units=6, activation='relu'))  #adding 1st hidden layer
ann.add(tf.keras.layers.Dense(units=1, activation='sigmoid')) #adding output layer


#compiling the ANN

ann.compile(optimizer='adam', loss='binary_crossentropy', metrics = ['accuracy'])

#training the ANN

ann.fit(x_train, y_train, batch_size=32, epochs=100)

#make a prediction
myPrediction = ann.predict(sc.transform([[1,0,0,600,1,40,3,60000,2,1,1,50000]]))


#predicting Test Set Results

y_pred = np.array(ann.predict(x_test))
y_pred = (y_pred>0.5)
#print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))
print(np.concatenate((y_pred.reshape(len(y_pred),1), np.array(y_test).reshape(len(y_test),1)),1))

# making confusion matrix

from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
accuracy_score(y_test, y_pred)




