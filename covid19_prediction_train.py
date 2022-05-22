# -*- coding: utf-8 -*-
"""
Created on Sun May 22 17:02:08 2022

@author: umium
"""

import os
import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.callbacks import TensorBoard
from covid19_prediction_modules import ModelCreation, ModelEvaluation

#PATH
TRAIN_CASES = os.path.join(os.getcwd(), 'cases_malaysia_train.csv')
TEST_CASES = os.path.join(os.getcwd(), 'cases_malaysia_test.csv')
PATH = os.path.join(os.getcwd(),'logs')
MODEL_SAVE_PATH = os.path.join(os.getcwd(),'model.h5')

#constant
window_size = 30 #30days

#%%EDA
#Step 1) Data loading
X_train = pd.read_csv(TRAIN_CASES)
X_test = pd.read_csv(TEST_CASES)

#%%
#Step 2) Data inspection
#just focusing on case_new only
X_train.info() 
X_train.describe().T
X_train.duplicated().sum() 
#there are nan values in cases_new
#no duplicate values

X_test.info() #there are nan values in cases_new
X_test.describe().T
X_test.duplicated().sum() 
#there are funny values in cases_new
#no duplicate values

#%%
#Step 3) Data cleaning
#X_train data cleaning
#to replace ? and ' ' with nan
X_train = X_train.replace("?", "nan")
X_train = X_train.replace(" ", "nan")

#to convert string to int for column cases_new in X_train
X_train['cases_new'] = pd.to_numeric(X_train['cases_new'], errors='coerce')

#to drop row with nan values 
#drop nan values since the row doesnt effect much the trend of data
x_train = X_train.dropna(subset=['cases_new'])

#X_test data cleaning
#to drop nan value in x_test
#drop nan value since only one row for column cases_new that having the nan value
x_test = X_test.dropna(subset=['cases_new'])

#to select cases_new column only
x_train = x_train['cases_new']
x_test = x_test['cases_new']

#Data visualization
plt.figure()
plt.plot(x_train)
plt.show()

#%%
#Step 4) Features selection
#Step 5) Data preprocessing
mms = MinMaxScaler()
x_train_scaled = mms.fit_transform(np.expand_dims(x_train, -1))
x_test_scaled = mms.transform(np.expand_dims(x_test, -1))

#Data separation
#Training dataset
x_train = []
y_train = []

for i in range(window_size,len(x_train_scaled)):
    x_train.append(x_train_scaled[i-window_size:i,0])
    y_train.append(x_train_scaled[i,0])
    
#to return from list to array
x_train = np.array(x_train)
y_train = np.array(y_train)

#Testing dataset
temp = np.concatenate((x_train_scaled, x_test_scaled))
length_window = window_size+len(x_test_scaled)
temp = temp[-length_window:]

x_test = []
y_test = []
for i in range(window_size,len(temp)):
    x_test.append(temp[i-window_size:i,0])
    y_test.append(temp[i,0])
    
#to return from list to array
x_test = np.array(x_test)
y_test = np.array(y_test)

#expand from 2d to 3d
x_train = np.expand_dims(x_train, axis=-1)
x_test = np.expand_dims(x_test, axis=-1)

#%% Model creation
mc = ModelCreation()
model = mc.LSTM_layers(x_train)

model.compile(optimizer='adam',
              loss='mse',
              metrics='mse')

log_files = os.path.join(PATH,
                         datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))

tensorboard_callback = TensorBoard(log_dir=log_files, histogram_freq=1)

hist = model.fit(x_train,y_train, 
                 epochs=100, 
                 callbacks=(tensorboard_callback))
                 
print(hist.history.keys())

#%% Model evaluation
predicted = []
for i in x_test:
    predicted.append(model.predict(np.expand_dims(i,axis=0)))

#%% Model analysis
y_true = y_test
y_predicted = np.array(predicted).reshape(99,1)

me = ModelEvaluation()
me.mape(y_test,y_predicted)
me.evaluation_plot(y_test,y_predicted)

#%% Model deployment
model.save(MODEL_SAVE_PATH)

