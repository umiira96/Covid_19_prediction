# -*- coding: utf-8 -*-
"""
Created on Fri May 20 09:39:24 2022

@author: umium
"""

import os
import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from sklearn.metrics import mean_absolute_error
from tensorflow.keras.callbacks import TensorBoard

TRAIN_CASES = os.path.join(os.getcwd(), 'cases_malaysia_train.csv')
TEST_CASES = os.path.join(os.getcwd(), 'cases_malaysia_test.csv')
PATH = os.path.join(os.getcwd(),'logs')

#%%EDA
#Step 1) Data loading
X_train = pd.read_csv(TRAIN_CASES)
X_test = pd.read_csv(TEST_CASES)

#%%
#Step 2) Data inspection
X_train.info() #there are nan values in cases_new
X_train.describe().T

X_test.info() #there are nan values in cases_new
X_test.describe().T

#%%
#Step 3) Data visualization
plt.figure()
plt.plot(X_train['cases_new'])
plt.show()

#%%
#Step 4) Data cleaning
X_test = X_test.dropna(subset=['cases_new'])

x_test = X_test['cases_new']
x_test.isnull().sum()

X_train = X_train.replace("?", "Nan")
X_train = X_train.replace(" ", "Nan")

x_train = X_train['cases_new'].values
x_test = X_test['cases_new'].values

x_train = pd.to_numeric(x_train, errors='coerce')

#%%
#Step 5) Features selection
#Step 6) Data preprocessing
mms = MinMaxScaler()
x_train_scaled = mms.fit_transform(np.expand_dims(x_train, -1))
x_test_scaled = mms.transform(np.expand_dims(x_test, -1))

#training dataset
x_train = []
y_train = []

window_size = 30 #30days
for i in range(window_size,len(x_train_scaled)):
    x_train.append(x_train_scaled[i-window_size:i,0])
    y_train.append(x_train_scaled[i,0])
    
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
    
x_test = np.array(x_test)
y_test = np.array(y_test)

x_train = np.expand_dims(x_train, axis=-1)
x_test = np.expand_dims(x_test, axis=-1)

#%% Model creation

model = Sequential()
model.add(LSTM(32, activation='tanh',
               return_sequences=True,
               input_shape=(x_train.shape[1],1))) 
model.add(Dropout(0.2))
model.add(LSTM(32))
model.add(Dropout(0.2))
model.add(Dense(1, activation='relu'))

model.compile(optimizer='adam',
              loss='mse',
              metrics='mse')

log_files = os.path.join(PATH,
                         datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))

tensorboard_callback = TensorBoard(log_dir=log_files, histogram_freq=1)

hist = model.fit(x_train,y_train, 
                 epochs=10, 
                 batch_size=128,
                 callbacks=(tensorboard_callback))
                 
print(hist.history.keys())

plt.figure()
plt.plot(hist.history['loss'])
plt.xlabel('epochs')
plt.ylabel('training loss')
plt.show()


#%% Model prediction

predicted = model.predict(np.expand_dims(x_test,axis=-1))
    
#%% 
y_true = y_test
y_predicted = predicted

print(mean_absolute_error(y_test,y_predicted)/sum(y_predicted)*100)

plt.figure()
plt.plot(y_test, color='r', label='Actual' )
plt.plot(y_predicted, color='b', label='Predicted')
plt.legend(['Actual','Predicted'])
plt.show()
    

