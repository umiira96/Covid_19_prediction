# -*- coding: utf-8 -*-
"""
Created on Sun May 22 08:13:19 2022

@author: umium
"""

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt

class ModelCreation():
    
    def LSTM_layers(self,x_train):
        model = Sequential()
        model.add(LSTM(64, activation='tanh',
                       return_sequences=True,
                       input_shape=(x_train.shape[1],1))) 
        model.add(Dropout(0.2))
        model.add(LSTM(64))
        model.add(Dropout(0.2))
        model.add(Dense(1, activation='relu'))
        return model


class ModelEvaluation():
    
    def mape(self,y_test,y_predicted):
        MAPE = mean_absolute_error(y_test,y_predicted)/sum(y_predicted)*100
        print('Mean Absolute Percentage Error is', "%.4f" % MAPE,'%')

    def evaluation_plot(self,y_test,y_predicted):
        plt.figure()
        plt.plot(y_test, color='r', label='Actual Cases')
        plt.plot(y_predicted, color='b', label='Predicted Cases')
        plt.legend(['Actual Cases','Predicted Cases'])
        plt.show()


