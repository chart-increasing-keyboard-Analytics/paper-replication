# -*- coding: utf-8 -*-
"""
Created on Mon Oct 15 14:33:31 2018

@author: andrei
"""


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from keras.layers import Input, Dense, Dropout
from keras.models import Model
from keras.optimizers import SGD
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from keras import backend as K
from sklearn.linear_model import LinearRegression
import seaborn as sns
import matplotlib.pyplot as plt

def rmse_metric(y_true, y_pred):
   return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1)) 

def r2_metric(y_true, y_pred):
    SS_res =  K.sum(K.square(y_true - y_pred)) 
    SS_tot = K.sum(K.square(y_true - K.mean(y_true))) 
    return ( 1 - SS_res/(SS_tot + K.epsilon()) )

def linear_model(feats_size):
    X_input =  Input(shape=(feats_size,), name = 'Input')
    X = Dense(1, activation = 'linear', name='Output', 
              kernel_initializer='glorot_uniform')(X_input)
    model =  Model(inputs = X_input, outputs = X, name = "linear_reg")
    sgd = SGD(lr=0.005)
    model.compile(loss= rmse_metric, optimizer=sgd, 
                  metrics = [r2_metric])
    return model


def nn_model(feats_size):
    X_input =  Input(shape=(feats_size,), name = 'Input')
    X = Dense(units = feats_size, activation = 'relu', 
              name = 'Hidden_1')(X_input)
    X = Dropout(rate = 0.5, name = "Dropout_half")(X)
    X = Dense(units = feats_size // 2, activation = 'relu', 
              name = 'Hidden_2')(X)
    X = Dense(units = 1, name = 'Output')(X)
    model = Model(inputs = X_input, outputs = X, name = "FC_32_16_1")
    model.compile(optimizer = "adam", loss = rmse_metric,  
                  metrics = [r2_metric])
    return model

def plot_btime_hist(btime):
    sns.set()
    btime_insec = [elem / 10**3 for elem in btime]
    plot = sns.distplot(btime_insec, hist=True, kde=False, 
                        bins=50, color = 'blue', hist_kws={'edgecolor':'black'})
    plot.set_yscale('log')
    plot.set_title("Histogram of build time")
    plot.set(xlabel = "Build time(s)", ylabel="Count")
    figure = plot.get_figure()
    figure.savefig("build_time.png", dpi = 400)
    plt.close()
    
def plot_training_evol(metric_h, plt_title, x_label, y_label,
                       fig_title, linecolor):
    sns.set()
    epochs = [i for i in range(1, len(metric_h) + 1)]
    plt.xticks(epochs, [str(i) for i in epochs])
    plt.title(plt_title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.plot(epochs, metric_h, linestyle='--', marker='o', color=linecolor)
    plt.savefig(fig_title, dpi=400)
    plt.close()
    
   
if __name__ == "__main__":

    df = pd.read_csv("build_time1m.csv", low_memory=False)
    nans = df.isna().sum()
    nans = nans[nans > 0]
    
    df.dropna(subset=['tr_duration'], inplace = True)
    
    df.fillna(value = 0, inplace=True)
    df['tr_jobs'] = df['tr_jobs'].apply(lambda x: len(x))
    df['tr_log_bool_tests_failed'] = df['tr_log_bool_tests_failed'].astype("int64")
    df[df.dtypes[df.dtypes == bool].index.values] = \
        df[df.dtypes[df.dtypes == bool].index.values].astype("int64")
    
    df.describe().to_excel("df_stats.xlsx")
    
    X = df.loc[:, df.columns != 'tr_duration'].values
    y = df['tr_duration'].values
    
    data_scaler = MinMaxScaler(feature_range=(0, 32))
    X = data_scaler.fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    
    model = nn_model(X_train.shape[1])
    #model = linear_model(X_train.shape[1])
    history_callback = model.fit(x = X_train, y = y_train,
                                 batch_size = 128, epochs=20)
    
    plot_btime_hist(y)
    loss = [elem / 10**3 for elem in history_callback.history['loss']]
    plot_training_evol(loss, 
                       plt_title = "RMSE during training",
                       x_label = "Epoch", y_label = "RMSE",
                       fig_title = "rmse.png", linecolor="r")
    plot_training_evol(history_callback.history['r2_metric'], 
                       plt_title = "R-squared during training",
                       x_label = "Epoch", y_label = "R-squared",
                       fig_title = "r2.png", linecolor="b")
    
    
    
    
    
    
    
    
    
    
    
    
    
