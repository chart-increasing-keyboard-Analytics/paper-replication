# -*- coding: utf-8 -*-
"""
Created on Wed Oct 24 16:52:13 2018

@author: andrei
"""

from logger import Logger
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import LinearSVR
from time import sleep
from scipy.stats import pearsonr
import xgboost as xgb
import pandas as pd
import numpy as np
from keras.layers import Input, Dense, Dropout
from keras.models import Model
from keras.optimizers import SGD
from keras import backend as K
from keras.callbacks import EarlyStopping, TerminateOnNaN, ModelCheckpoint
#from keras.models import load_model
import matplotlib.pyplot as plt
import seaborn as sns
import json


def plot_training_evol(metric_history, plt_title, x_label, y_label, fig_title, 
                       linecolor):
    sns.set()
    epochs = [i for i in range(1, len(metric_history) + 1)]
    plt.xticks(epochs, [str(i) for i in epochs])
    plt.title(plt_title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.plot(epochs, metric_history, linestyle='--', marker='o', 
             color=linecolor)
    plt.savefig(fig_title, dpi=400)
    plt.close()
    
def load_config_data(config_file = "config.txt"):
    with open(config_file, 'r') as fp:
        config_data = json.load(fp)
    return config_data


def calculate_correlation(df, features_names, target_name):
    correlations = {}
    for feat_name in features_names:
        x1 = df[feat_name].values
        x2 = df[target_name].values
        key = feat_name + ' vs ' + target_name
        correlations[key] = pearsonr(x1,x2)[0]
    data_correlations = pd.DataFrame(correlations, index=['Value']).T
    data_correlations.loc[data_correlations['Value'].abs().sort_values(
            ascending=False).index]
    return data_correlations

def evaluate_model(model, X, y):
    
    preds = model.predict(X)
    rmse = np.sqrt(mean_squared_error(y, preds))
    r2 = r2_score(y, preds)
    
    return rmse, r2


def test_knn(logger, X, y):
    _, new_X, _, new_y = train_test_split(X, y, test_size=0.068)
    
    X_train, X_test, y_train, y_test = train_test_split(new_X, new_y, 
                                                        test_size=0.2)
    
    model = KNeighborsRegressor(5, weights='distance')
    logger.log("Start training KNN model...")
    model.fit(X_train, y_train)
    logger.log("Finish training KNN model", show_time = True)
    
    logger.log("Start evaluating training performance...", tabs = 1)
    rmse, r2 = evaluate_model(model, X_train, y_train)
    logger.log("KNN training on {} result RMSE:{:.3f} / R2:{:.2f}".format(
            X_train.shape[0], rmse / 10**3, r2), tabs = 1, show_time = True)
    logger.log("Start evaluating testing performance...", tabs = 1)
    rmse, r2 = evaluate_model(model, X_test, y_test)
    logger.log("KNN testing on {} result RMSE:{:.3f} / R2:{:.2f}".format(
             X_test.shape[0], rmse / 10**3, r2), tabs = 1, show_time = True)
    

def test_xgboost(logger, X, y):
     _, new_X, _, new_y = train_test_split(X, y, test_size=0.068)
     
     X_train, X_test, y_train, y_test = train_test_split(new_X, new_y, 
                                                        test_size=0.2)
     
     model = xgb.XGBRegressor(objective ='reg:linear', colsample_bytree = 0.3,
                               learning_rate = 0.1, max_depth = 50, alpha = 10, 
                               n_estimators = 100)
     
     logger.log("Start training XGBoost model...")
     model.fit(X_train,y_train)
     logger.log("Finish training XGBoost model", show_time = True)
     
     logger.log("Start evaluating training performance...", tabs = 1)
     rmse, r2 = evaluate_model(model, X_train, y_train)
     logger.log("XGBoost training on {} result RMSE:{:.3f} / R2:{:.2f}".format(
            X_train.shape[0], rmse / 10**3, r2), tabs = 1, show_time = True)
     logger.log("Start evaluating testing performance...", tabs = 1)
     rmse, r2 = evaluate_model(model, X_test, y_test)
     logger.log("XGBoost testing on {} result RMSE:{:.3f} / R2:{:.2f}".format(
             X_test.shape[0], rmse / 10**3, r2), tabs = 1, show_time = True)
     

def test_svm(logger, X, y):
    
     _, new_X, _, new_y = train_test_split(X, y, test_size=0.068)
     
     X_train, X_test, y_train, y_test = train_test_split(new_X, new_y, 
                                                        test_size=0.2)
     
     model = LinearSVR(loss = 'squared_epsilon_insensitive', max_iter = 20000, 
                       C = 0.1)
     #SVR(kernel='linear')
     
     logger.log("Start training SVM model...")
     model.fit(X_train,y_train)
     logger.log("Finish training SVM model", show_time = True)
     
     logger.log("Start evaluating training performance...", tabs = 1)
     rmse, r2 = evaluate_model(model, X_train, y_train)
     logger.log("SVM training on {} result RMSE:{:.3f} / R2:{:.2f}".format(
            X_train.shape[0], rmse / 10**3, r2), tabs = 1, show_time = True)
     logger.log("Start evaluating testing performance...", tabs = 1)
     rmse, r2 = evaluate_model(model, X_test, y_test)
     logger.log("SVM testing on {} result RMSE:{:.3f} / R2:{:.2f}".format(
             X_test.shape[0], rmse / 10**3, r2), tabs = 1, show_time = True)
     

def rmse_metric(y_true, y_pred):
   return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1)) 

def r2_metric(y_true, y_pred):
    SS_res =  K.sum(K.square(y_true - y_pred)) 
    SS_tot = K.sum(K.square(y_true - K.mean(y_true))) 
    return ( 1 - SS_res/(SS_tot + K.epsilon()) )
     
def test_linear_regression(logger, X, y):
    
    _, new_X, _, new_y = train_test_split(X, y, test_size=0.068)
     
    X_train, X_test, y_train, y_test = train_test_split(new_X, new_y, 
                                                        test_size=0.2)
    
    X_input =  Input(shape=(X.shape[1],), name = 'Input')
    X = Dense(1, activation = 'linear', name='Output')(X_input)
    model =  Model(inputs = X_input, outputs = X, name = "LR_SGD")
    sgd = SGD(lr=0.04)
    model.compile(loss= rmse_metric, optimizer=sgd, metrics = [r2_metric])
    
    callbacks = []
    callbacks.append(EarlyStopping(monitor="loss", min_delta=0.5, 
                                   patience=3))
    callbacks.append(TerminateOnNaN())
    history_callback = model.fit(x = X_train, y = y_train, batch_size = 32, 
                                 epochs = 50, callbacks = callbacks)
    
    loss = [elem / 10**3 for elem in history_callback.history['loss']]
    plot_training_evol(loss, plt_title = "RMSE during training",
                       x_label = "Epoch", y_label = "RMSE",
                       fig_title = logger.get_output_file("lr_rmse.png"), 
                       linecolor="r")
    plot_training_evol(history_callback.history['r2_metric'], 
                       plt_title = "R-squared during training",
                       x_label = "Epoch", y_label = "R-squared",
                       fig_title = logger.get_output_file("lr_r2.png"), 
                       linecolor="b")


def test_neural_network(logger, X, y):
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    
    X_input =  Input(shape=(X.shape[1] * 2,), name = 'Input')
    X = Dense(units = X_train.shape[1], activation = 'relu', 
              name = 'Hidden_1')(X_input)
    X = Dense(units = X_train.shape[1], activation = 'relu', 
              name = 'Hidden_2')(X)
    X = Dropout(rate = 0.5, name = "Dropout_half")(X)
    X = Dense(units = X_train.shape[1] // 2, activation = 'relu', 
              name = 'Hidden_3')(X)
    X = Dense(units = 1, name = 'Output')(X)
    model = Model(inputs = X_input, outputs = X, name = "FC_FN_halfFN_1")
    model.compile(optimizer = "adam", loss = rmse_metric, metrics = [r2_metric])
    model.summary()
    
    
    callbacks = []
    callbacks.append(EarlyStopping(monitor="loss", min_delta=0.5, 
                                   patience=3))
    callbacks.append(TerminateOnNaN())
    file_path =  model.name + "_E{epoch:02d}_L{" + "loss" + ":.6f}" + ".h5"
    file_path = logger.get_model_file(file_path)
    callbacks.append(ModelCheckpoint(filepath = file_path, monitor = "loss"))
    history_callback = model.fit(x = X_train, y = y_train, batch_size = 32, 
                                 epochs = 50, callbacks = callbacks)
    
    loss = [elem / 10**3 for elem in history_callback.history['loss']]
    plot_training_evol(loss, plt_title = "RMSE during training",
                       x_label = "Epoch", y_label = "RMSE",
                       fig_title = logger.get_output_file("nn_rmse.png"), 
                       linecolor="r")
    plot_training_evol(history_callback.history['r2_metric'], 
                       plt_title = "R-squared during training",
                       x_label = "Epoch", y_label = "R-squared",
                       fig_title = logger.get_output_file("nn_r2.png"), 
                       linecolor="b")
    
    
if __name__ == "__main__":
    
    logger = Logger(show = True, html_output = True)
    config_data = load_config_data()
    
    input_file = logger.get_data_file(config_data['input_file'])
    logger.log("Start reading build time file {}...".format(config_data['input_file']))
    df = pd.read_csv(input_file, low_memory = False)
    logger.log("Finish reading build time file", show_time = True)
    
    colnames_file = logger.get_data_file(config_data['colnames_file'])
    selected_cols = np.loadtxt(colnames_file, dtype = str)
    logger.log("Start selecting columns...")
    df = df[selected_cols]
    logger.log("Finish selecting columns", show_time = True)
    
    nans = df.isna().sum()
    nans = nans[nans > 0]
    np.save(logger.get_output_file("nans"), list(zip(nans.index, nans.values)))
    logger.log("Finish computing and saving Nans", show_time = True)
    
    df.dropna(subset=['tr_duration'], inplace = True)
    df.drop(nans[nans > 0.8 * df.shape[0]].index.tolist(), axis = 1, inplace=True)
    logger.log("Finished dropping Nans", show_time = True)
    
    df.describe().to_excel(logger.get_output_file("df_stats.xlsx"))
    logger.log("Finished computing and saving stats", show_time = True)
    
    df.fillna(value = 0, inplace=True)
    logger.log("Finished zero-filling Nans", show_time = True)
    
    df['tr_jobs'] = df['tr_jobs'].apply(lambda x: len(x))
    logger.log("Finished convert tr_jobs from job list to number of jobs", 
               show_time = True)
    
    remaining_cols = df.columns.values.tolist()
    corr_df = calculate_correlation(df, remaining_cols[:-1], remaining_cols[-1])
    corr_df.to_csv(logger.get_output_file("features_corr_with_target.csv"))
    logger.log("Finished computing and saving correlation", show_time = True)
    
    df['tr_log_bool_tests_failed'] = df['tr_log_bool_tests_failed'].astype("int64")
    df[df.dtypes[df.dtypes == bool].index.values] = \
        df[df.dtypes[df.dtypes == bool].index.values].astype("int64")
    logger.log("Finish converting bool values", show_time = True)
    
    X = df.loc[:, df.columns != 'tr_duration'].values
    y = df['tr_duration'].values
    
    data_scaler = MinMaxScaler(feature_range=(0, 28))
    X = data_scaler.fit_transform(X)
    
    '''
    test_knn(logger, X, y)
    sleep(1)
    test_xgboost(logger, X, y)
    sleep(1)
    test_svm(logger, X, y)
    sleep(1)
    test_linear_regression(logger, X, y)
    sleep(1)
    '''
    test_neural_network(logger, X, y)
    
    logger.close()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    