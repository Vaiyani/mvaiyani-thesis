import pandas as pd
import math
import matplotlib.pyplot as plt
import keras_tuner as kt
import plotly.express as px
import numpy as np
import tensorflow as tf
import plotly.graph_objects as go
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import save_model, model_from_json, Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow import keras
from keras_tuner.tuners import RandomSearch
from keras.models import load_model
from plotly.subplots import make_subplots
from keras_tuner.engine.hyperparameters import HyperParameters
from typing import Union
import warnings
warnings.simplefilter("ignore", UserWarning)

df = pd.read_csv('cleaned_data.csv')
data = df['close'].values
data = data.reshape(-1,1)


def min_max_scale(train: np.array, val: np.array, test: np.array) -> Union[MinMaxScaler, np.array, np.array, np.array]:
    """ Tranform the train and test data into min max scale of train data"""
    scaler = MinMaxScaler()
    scaler = scaler.fit(train)
    train_normalized = scaler.transform(train)
    test_normalized = scaler.transform(test)
    val_normalized = scaler.transform(val)
    return scaler, train_normalized, val_normalized, test_normalized


def data_divider(data: np.array, threshold: int):
    """ This functions divideds the data (close price) into 80 20 ration for test and train data """
    train_test_divider = int(len(data) * threshold)
    training_data, testing_data = data[:train_test_divider], data[train_test_divider:]
    return training_data, testing_data


def sliding_window(data: [], window_length: int, pred_len: int = 1) -> Union[np.array, np.array]:
    """
    This function creates a sliding window pattern from the data given and window length given.
    For example:
    Data = [[1],[2],[3],[4],[5],[6]]
    sliding window = 2
    pred_len = 1
    X = [[[1],[2]],[[2],[3]],[[3],[4]],[[4],[5]]]
    Y = [[3],[4],[5],[6]]

    """
    X = []
    Y = []
    for i in range(len(data) - window_length):
        input_end = i + window_length
        output_end = input_end + pred_len
        if output_end > len(data):
            break
        X.append(data[i: input_end])
        Y.append(data[input_end: output_end])

    return np.array(X), np.array(Y)


def model_builder(hp):
    """ Keras hyper paramter tuner model builder"""
    hp_dropout = hp.Choice('dropout', values=[0.05, 0.1, 0.2])
    model = Sequential()
    model.add(LSTM(hp.Int('first_layer_neurons', min_value=32, max_value=512, step=32), return_sequences=True))
    for i in range(hp.Int('n_layers', 1, 4)):
        model.add(LSTM(hp.Int(f'lstm_{i}_units', min_value=32, max_value=512, step=32), return_sequences=True))
    model.add(Dropout(hp_dropout))
    model.add(LSTM(hp.Int('second_layer_neurons', min_value=32, max_value=512, step=32)))
    model.add(Dense(1))
    hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])

    model.compile(optimizer=keras.optimizers.Adam(learning_rate=hp_learning_rate),
                  loss='mse',
                  metrics='mse')

    return model


def show_best_hyperparamters(best_hps):
    print('Best Hyper Parameters\n')
    print('Layer 1 neuron: ', best_hps.get('first_layer_neurons'))
    print('Layer 2 neuron: ', best_hps.get('second_layer_neurons'))
    print('n_layers: ', best_hps.get('n_layer'))
    print('learning_rate: ', best_hps.get('learning_rate'))
    print('Dropout rate: ', best_hps.get('dropout'))


def calculate_metrics(test: np.ndarray, predict: np.ndarray) -> float:
    """."""
    RMSE = mean_squared_error(test.flatten(), predict.flatten(), squared=False)
    MSE = mean_squared_error(test.flatten(), predict.flatten())
    MAE = mean_absolute_error(test.flatten(), predict.flatten())
    MAPE = mean_absolute_percentage_error(test.flatten(), predict.flatten())
    r2 = r2_score(test.flatten(), predict.flatten())
    print('mse: {}, mae: {}, rmse: {}, mape: {}, R2: {}'.format(MSE, MAE, RMSE, MAPE, r2))


lookback = 48
pred_len = 1
train, test = data_divider(data, 0.8)
val, test = data_divider(test, 0.5)
scaler, train_normalized, val_normalized, test_normalized = min_max_scale(train, val, test)
x_train, y_train = sliding_window(train_normalized, lookback, pred_len)
x_val, y_val = sliding_window(val_normalized, lookback, pred_len)
x_test, y_test = sliding_window(test_normalized, lookback, pred_len)


tuner = kt.RandomSearch(model_builder, objective='val_mse', max_trials=30, directory='my_dir', project_name='intro_to_kt2')
stop_early = keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)
tuner.search(x_train, y_train, epochs=100, validation_data=(x_val,y_val), callbacks=[stop_early])

best_hps = tuner.get_best_hyperparameters()[0]
show_best_hyperparamters(best_hps)