import pandas as pd
import numpy as np
import argparse
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import save_model, model_from_json, Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow import keras
from keras_tuner.engine.hyperparameters import HyperParameters
from typing import Union
import warnings
warnings.simplefilter("ignore", UserWarning)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

parser = argparse.ArgumentParser(description='Time Series Forecasting With LSTMS')
parser.add_argument('--lookback', type=int, required=False, default=10, help='past time step to look into')
parser.add_argument('--future', type=int, required=False, default=1, help='time steps to predict in future')
args = parser.parse_args()
lookback = args.lookback
pred_len = args.future
identifier = 'LSTM_Hourly_lookback-{}_future-{}'.format(lookback, pred_len)

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
    Creates a sliding window pattern from the data given and window length given.
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

def model():

    model = keras.Sequential()
    model.add(LSTM(100, activation='relu', return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(100, activation='relu'))
    model.add(Dense(pred_len))

    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0001),
                  loss='mse')

    return model

def show_best_hyperparamters(best_hps):
    print('Best Hyper Parameters\n')
    print('Layer 1 neuron: ', best_hps.get('first_layer_neurons'))
    print('Layer 2 neuron: ', best_hps.get('second_layer_neurons'))
    print('Activation Function: ', best_hps.get('activationfunc'))
    print('learning_rate: ', best_hps.get('learning_rate'))
    print('Dropout rate: ', best_hps.get('dropout'))


def calculate_metrics(test: np.ndarray, predict: np.ndarray, identifier: str) -> float:
    """."""
    RMSE = mean_squared_error(test.flatten(), predict.flatten(), squared=False)
    MSE = mean_squared_error(test.flatten(), predict.flatten())
    MAE = mean_absolute_error(test.flatten(), predict.flatten())
    MAPE = mean_absolute_percentage_error(test.flatten(), predict.flatten())
    r2 = r2_score(test.flatten(), predict.flatten())
    print('mse: {}, mae: {}, rmse: {}, mape: {}, R2: {}'.format(MSE, MAE, RMSE, MAPE, r2))
    f = open("result.txt", 'a')
    f.write(identifier + "  \n")
    f.write('mse: {}, mae: {}, rmse: {}, mape: {}, R2: {}'.format(MSE, MAE, RMSE, MAPE, r2))
    f.write('\n')
    f.write('\n')
    f.close()



df = pd.read_csv('../../data/cleaned_data.csv')
data = df['close'].values
data = data.reshape(-1,1)


train, test = data_divider(data, 0.8)
val, test = data_divider(test, 0.5)
scaler, train_normalized, val_normalized, test_normalized = min_max_scale(train, val, test)
x_train, y_train = sliding_window(train_normalized, lookback, pred_len)
x_val, y_val = sliding_window(val_normalized, lookback, pred_len)
x_test, y_test = sliding_window(test_normalized, lookback, pred_len)


# Build the model with the best hp.
# model = model_builder(best_hps)
model = model()

# stop_training_early = keras.callbacks.EarlyStopping()
stop_training_early = keras.callbacks.EarlyStopping(monitor="val_loss", patience=3)
history = model.fit(x_train, y_train, epochs=1 , verbose=1, shuffle=False, validation_data=(x_val, y_val),
                    callbacks=[stop_training_early])


y_predict = model.predict(x_test)
calculate_metrics(y_test, y_predict, identifier)