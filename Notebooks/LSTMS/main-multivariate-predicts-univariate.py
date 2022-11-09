import pandas as pd
import numpy as np
import argparse
import random
import matplotlib.pyplot as plt
import os

parser = argparse.ArgumentParser(description='Time Series Forecasting With LSTMS')
parser.add_argument('--lookback', type=int, required=False, default=24, help='past time step to look into')
parser.add_argument('--future', type=int, required=False, default=12, help='time steps to predict in future')
parser.add_argument('--gpu', action='store_true', help='gpu visible')
parser.add_argument('--reduce_complexity', action='store_true', help='gpu visible')
args = parser.parse_args()
lookback = args.lookback
pred_len = args.future
gpu = args.gpu
reduce_complexity = args.reduce_complexity
print(args)
if gpu:
    os.environ["CUDA_VISIBLE_DEVICES"] = "4"
    activation = 'tanh'
else:
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    activation = 'relu'
identifier = 'LSTM_Hourly_lookback-{}_future-{}'.format(lookback, pred_len)
print(identifier)
print("lookback {} future {} activation {} gpu {}".format(lookback, pred_len, activation, gpu))

import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, mean_absolute_error, r2_score
# from tensorflow.keras.models import save_model, model_from_json, Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow import keras
from typing import Union

fix_seed = 0
tf.random.set_seed(fix_seed)
np.random.seed(fix_seed)

# isExist = os.path.exists('./saved_models')
# if not isExist:
#   os.makedirs('./saved_models')

def min_max_scale(train: np.array, val: np.array, test: np.array) -> Union[StandardScaler, np.array, np.array, np.array]:
    """ Tranform the train and test data into min max scale of train data"""
    scaler = StandardScaler()
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


def sliding_window(data: pd.DataFrame, window_length: int, pred_len: int = 1) -> Union[np.array, np.array]:
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
        #         print(data[i: output_end+1, :])
        X.append(data[i: input_end, :])
        Y.append(data[input_end: output_end, -1:])

    return np.array(X), np.array(Y)

def model():

    model = keras.Sequential()
    model.add(LSTM(100, activation=activation, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(100, activation=activation))
    model.add(Dense(pred_len))

    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0001),
                  loss='mse')

    return model

def model2():

    model = keras.Sequential()
    model.add(LSTM(50, activation=activation))
    model.add(Dropout(0.3))
    model.add(Dense(pred_len))
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0001),
                  loss='mse')
    return model

# def show_best_hyperparamters(best_hps):
#     print('Best Hyper Parameters\n')
#     print('Layer 1 neuron: ', best_hps.get('first_layer_neurons'))
#     print('Layer 2 neuron: ', best_hps.get('second_layer_neurons'))
#     print('Activation Function: ', best_hps.get('activationfunc'))
#     print('learning_rate: ', best_hps.get('learning_rate'))
#     print('Dropout rate: ', best_hps.get('dropout'))


def calculate_metrics(test: np.ndarray, predict: np.ndarray, identifier: str) -> float:
    """."""
    RMSE = mean_squared_error(test.flatten(), predict.flatten(), squared=False)
    MSE = mean_squared_error(test.flatten(), predict.flatten())
    MAE = mean_absolute_error(test.flatten(), predict.flatten())
    MAPE = mean_absolute_percentage_error(test.flatten(), predict.flatten())
    r2 = r2_score(test.flatten(), predict.flatten())
    print('mse: {}, mae: {}, rmse: {}, mape: {}, R2: {}'.format(MSE, MAE, RMSE, MAPE, r2))
    f = open("result-lstm-multivariate-more-feature.txt", 'a')
    f.write(identifier + "  \n")
    f.write('mse: {}, mae: {}, rmse: {}, mape: {}, R2: {}'.format(MSE, MAE, RMSE, MAPE, r2))
    f.write('\n')
    f.write('\n')
    f.close()



df = pd.read_csv('./data/multivariate.csv')
df = df.drop(['date'], axis = 1)
len_ = len(df)
df_train = df.iloc[:int(len_*0.8), :]
df_val = df.iloc[int(len_*0.8):int(len_*0.9),:].reset_index(drop=True)
df_test = df.iloc[int(len_*0.9):,:].reset_index(drop=True)

scaler, train_normalized, val_normalized, test_normalized = min_max_scale(df_train, df_val, df_test)

x_train, y_train = sliding_window(train_normalized, lookback, pred_len)
x_val, y_val = sliding_window(val_normalized, lookback, pred_len)
x_test, y_test = sliding_window(test_normalized, lookback, pred_len)

print("train :",x_train.shape)
print("Val :", x_val.shape)
print("Test :", x_test.shape)

# Build the model with the best hp.
# model = model_builder(best_hps)
if reduce_complexity:
    model = model2()
else:
    model = model()


# stop_training_early = keras.callbacks.EarlyStopping()
stop_training_early = keras.callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
history = model.fit(x_train, y_train, epochs=60 , verbose=1, shuffle=False, validation_data=(x_val, y_val),
                    callbacks=[stop_training_early])


## predict model and save result
# model.save('saved_models/' + identifier)
y_predict = model.predict(x_test)
calculate_metrics(y_test, y_predict, identifier)


plt.plot(history.history['loss'], label='train loss')
plt.plot(history.history['val_loss'], label='validation loss')
plt.legend()
plt.savefig('fig/'+'loss_'+'lookback-'+str(lookback)+'future-'+str(pred_len)+'.png')
# assert  1==1
# plt.plot(y_test.flatten(), label='test')
# plt.plot(y_predict.flatten(), 'r-', label='predict')
# plt.legend()
# plt.show()