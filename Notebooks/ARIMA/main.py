import pmdarima as pm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
import statsmodels.api as sm
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from numpy import log
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf


threshold = -15
df = pd.read_csv('./cleaned_data.csv')
df_ = np.log(df['close'].values)
train = df_[:threshold]
test = df_[threshold:]
scaler = StandardScaler()
train = scaler.fit_transform(train.reshape(-1,1)).squeeze()
test = scaler.transform(test.reshape(-1,1)).squeeze()

# model = pm.auto_arima(train,
#                       start_p=10,
#                       start_q=10,
#                       test='adf',       # use adftest to find optimal 'd'
#                       max_p=10,
#                       max_q=10,
#                       m=24,             # frequency of series
#                       d=None,           # let model determine 'd'
# #                       seasonal=False,   # No Seasonality
#                       trace=True,
#                       error_action='ignore',
#                       suppress_warnings=True,
#                       stepwise=False)

model = sm.tsa.arima.ARIMA(train, order = (1,1,2), seasonal_order=(1, 1, 2, 24))
fitted = model.fit()
fc = fitted.forecast(-threshold)
# fc_ = (fc_.summary_frame(alpha=0.05))
# fc_
# print(model.summary())
# fc = model.predict(n_periods=-threshold)

appended_org = np.concatenate((train[-20:],test))
appended_fc = np.concatenate((train[-20:],fc))

plt.plot(appended_org)
plt.plot(appended_fc)
plt.savefig(str(-threshold) + '_result.png')
