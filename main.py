import pandas_datareader.data as web
import datetime
start = datetime.datetime(2000, 1, 1)
end = datetime.datetime(2022, 1, 1)

df = web.DataReader('GOOGL', 'stooq', start, end)

df.dropna(inplace=True)
df.sort_index(inplace=True)

pre_days = 10
df['label'] = df['Close'].shift(-pre_days)
print(df)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
sca_x = scaler.fit_transform(df.iloc[:,:-1])
print(f'x = {sca_x}')

mem_his_days = 10

from collections import deque

deq = deque(maxlen=mem_his_days)

x = []

for i in sca_x:
    deq.append(list(i))
    if len(deq) == mem_his_days:
        x.append(list(deq))

# print(x)
print(len(x))
x_lately = x[-pre_days:]
x = x[:-pre_days]

y = df['label'].values[mem_his_days-1:-pre_days]
print(len(y))

import numpy as np
x = np.array(x)
y = np.array(y)
print(x.shape)
print(y.shape)

from sklearn.model_selection import train_test_split
x_tran,x_test,y_train,y_test = train_test_split(x,y, test_size=0.1)

import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM,Dense, Dropout

model = Sequential()
model.add(LSTM(10, input_shape=x.shape[1:],activation='relu',return_sequences=True))
model.add(Dropout(0.1))

model.add(LSTM(10, activation='relu',return_sequences=True))
model.add(Dropout(0.1))

model.add(LSTM(10, activation='relu',return_sequences=True))
model.add(Dropout(0.1))

model.add(LSTM(10, activation='relu'))
model.add(Dropout(0.1))

model.add(Dense(1))

model.compile(optimizer='adam',
              loss='mse',
              metrics=['mape'])

model.fit(x_tran, y_train, batch_size=32, epochs=50,validation_data=(x_test,y_test))