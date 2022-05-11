import pandas_datareader.data as web
import datetime
start = datetime.datetime(2000, 1, 1)
end = datetime.datetime(2022, 1, 1)

df = web.DataReader('GOOGL', 'stooq', start, end)
pre_days = 10

def stock_price_lstm_data_precessing(df, mem_his_days, pre_days):
    df.dropna(inplace=True)
    df.sort_index(inplace=True)

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
    return x, y

from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
def start_train():
    mem_days = [5, 10, 15]
    lstm_layers = [1, 2, 3]
    dense_layers = [1, 2, 3]
    units = [16, 32]


    for the_mem_days in mem_days:
        for the_lstm_layers in lstm_layers:
            for the_dense_layers in dense_layers:
                for the_units in units:
                    filepath = "./models/{val_mape:.2f}_{epoch:02d}_" + f'mem_{the_mem_days}_lstm_{the_lstm_layers}_dense_{the_dense_layers}_unit_{the_units}'
                    checkpoint = ModelCheckpoint(
                        filepath=filepath,
                        save_weights_only=False,
                        monitor='val_mape',
                        mode='min',
                        save_best_only=True
                    )
                    x, y = stock_price_lstm_data_precessing(df, the_mem_days, pre_days)
                    x_train,x_test,y_train,y_test = train_test_split(x,y,shuffle=False, test_size=0.1)

                    import tensorflow as tf

                    from tensorflow.keras.models import Sequential
                    from tensorflow.keras.layers import LSTM,Dense, Dropout

                    model = Sequential()
                    model.add(LSTM(the_units, input_shape=x.shape[1:],activation='relu',return_sequences=True))
                    model.add(Dropout(0.1))

                    for i in range(the_lstm_layers):
                        model.add(LSTM(the_units, activation='relu',return_sequences=True))
                        model.add(Dropout(0.1))

                    model.add(LSTM(the_units, activation='relu'))
                    model.add(Dropout(0.1))

                    for i in range(the_dense_layers):
                        model.add(Dense(the_units, activation='relu'))
                        model.add(Dropout(0.1))


                    model.add(Dense(1))

                    model.compile(optimizer='adam',
                                loss='mse',
                                metrics=['mape'])

                    model.fit(x_train, y_train, batch_size=32, epochs=50,validation_data=(x_test,y_test),callbacks=[checkpoint])

from tensorflow.keras.models import load_model
best_model = load_model('./models/6.86_04_mem_5_lstm_1_dense_1_unit_32')

x, y = stock_price_lstm_data_precessing(df, 5, pre_days)
x_train,x_test,y_train,y_test = train_test_split(x,y,shuffle=False, test_size=0.1)

best_model.evaluate(x_test, y_test)
pre = best_model.predict(x_test)
# model.evaluate(x_test, y_test)

import matplotlib.pyplot as plt
df_time = df.index[-len(y_test):]
plt.plot(df_time, y_test, color='red', label='price')
plt.plot(df_time, pre, color='green', label='predicit')

plt.show()