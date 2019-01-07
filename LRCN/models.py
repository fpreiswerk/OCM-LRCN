from keras.models import Sequential
from keras.layers import Dense, Conv1D, LSTM, AveragePooling1D, Dropout
from keras.optimizers import Adam

conv_activation = 'tanh'


def LRCN(input_shape, output_dim):

    model = Sequential()

    model.add(Conv1D(64, 3, activation=conv_activation, padding='causal',
                     input_shape=(input_shape[0], input_shape[1])))
    model.add(AveragePooling1D(pool_size=2))
    model.add(Dropout(0.2))

    model.add(Conv1D(32, 3, activation=conv_activation, padding='causal'))
    model.add(AveragePooling1D(pool_size=2))
    model.add(Dropout(0.2))

    model.add(Conv1D(16, 3, activation=conv_activation, padding='causal'))
    model.add(AveragePooling1D(pool_size=2))
    model.add(Dropout(0.2))

    model.add(Conv1D(1, 3, activation=conv_activation, padding='causal'))
    model.add(AveragePooling1D(pool_size=2))
    model.add(Dropout(0.2))

    model.add(LSTM(10, stateful=False, unroll=False, return_sequences=True,
                   dropout=0))
    model.add(LSTM(10, stateful=False, unroll=False, return_sequences=False,
                   dropout=0))

    model.add(Dense(output_dim, activation='linear'))

    optim = Adam()
    model.compile(loss='mean_squared_error', optimizer=optim)

    return model
