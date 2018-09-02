from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import Adam
from sklearn.metrics import r2_score
import pandas as pd
from datetime import datetime

if __name__ == '__main__':

    time_begin = datetime.now()

    model_name = 'models/DenseNeuralNetWork_01'
    key = "DNN_01"

    """
    Start loading the data
    """
    X_train = pd.read_csv("X_train.csv.gz")
    Y_train = pd.read_csv("Y_train.csv.gz")

    X_test = pd.read_csv("X_test.csv.gz")

    """
    Finishing loading the data
    """

    n_features = len(X_train.columns)

    regressor = Sequential([
                            Dense(256, activation='elu', input_shape=(n_features,)),
                            Dense(1024, activation='elu'),
                            Dense(4096, activation='elu'),
                            Dense(1, activation='relu')
                            ])

    checkpoint = ModelCheckpoint(model_name, verbose=1, monitor='val_loss', save_best_only=True, mode='auto')
    reduceLR = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10)

    optimizer=Adam(lr=0.0001, amsgrad=True)

    regressor.compile(loss="mse", optimizer=optimizer, metrics=['mse'])

    regressor.fit(X_train.values, Y_train.values.reshape(-1,), epochs=200, batch_size=256, validation_split=0.1, callbacks=[checkpoint, reduceLR], verbose=0)

    regressor.load_weight(model_name)

    prediction_1 = regressor.predict(X_train.values, batch_size=256).reshape(-1, 1)
    prediction_2 = regressor.predict(X_test.drop(labels=['ID'], axis=1).values, batch_size=256).reshape(-1, 1)

    r2 = np.round(r2_score(Y_train, prediction_1), 4)
    print('r2_score = {}'.format(r2_score(r2)))

    prediction_1 = pd.DataFrame(data=prediction_1, columns=[key_word])
    prediction_1.to_csv("predictions_training/{}.csv".format(key_word), index=False)

    prediction_2 = pd.DataFrame(data=prediction_2, columns=[key_word])
    prediction_2 = pd.concat([X_test[['ID']], prediction_2], axis=1)
    prediction_2.to_csv("predictions_testing/{}.csv".format(key_word), index=False)

    Running_time = datetime.now() - time_begin

    print('Total training time = ', Running_time)