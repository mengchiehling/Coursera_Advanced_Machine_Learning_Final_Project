import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.externals import joblib
from sklearn.model_selection import ShuffleSplit, GridSearchCV
from datetime import datetime

def Cross_Validation(model, X, y, params, n_splits=5, test_size=0.2, random_state=42, scoring='neg_mean_squared_error',
                     output_model_name="models/myModel"):

    cv = ShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=random_state)
    regression_cv = GridSearchCV(model, param_grid=params, cv=cv, scoring=scoring, verbose=2)

    regression_cv.fit(X, y)

    print(regression_cv.best_score_)
    print(regression_cv.best_params_)

    joblib.dump(regression_cv, output_model_name)

def RandomForestRegressor_Fitting(X, y, params, scoring='neg_mean_squared_error', output_model_name='models/RandomForestRegressor'):

    regressor = RandomForestRegressor(oob_score=True, random_state=0)

    Cross_Validation(regressor, X, y, params, n_splits=5, test_size=0.2, random_state=42, scoring=scoring,
                     output_model_name=output_model_name)

def Running_prediction(X_train, X_test, Y_train, model_name, key_word):

    regressor = joblib.load(model_name)

    prediction_1 = regressor.predict(X_train).reshape(-1, 1)
    prediction_2 = regressor.predict(X_test.drop(labels=['ID'], axis=1)).reshape(-1, 1)

    r2 = np.round(r2_score(Y_train, prediction_1), 4)
    print('r2_score = {}'.format(r2))

    prediction_1 = pd.DataFrame(data=prediction_1, columns=[key_word])
    prediction_1.to_csv("predictions_training/{}.csv".format(key_word), index=False)

    prediction_2 = pd.DataFrame(data=prediction_2, columns=[key_word])
    prediction_2 = pd.concat([X_test[['ID']], prediction_2], axis=1)
    prediction_2.to_csv("predictions_testing/{}.csv".format(key_word), index=False)

if __name__ == '__main__':

    time_begin = datetime.now()

    model_name = 'models/RandomForestRegressor'
    key = "RF"

    # -24.797595364626634
    # {'n_estimators': 400, 'max_depth': 15, 'min_samples_leaf': 5}

    """
    Start loading the data
    """
    X_train = pd.read_csv("X_train.csv.gz")
    Y_train = pd.read_csv("Y_train.csv.gz")

    X_test = pd.read_csv("X_test.csv.gz")

    """
    Finishing loading the data
    """

    n_estimators = [400]
    max_depth = [15] #np.arange(5, 16, 5)
    min_samples_leaf = [5] #np.arange(2, 19, 3)

    # Drop 'item_price_inv'

    X_train.drop(labels=['item_price_inv'], axis=1, inplace=True)
    X_test.drop(labels=['item_price_inv'], axis=1, inplace=True)

    params = {'n_estimators': n_estimators,
              'max_depth': max_depth,
              'min_samples_leaf': min_samples_leaf}

    RandomForestRegressor_Fitting(X_train, Y_train, params,
                         output_model_name=model_name)

    Running_prediction(X_train, X_test, Y_train, model_name, key)

    Running_time = datetime.now() - time_begin

    print('Total training time = ', Running_time)