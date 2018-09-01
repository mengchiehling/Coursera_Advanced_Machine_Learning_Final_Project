from lightgbm import LGBMRegressor
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.model_selection import ShuffleSplit, GridSearchCV
from sklearn.externals import joblib

def Cross_Validation(model, X, y, params, n_splits=5, test_size=0.2, random_state=42, scoring='neg_mean_squared_error', output_model_name="models/myModel"):

    cv = ShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=random_state)
    regression_cv = GridSearchCV(model, param_grid=params, cv=cv, scoring=scoring, verbose=2)

    regression_cv.fit(X, y)

    print(regression_cv.best_score_)
    print(regression_cv.best_params_)

    joblib.dump(regression_cv, output_model_name, protocol=2)

def LGBMRegressor_Fitting(X, y, params, scoring='neg_mean_squared_error', output_model_name='models/LGBMRegressor'):

    regressor = LGBMRegressor(metric='l2_regression')

    Cross_Validation(regressor, X, y, params, n_splits=5, test_size=0.2, random_state=42, scoring=scoring, output_model_name=output_model_name)

if __name__ == "__main__":
    
    time_begin = datetime.now()
    
    model_name = "models/LGBMRegressor"
    key = "LGBM"
    
    num_leaves = np.arange(11, 52, 10).astype(int)
    learning_rate = [0.025, 0.05, 0.075, 0.1]
    n_estimators = [400]
    max_depth = [9, 12, 15, 18]
    min_data_in_leaf = np.arange(20, 51, 10).astype(int)

    #-26.2931161011
    #{'n_estimators': 400, 'num_leaves': 51, 'learning_rate': 0.05, 'max_depth': 18, 'min_data_in_leaf': 50}

    params = {'num_leaves': num_leaves,
              'learning_rate': learning_rate,
              'n_estimators': n_estimators,
              'max_depth': max_depth,
              'min_data_in_leaf': min_data_in_leaf}
    
    X_train = pd.read_csv("X_train.csv.gz")
    X_test = pd.read_csv("X_test.csv.gz")
    Y_train = pd.read_csv("Y_train.csv.gz")

    LGBMRegressor_Fitting(X_train, Y_train.values.reshape(-1,), params, output_model_name=model_name)
    
    time_end = datetime.now()
    
    period = time_end - time_begin
    
    print period
    
    regressor = joblib.load(model_name)
    
    prediction1 = regressor.predict(X_train).reshape(-1,1)
    prediction1_df = pd.DataFrame(data=prediction1, columns=[key])
    prediction1_df.to_csv("predictions_training/{}.csv".format(key), index=False)

    prediction2 = regressor.predict(X_test.drop(['ID'], axis=1)).reshape(-1,1)
    prediction2_df = pd.DataFrame(data=prediction2, columns=[key])
    prediction2_df = pd.concat([X_test[['ID']], prediction2_df], axis=1)
    prediction2_df.to_csv("predictions_testing/{}.csv".format(key), index=False)
