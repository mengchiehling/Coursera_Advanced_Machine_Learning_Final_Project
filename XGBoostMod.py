from Regression import *
import pandas as pd
import numpy as np
from datetime import datetime

if __name__ == "__main__":
    
    time_begin = datetime.now()
    
    model_name = "models/XGBoostRegressor"
    key = "XGB"
    
    n_estimators = [400] # np.arange(100, 301, 50)
    learning_rate = [0.025, 0.05, 0.075, 0.1]
    gamma = [1] #np.logspace(0, 1, 4)
    max_depth = [7, 10, 13, 16]
    min_child_weight = [1] #np.linspace(1, 17, 5).astype(int)
    colsample_bytree = [0.7]
    subsample = [0.8]

    params = {'n_estimators': n_estimators,
              'learning_rate': learning_rate,
              'gamma': gamma,
              'max_depth': max_depth,
              'min_child_weight': min_child_weight,
              'colsample_bytree': colsample_bytree,
              'subsample': subsample}

    
    X_train = pd.read_csv("X_train.csv.gz")
    X_test = pd.read_csv("X_test.csv.gz")
    Y_train = pd.read_csv("Y_train.csv.gz")
    
    XGBRegressor_Fitting(X_train, Y_train, params, output_model_name=model_name)
    
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
