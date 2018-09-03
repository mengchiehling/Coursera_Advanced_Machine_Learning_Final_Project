from Regression import *
import pandas as pd
import numpy as np
from datetime import datetime

if __name__ == "__main__":
    
    time_begin = datetime.now()
    
    model_name = "models/ElasticNet"
    key = "EN"
    
    alpha = np.logspace(-1, 2, 11)
    l1_ratio = np.linspace(0, 1, 11)

    # -36.5387269502
    # {'alpha': 0.10000000000000001, 'l1_ratio': 0.0}

    params = {'alpha': alpha,
              'l1_ratio': l1_ratio
              }
    
    X_train = pd.read_csv("X_train.csv.gz")
    X_test = pd.read_csv("X_test.csv.gz")
    Y_train = pd.read_csv("Y_train.csv.gz")
    
    ElasticNet_Fitting(X_train, Y_train, params, output_model_name=model_name)
    
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
