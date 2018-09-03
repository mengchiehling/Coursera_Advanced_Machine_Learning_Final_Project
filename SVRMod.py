from Regression import *
import pandas as pd
import numpy as np
from datetime import datetime

if __name__ == "__main__":
    
    time_begin = datetime.now()
    
    model_name = "models/SVR_poly"
    key = "SVR_poly"
    
    C = np.linspace(0, 1000, 51)
    C[0] = 1
    epsilon = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    
    
    params = {'C': C,
              'epsilon': epsilon}
    
    X_train = pd.read_csv("X_train.csv.gz")
    X_test = pd.read_csv("X_test.csv.gz")
    Y_train = pd.read_csv("Y_train.csv.gz")
    
    SVR_Fitting(X_train, Y_train, params, output_model_name=model_name)
    
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
