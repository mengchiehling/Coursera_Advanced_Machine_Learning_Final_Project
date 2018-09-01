from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, AdaBoostRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import ElasticNet
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from sklearn.svm import SVR
from sklearn.model_selection import ShuffleSplit, GridSearchCV
from sklearn.externals import joblib

def Running_prediction(X_train, X_test, Y_train, model_name, key_word):

    regressor = joblib.load(model_name)

    prediction_1 = regressor.predict(X_train).reshape(-1, 1)
    prediction_2 = regressor.predict(X_test.drop(labels=['ID'], axis=1)).reshape(-1, 1)

    r2 = np.round(r2_score(Y_train, prediction_1), 4)
    print('r2_score = {}'.format(r2_score(r2)))

    prediction_1 = pd.DataFrame(data=prediction_1, columns=[key_word])
    prediction_1.to_csv("predictions_training/{}.csv".format(key_word), index=False)

    prediction_2 = pd.DataFrame(data=prediction_2, columns=[key_word])
    prediction_2 = pd.concat([X_test[['ID']], prediction_2], axis=1)
    prediction_2.to_csv("predictions_testing/{}.csv".format(key_word), index=False)

def Cross_Validation(model, X, y, params, n_splits=5, test_size=0.2, random_state=42, scoring='neg_mean_squared_error', output_model_name="models/myModel"):

    cv = ShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=random_state)
    regression_cv = GridSearchCV(model, param_grid=params, cv=cv, scoring=scoring, verbose=2)

    regression_cv.fit(X, y)

    print(regression_cv.best_score_)
    print(regression_cv.best_params_)

    joblib.dump(regression_cv, output_model_name, protocol=2)

def RandomForestRegressor_Fitting(X, y, params, scoring='neg_mean_squared_error', output_model_name='models/RandomForestRegressor'):

    regressor = RandomForestRegressor(oob_score=True)

    Cross_Validation(regressor, X, y, params, n_splits=5, test_size=0.2, random_state=42, scoring=scoring,
                     output_model_name=output_model_name)

def ExtraTreesRegressor_Fitting(X, y, params, scoring='neg_mean_squared_error', output_model_name='models/ExtraTreesRegressor'):

    regressor = ExtraTreesRegressor(oob_score=True, bootstrap=True)

    Cross_Validation(regressor, X, y, params, n_splits=5, test_size=0.2, random_state=42, scoring=scoring,
                     output_model_name=output_model_name)

def AdaBoostRegressor_Fitting(X, y, params, scoring='neg_mean_squared_error', output_model_name='models/AdaBoostRegressor'):

    regressor = AdaBoostRegressor()

    Cross_Validation(regressor, X, y, params, n_splits=5, test_size=0.2, random_state=42, scoring=scoring,
                     output_model_name=output_model_name)
    

def GradientBoostingRegressor_Fitting(X, y, params, scoring='neg_mean_squared_error', output_model_name='models/GradientBoostingRegressor'):

    regressor = GradientBoostingRegressor()

    Cross_Validation(regressor, X, y, params, n_splits=5, test_size=0.2, random_state=42, scoring=scoring,
                     output_model_name=output_model_name)

def XGBRegressor_Fitting(X, y, params, scoring='neg_mean_squared_error', output_model_name='models/XGBRegressor'):

    regressor = XGBRegressor(tree_method='gpu_hist')

    Cross_Validation(regressor, X, y, params, n_splits=5, test_size=0.2, random_state=42, scoring=scoring,
                     output_model_name=output_model_name)

def KNeighborsRegressor_Fitting(X, y, params, scoring='neg_mean_squared_error', output_model_name='models/KNNRegressor'):

    regressor = KNeighborsRegressor()

    Cross_Validation(regressor, X, y, params, n_splits=5, test_size=0.2, random_state=42, scoring=scoring,
                     output_model_name=output_model_name)

def ElasticNet_Fitting(X, y, params, scoring='neg_mean_squared_error', output_model_name='models/ElasticNet'):

    regressor = ElasticNet()

    Cross_Validation(regressor, X, y, params, n_splits=5, test_size=0.2, random_state=42, scoring=scoring,
                     output_model_name=output_model_name)
    
def LGBMRegressor_Fitting(X, y, params, scoring='neg_mean_squared_error', output_model_name='models/LGBMRegressor'):

    regressor = LGBMRegressor(metric='l2_regression')

    Cross_Validation(regressor, X, y, params, n_splits=5, test_size=0.2, random_state=42, scoring=scoring,
                     output_model_name=output_model_name)

def SVR_Fitting(X, y, params, scoring='neg_mean_squared_error', output_model_name='models/SVR'):

    regressor = SVR(kernel='poly', degree=3)

    Cross_Validation(regressor, X, y, params, n_splits=5, test_size=0.2, random_state=42, scoring=scoring,
                     output_model_name=output_model_name)