from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, AdaBoostRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import ElasticNet
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import ShuffleSplit, GridSearchCV
from sklearn.externals import joblib

def Cross_Validation(model, X, y, params, n_splits=5, test_size=0.2, random_state=42, scoring='neg_mean_squared_error', output_model_name="models/myModel"):

    cv = ShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=random_state)
    regression_cv = GridSearchCV(model, param_grid=params, cv=cv, scoring=scoring)

    regression_cv.fit(X, y)

    print(regression_cv.best_score_)
    print(regression_cv.best_params_)

    joblib.dump(regression_cv, output_model_name)

def RandomForestRegressor_Fitting(X, y, params, scoring='neg_mean_squared_error', output_model_name='models/RandomForestRegressor'):

    regressor = RandomForestRegressor(oob_score=True, n_jobs=-1)

    Cross_Validation(regressor, X, y, params, n_splits=5, test_size=0.2, random_state=42, scoring=scoring,
                     output_model_name=output_model_name)

def ExtraTreesRegressor_Fitting(X, y, params, scoring='neg_mean_squared_error', output_model_name='models/ExtraTreesRegressor'):

    regressor = ExtraTreesRegressor(oob_score=True, bootstrap=True, n_jobs=-1)

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