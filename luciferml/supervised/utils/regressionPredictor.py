import tensorflow as tf
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from luciferml.supervised.utils.regression_params import *
from sklearn.ensemble import (AdaBoostRegressor, BaggingRegressor,
                              ExtraTreesRegressor, GradientBoostingRegressor,
                              RandomForestRegressor)
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import (BayesianRidge, ElasticNet, LinearRegression,
                                  SGDRegressor)
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor


def regressionPredictor(
        predictor, params, X_train, X_val, y_train, y_val, epochs, hidden_layers,
        input_activation, loss,
        batch_size, validation_split, optimizer, output_units, input_units,
        tune_mode, dropout_rate=0, all_mode=False, verbose=False
):
    """
    Takes Predictor string , parameters , Training and Validation set and Returns a regressor for the Choosen Predictor.
    """
    try:
        if predictor == 'lin':
            if not all_mode:
                print('Training Logistic Regression on Training Set [*]\n')
            regressor = LinearRegression(**params)
            if tune_mode == 1:
                parameters = parameters_lin
            elif tune_mode == 2:
                parameters = parameters_lin
            elif tune_mode == 3:
                parameters = parameters_lin

        elif predictor == 'sgd':
            if not all_mode:
                print(
                    'Training Stochastic Gradient Descent Regressor on Training Set [*]\n')
            regressor = SGDRegressor(**params)
            if tune_mode == 1:
                parameters = parameters_sgd_1
            elif tune_mode == 2:
                parameters = parameters_sgd_2
            elif tune_mode == 3:
                parameters = parameters_sgd_3

        elif predictor == 'krr':
            if not all_mode:
                print('Training Kernel Ridge Regressor on Training Set [*]\n')
            regressor = KernelRidge(**params)
            if tune_mode == 1:
                parameters = parameters_ker_1
            elif tune_mode == 2:
                parameters = parameters_ker_2
            elif tune_mode == 3:
                parameters = parameters_ker_3

        elif predictor == 'elas':
            if not all_mode:
                print('Training ElasticNet Regressor on Training Set [*]\n')
            regressor = ElasticNet(**params)
            if tune_mode == 1:
                parameters = parameters_elas
            elif tune_mode == 2:
                parameters = parameters_elas
            elif tune_mode == 3:
                parameters = parameters_elas

        elif predictor == 'br':
            if not all_mode:
                print('Training BayesianRidge Regressor on Training Set [*]\n')
            regressor = BayesianRidge(**params)
            if tune_mode == 1:
                parameters = parameters_br
            elif tune_mode == 2:
                parameters = parameters_br
            elif tune_mode == 3:
                parameters = parameters_br

        elif predictor == 'svr':
            if not all_mode:
                print('Training Support Vector Machine on Training Set [*]\n')
            regressor = SVR(**params)
            if tune_mode == 1:
                parameters = parameters_svr_1
            elif tune_mode == 2:
                parameters = parameters_svr_2
            elif tune_mode == 3:
                parameters = parameters_svr_3

        elif predictor == 'knr':
            if not all_mode:
                print('Training KNeighbors Regressor on Training Set [*]\n')
            regressor = KNeighborsRegressor(**params)
            if tune_mode == 1:
                parameters = parameters_knr_1
            elif tune_mode == 2:
                parameters = parameters_knr_2
            elif tune_mode == 3:
                parameters = parameters_knr_3

        elif predictor == 'dt':
            if not all_mode:
                print('Training Decision Tree regressor on Training Set [*]\n')
            regressor = DecisionTreeRegressor(**params)
            if tune_mode == 1:
                parameters = parameters_dt_1
            elif tune_mode == 2:
                parameters = parameters_dt_2
            elif tune_mode == 3:
                parameters = parameters_dt_3

        elif predictor == 'rfr':
            if not all_mode:
                print('Training Random Forest regressor on Training Set [*]\n')
            regressor = RandomForestRegressor(**params)
            if tune_mode == 1:
                parameters = parameters_rfr_1
            elif tune_mode == 2:
                parameters = parameters_rfr_2
            elif tune_mode == 3:
                parameters = parameters_rfr_3

        elif predictor == 'gbr':
            if not all_mode:
                print(
                    'Training Gradient Boosting Regressor  on Training Set [*]\n')
            regressor = GradientBoostingRegressor(**params)
            if tune_mode == 1:
                parameters = parameters_gbr_1
            elif tune_mode == 2:
                parameters = parameters_gbr_2
            elif tune_mode == 3:
                parameters = parameters_gbr_3

        elif predictor == 'ada':
            if not all_mode:
                print('Training AdaBoost Regressor on Training Set [*]\n')
            regressor = AdaBoostRegressor(**params)
            if tune_mode == 1:
                parameters = parameters_ada_1
            elif tune_mode == 2:
                parameters = parameters_ada_2
            elif tune_mode == 3:
                parameters = parameters_ada_3

        elif predictor == 'bag':
            if not all_mode:
                print('Training Bagging Regressor on Training Set [*]\n')
            regressor = BaggingRegressor(**params)
            if tune_mode == 1:
                parameters = parameters_bag_1
            elif tune_mode == 2:
                parameters = parameters_bag_2
            elif tune_mode == 3:
                parameters = parameters_bag_3

        elif predictor == 'extr':
            if not all_mode:
                print('Training Extra Trees Regressor on Training Set [*]\n')
            regressor = ExtraTreesRegressor(**params)
            if tune_mode == 1:
                parameters = parameters_extr_1
            elif tune_mode == 2:
                parameters = parameters_extr_2
            elif tune_mode == 3:
                parameters = parameters_extr_3

        elif predictor == 'xgb':
            if not all_mode:
                print('Training XGBregressor on Training Set [*]\n')
            regressor = XGBRegressor(**params)
            if tune_mode == 1:
                parameters = parameters_xgb_1
            elif tune_mode == 2:
                parameters = parameters_xgb_2
            elif tune_mode == 3:
                parameters = parameters_xgb_3

        elif predictor == 'lgbm':
            if not all_mode:
                print('Training LGBMRegressor on Training Set [*]\n')
            regressor = LGBMRegressor(**params)
            if tune_mode == 1:
                parameters = parameters_lgbm_1
            elif tune_mode == 2:
                parameters = parameters_lgbm_2
            elif tune_mode == 3:
                parameters = parameters_lgbm_3

        elif predictor == 'cat':
            if not all_mode:
                print('Training CatBoost Regressor on Training Set [*]\n')
            params['verbose'] = verbose
            regressor = CatBoostRegressor(**params)
            if tune_mode == 1:
                parameters = parameters_cat
            elif tune_mode == 2:
                parameters = parameters_cat
            elif tune_mode == 3:
                parameters = parameters_cat

        elif predictor == 'ann':
            def build_ann_model(input_units, optimizer, rate):
                try:
                    regressor = tf.keras.models.Sequential()
                    for i in range(0, hidden_layers):
                        regressor.add(tf.keras.layers.Dense(
                            units=input_units, activation=input_activation))
                        regressor.add(tf.keras.layers.Dropout(rate=rate))
                    regressor.add(tf.keras.layers.Dense(
                        units=output_units))
                    regressor.compile(optimizer=optimizer,
                                      loss=loss)
                    return regressor

                except Exception as error:
                    print('ANN Build Failed with error :', error, '\n')
            if not all_mode:
                print('Training ANN on Training Set [*]\n')
            regressor = build_ann_model(input_units, optimizer, dropout_rate)

            if verbose == False:
                verbose = 0
            if verbose == True:
                verbose = 'auto'
            ann_history = regressor.fit(
                X_train, y_train, validation_split=validation_split,
                validation_data=(
                    X_val, y_val), epochs=epochs, batch_size=batch_size, verbose=verbose
            )
            regressor_wrap = tf.keras.wrappers.scikit_learn.KerasRegressor(
                build_fn=build_ann_model, input_units=input_units,
                epochs=epochs, batch_size=batch_size, optimizer=optimizer, rate=dropout_rate
            )
            if tune_mode == 1:
                parameters = parameters_ann_1
            elif tune_mode == 2:
                parameters = parameters_ann_2
            elif tune_mode == 3:
                parameters = parameters_ann_3

        if predictor == 'ann':
            return (parameters, regressor, regressor_wrap)
        return (parameters, regressor)
    except Exception as error:
        print('Model Build Failed with error :', error, '\n')
