import numpy as np

parameters_svr_1 = [
    {'kernel': ['rbf'], 'gamma': [0.1, 0.5, 0.9, 1],
        'C': np.logspace(-4, 4, 5)},
]

parameters_svr_2 = [
    {'kernel': ['rbf'], 'gamma': [1e-4, 0.1, 0.3,
                                  0.5, 0.7, 0.9, 1], 'C': np.logspace(-4, 4, 10)},
    {'kernel': ['linear'], 'gamma': [1e-4, 0.1, 0.3,
                                     0.5, 0.7, 0.9, 1], 'C': np.logspace(-4, 4, 10)},
    {'kernel': ['poly'], 'gamma': [1e-4, 0.1, 0.3,
                                   0.5, 0.7, 0.9, 1], 'C': np.logspace(-4, 4, 10)},
]
parameters_svr_3 = [
    {'kernel': ['rbf'], 'gamma': [1e-3, 1e-4, 0.1, 0.2, 0.3, 0.4,
                                  0.5, 0.6, 0.7, 0.8, 0.9], 'C': np.logspace(-4, 4, 20)},
    {'kernel': ['linear'], 'gamma': [1e-3, 1e-4, 0.1, 0.2, 0.3,
                                     0.4, 0.5, 0.6, 0.7, 0.8, 0.9], 'C': np.logspace(-4, 4, 20)},
    {'kernel': ['poly'], 'gamma': [1e-3, 1e-4, 0.1, 0.2, 0.3,
                                   0.4, 0.5, 0.6, 0.7, 0.8, 0.9], 'C': np.logspace(-4, 4, 20)},
    {'kernel': ['sigmoid'], 'gamma': [1e-3, 1e-4, 0.1, 0.2, 0.3,
                                      0.4, 0.5, 0.6, 0.7, 0.8, 0.9], 'C': np.logspace(-4, 4, 20)},
]


parameters_knr_1 = [{
    'n_neighbors': list(range(1, 11)),
    'weights': ['uniform', 'distance'],
    'algorithm': ['auto', 'kd_tree', 'brute'],
}]
parameters_knr_2 = [{
    'n_neighbors': list(range(1, 21)),
    'weights': ['uniform', 'distance'],
    'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
}]
parameters_knr_3 = [{
    'n_neighbors': list(range(1, 31)),
    'weights': ['uniform', 'distance'],
    'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
}]


parameters_dt_1 = [{
    'criterion': ['mse', 'mae'],
    "min_samples_leaf": [20, 40],
    "max_leaf_nodes": [5, 20],
    'max_depth': [4,  6,  8,  10,  12,  20,  40, 70],

}]
parameters_dt_2 = [{
    'criterion': ['mse', 'friedman_mse', 'mae'],
    "min_samples_leaf": [20, 40, 100],
    "max_leaf_nodes": [5, 20, 100],
    'max_features': [2, 3],
    'max_depth': [4, 6, 7,  9, 10, 12, 20,  40, 50, 90, 120],

}]
parameters_dt_3 = [{
    'criterion': ['mse', 'friedman_mse', 'mae', 'poisson'],
    "min_samples_leaf": [20, 40, 100, 150],
    "max_leaf_nodes": [5, 20, 100, 150],
    'max_features': [2, 3],
    'max_depth': [4, 5, 6, 7, 8, 9, 10, 11, 12, 15, 20, 30, 40, 50, 70, 90, 120, 150],

}]


parameters_rfr_1 = [{
    'n_estimators': [100, 200, 300, 400, 500, 750, 1000],
    'max_depth': [4,  6,  8,  10,  12,  20,  40, 70],
    'max_features': [2, 3],
    'min_samples_leaf': [3, 4, 5],
}]
parameters_rfr_2 = [{
    'criterion': ['mse', 'mae'],
    'n_estimators': [50, 100, 150, 200, 250, 300, 400, 500, 700, 900, 1000],
    'bootstrap': [True, False],
    'max_depth': [4, 6, 7,  9, 10, 12, 20,  40, 50, 90, 120],
    'max_features': [2, 3],
    'min_samples_leaf': [4, 5],
    'min_samples_split': [10, 12],
}]
parameters_rfr_3 = [{
    'criterion': ['mse', 'mae'],
    'n_estimators': [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 800, 900, 1000],
    'bootstrap': [True, False],
    'max_depth': [4, 5, 6, 7, 8, 9, 10, 11, 12, 15, 20, 30, 40, 50, 70, 90, 120, 150],
    'max_features': [2, 3],
    'min_samples_leaf': [3, 4, 5],
    'min_samples_split': [8, 10, 12],
}]


parameters_gbr_1 = [{
    'learning_rate': [0.02, 0.04],
    'subsample': [0.5, 0.2, 0.1],
    'n_estimators': [100, 200, 300, 400, 500, 750, 1000],
    'max_depth': [4, 6],
    'loss': ['ls', 'lad'],
    'criterion': ['mse', 'mae'],
    'min_samples_split': [8, 10],
    'min_samples_leaf': [3, 4],
}]
parameters_gbr_2 = [{
    'learning_rate': [0.02, 0.03, 0.04],
    'subsample': [0.5, 0.2, 0.1],
    'n_estimators': [50, 100, 150, 200, 250, 300, 400, 500, 700, 900, 1000],
    'max_depth': [4, 6, 8],
    'loss': ['ls', 'lad', 'huber'],
    'criterion':['friedman_mse', 'mse'],
    'min_samples_split': [8, 10, 12],
    'min_samples_leaf': [3, 4],
}]
parameters_gbr_3 = [{
    'learning_rate': [0.01, 0.02, 0.03, 0.04],
    'subsample': [0.9, 0.5, 0.2, 0.1],
    'n_estimators': [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 800, 900, 1000],
    'max_depth': [4, 6, 8, 10],
    'loss': ['ls', 'lad', 'huber', 'quantile'],
    'criterion': ['friedman_mse', 'mse', 'mae'],
    'min_samples_split': [8, 10, 12],
    'min_samples_leaf': [3, 4, 5],
}]

parameters_ada_1 = [{
    'n_estimators': [100, 200, 300, 400, 500, 750, 1000],
    'learning_rate': [0.02, 0.04],
    'loss': ['linear', 'square', 'exponential'],
}]
parameters_ada_2 = [{
    'n_estimators': [50, 100, 150, 200, 250, 300, 400, 500, 700, 900, 1000],
    'learning_rate': [0.02, 0.03, 0.04],
    'loss': ['linear', 'square', 'exponential'],
    'random_state': [42],
}]
parameters_ada_3 = [{
    'n_estimators': [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 800, 900, 1000],
    'learning_rate': [0.01, 0.02, 0.03, 0.04],
    'loss': ['linear', 'square', 'exponential'],
}]

parameters_bag_1 = [{
    'n_estimators': [100, 200, 300, 400, 500, 750, 1000],
    'max_samples': [0.5, 0.2, 0.1],
    'bootstrap': [True, False],
    'bootstrap_features': [True, False],
}]

parameters_bag_2 = [{
    'n_estimators': [50, 100, 150, 200, 250, 300, 400, 500, 700, 900, 1000],
    'max_samples': [0.5, 0.2, 0.1],
    'max_features': [0.5, 0.2, 0.1],
    'bootstrap': [True, False],
    'bootstrap_features': [True, False],
    'warm_start': [True, False],
}]
parameters_bag_3 = [{
    'n_estimators': [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 800, 900, 1000],
    'max_samples': [0.5, 0.2, 0.1],
    'max_features': [0.5, 0.2, 0.1],
    'bootstrap': [True, False],
    'bootstrap_features': [True, False],
    'warm_start': [True, False],
}]

parameters_extr_1 = [{
    'n_estimators': [100, 200, 300, 400, 500, 750, 1000],
    'max_depth': [4, 6],
    'criterion': ['mse', 'mae'],
    'min_samples_split': [8, 10],
}]

parameters_extr_2 = [{
    'n_estimators': [50, 100, 150, 200, 250, 300, 400, 500, 700, 900, 1000],
    'max_depth': [4, 6, 8],
    'criterion': ['mse', 'mae'],
    'min_samples_split': [8, 10, 12],
    'min_samples_leaf': [3, 4],
}]
parameters_extr_3 = [{
    'n_estimators': [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 800, 900, 1000],
    'max_depth': [4, 6, 7, 9, 10, 12, 20, 40, 50, 90, 120],
    'criterion': ['mse', 'mae'],
    'min_samples_split': [8, 10, 12],
    'min_samples_leaf': [3, 4, 5],
}]


parameters_ann_1 = [{'batch_size': [20, 50, 32],
                     'nb_epoch': [200, 100, 300],
                     'input_units': [5, 6, 10, ],

                     }]
parameters_ann_2 = [{'batch_size': [20, 50, 25, 32],
                     'nb_epoch': [200, 100, 300, 350],
                     'input_units': [5, 6, 10, 11, 12, ],
                     'optimizer': ['adam', 'rmsprop'],

                     }]
parameters_ann_3 = [{'batch_size': [100, 20, 50, 25, 32],
                     'nb_epoch': [200, 100, 300, 400],
                     'input_units': [5, 6, 10, 11, 12, 15],
                    'optimizer': ['adam', 'rmsprop'],
                     }]


parameters_lin = [{
    "fit_intercept": [True, False],
    "positive":[True, False]
}]


parameters_sgd_1 = [{
    'penalty': ['l1', 'l2'],
    'loss': ['squared_loss', 'huber'],
    'alpha':  [0.1, 0.5, 0.9, 1],
    'learning_rate': ['constant', 'optimal'],

}]
parameters_sgd_2 = [{
    'penalty': ['l1', 'l2', 'elasticnet', ],
    'loss': ['squared_loss', 'huber', 'epsilon_insensitive'],
    'alpha':  [1e-4, 0.1, 0.3,
               0.5, 0.7, 0.9, 1],
    "fit_intercept": [True, False],
    'learning_rate': ['constant', 'optimal', 'invscaling'],
    'eta0': [10, 100],
}]
parameters_sgd_3 = [{
    'penalty': ['l1', 'l2', 'elasticnet'],
    'loss': ['squared_loss', 'huber', 'epsilon_insensitive', 'squared_epsilon_insensitive'],
    "fit_intercept": [True, False],
    'alpha':  [1e-3, 1e-4, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    'learning_rate': ['constant', 'optimal', 'invscaling', 'adaptive'],
    'eta0': [1, 10, 100],
}]


parameters_ker_1 = [{
    'alpha':  [0.1, 0.5, 0.9, 1],
    'gamma': [0.1, 0.5, 0.9, 1],
}]
parameters_ker_2 = [{
    'alpha': [1e-4, 0.1, 0.3,
              0.5, 0.7, 0.9, 1],
    'gamma': [1e-4, 0.1, 0.3,
              0.5, 0.7, 0.9, 1],
}]

parameters_ker_3 = [{
    'alpha':  [1e-3, 1e-4, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    'gamma':  [1e-3, 1e-4, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
}]


parameters_elas = [{
    'alpha':  [0.1, 0.5, 0.9, 1],
    'l1_ratio': [0, 0.25, 0.5, 0.75, 1],
}]


parameters_br = [{
    'alpha_1':  [1e-3, 1e-4, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    'alpha_2':  [1e-3, 1e-4, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    'lambda_1':  [1e-3, 1e-4, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    'lambda_1':  [1e-3, 1e-4, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
}]


parameters_lgbm_1 = [{
    'n_estimators': [100, 200, 300, 400, 500, 750, 1000],
    'min_child_weight': [1, 5, 10],
    'colsample_bytree': [0.8, 1.0],
    'reg_alpha': [0, 1, 2, 5, 7, 10],
    'reg_lambda': [0, 1, 2, 5, 7, 10],
}]
parameters_lgbm_2 = [{
    'n_estimators': [50, 100, 150, 200, 250, 300, 400, 500, 700, 900, 1000],
    'min_child_weight': [1, 5, 10],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0],
    'reg_alpha': [0, 1e-1, 1, 2, 5, 7, 10, 50],
    'reg_lambda': [0, 1e-1, 1, 5, 10, 20, 50],
}]
parameters_lgbm_3 = [{
    'n_estimators': [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 800, 900, 1000],
    'min_child_weight': [1e-5, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1e4],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0],
    'reg_alpha': [0, 1e-1, 1, 2, 5, 7, 10, 50, 100],
    'reg_lambda': [0, 1e-1, 1, 5, 10, 20, 50, 100],
}]


parameters_xgb_1 = [{
    'min_child_weight': [1, 5, 10],
    'n_estimators': [100, 200, 300, 400, 500, 750, 1000],
    'gamma': [0.1, 0.5, 0.9, 1],
    'max_depth': [4,  6,  8,  10,  12,  20,  40, 70],
    'learning_rate': [0.3, 0.1],
}]
parameters_xgb_2 = [{
    'min_child_weight': [1e-3, 1e-2, 1e-1, 1, 1e1, 1e2],
    'n_estimators': [50, 100, 150, 200, 250, 300, 400, 500, 700, 900, 1000],
    'gamma':  [1e-4, 0.1, 0.3,
               0.5, 0.7, 0.9, 1],
    'subsample': [0.6, 0.8],
    'colsample_bytree': [0.6, 0.8, 1.0],
    'max_depth': [4, 6, 7,  9, 10, 12, 20,  40, 50, 90, 120],
    'learning_rate': [0.3, 0.1, 0.01],
}]
parameters_xgb_3 = [{
    'min_child_weight': [1e-3, 1e-2, 1e-1, 1, 1e1, 1e2],
    'n_estimators': [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 800, 900, 1000],
    'gamma': [1e-3, 1e-4, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0],
    'max_depth': [4, 5, 6, 7, 8, 9, 10, 11, 12, 15, 20, 30, 40, 50, 70, 90, 120, 150],
    'learning_rate': [0.3, 0.1, 0.03],
}]


parameters_cat = [{
    'depth': [6, 8, 10],
    'learning_rate': [0.01, 0.05, 0.1],
    'iterations': [30, 50, 100],
    'depth': [2, 4, 6, 8],
    'l2_leaf_reg': [0.2, 0.5, 1, 3]
}]
