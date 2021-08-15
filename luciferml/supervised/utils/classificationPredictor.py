import tensorflow as tf
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from luciferml.supervised.utils.classification_params import *
from sklearn.ensemble import (AdaBoostClassifier, BaggingClassifier,
                              ExtraTreesClassifier, GradientBoostingClassifier,
                              RandomForestClassifier)
from sklearn.linear_model import (LogisticRegression,
                                  PassiveAggressiveClassifier, Perceptron,
                                  RidgeClassifier, RidgeClassifierCV,
                                  SGDClassifier)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier


def classificationPredictor(
        predictor, params, X_train, X_val, y_train, y_val, epochs, hidden_layers,
        input_activation, output_activation, loss,
        batch_size, metrics, validation_split, optimizer, output_units, input_units, tune_mode, dropout_rate=0,
        all_mode=False, verbose=False

):
    """
    Takes Predictor string , parameters , Training and Validation set and Returns a classifier for the Choosen Predictor.
    """
    try:
        if predictor == 'lr':
            if not all_mode:
                print('Training Logistic Regression on Training Set [*]\n')
            classifier = LogisticRegression(**params)
            if tune_mode == 1:
                parameters = parameters_lin_1
            elif tune_mode == 2:
                parameters = parameters_lin_2
            elif tune_mode == 3:
                parameters = parameters_lin_3

        elif predictor == 'sgd':
            if not all_mode:
                print(
                    'Training Stochastic Gradient Descent on Training Set [*]\n')
            classifier = SGDClassifier(**params)
            if tune_mode == 1:
                parameters = parameters_sgd_1
            elif tune_mode == 2:
                parameters = parameters_sgd_2
            elif tune_mode == 3:
                parameters = parameters_sgd_3

        elif predictor == 'perc':
            if not all_mode:
                print('Training Perceptron on Training Set [*]\n')
            classifier = Perceptron(**params)
            if tune_mode == 1:
                parameters = parameters_perc
            elif tune_mode == 2:
                parameters = parameters_perc
            elif tune_mode == 3:
                parameters = parameters_perc

        elif predictor == 'pass':
            if not all_mode:
                print('Training Passive Aggressive on Training Set [*]\n')
            classifier = PassiveAggressiveClassifier(**params)
            if tune_mode == 1:
                parameters = parameters_pass
            elif tune_mode == 2:
                parameters = parameters_pass
            elif tune_mode == 3:
                parameters = parameters_pass

        elif predictor == 'ridg':
            if not all_mode:
                print('Training Ridge Classifier on Training Set [*]\n')
            classifier = RidgeClassifier(**params)
            if tune_mode == 1:
                parameters = parameters_ridg
            elif tune_mode == 2:
                parameters = parameters_ridg
            elif tune_mode == 3:
                parameters = parameters_ridg

        elif predictor == 'svm':
            if not all_mode:
                print('Training Support Vector Machine on Training Set [*]\n')
            classifier = SVC(**params)
            if tune_mode == 1:
                parameters = parameters_svm_1
            elif tune_mode == 2:
                parameters = parameters_svm_2
            elif tune_mode == 3:
                parameters = parameters_svm_3

        elif predictor == 'knn':
            if not all_mode:
                print('Training K-Nearest Neighbours on Training Set [*]\n')
            classifier = KNeighborsClassifier(**params)
            if tune_mode == 1:
                parameters = parameters_knn_1
            elif tune_mode == 2:
                parameters = parameters_knn_2
            elif tune_mode == 3:
                parameters = parameters_knn_3

        elif predictor == 'dt':
            if not all_mode:
                print(
                    'Training Decision Tree Classifier on Training Set [*]\n')
            classifier = DecisionTreeClassifier(**params)
            if tune_mode == 1:
                parameters = parameters_dt_1
            elif tune_mode == 2:
                parameters = parameters_dt_2
            elif tune_mode == 3:
                parameters = parameters_dt_3

        elif predictor == 'nb':
            if not all_mode:
                print('Training Naive Bayes Classifier on Training Set [*]\n')
            classifier = GaussianNB(**params)
            parameters = {}

        elif predictor == 'rfc':
            if not all_mode:
                print(
                    'Training Random Forest Classifier on Training Set [*]\n')
            classifier = RandomForestClassifier(**params)
            if tune_mode == 1:
                parameters = parameters_rfc_1
            elif tune_mode == 2:
                parameters = parameters_rfc_2
            elif tune_mode == 3:
                parameters = parameters_rfc_3

        elif predictor == 'gbc':
            if not all_mode:
                print(
                    'Training Gradient Boosting Classifier on Training Set [*]\n')
            classifier = GradientBoostingClassifier(**params)
            if tune_mode == 1:
                parameters = parameters_gbc_1
            elif tune_mode == 2:
                parameters = parameters_gbc_2
            elif tune_mode == 3:
                parameters = parameters_gbc_3
        elif predictor == 'ada':
            if not all_mode:
                print('Training AdaBoost Classifier on Training Set [*]\n')
            classifier = AdaBoostClassifier(**params)
            if tune_mode == 1:
                parameters = parameters_ada_1
            elif tune_mode == 2:
                parameters = parameters_ada_2
            elif tune_mode == 3:
                parameters = parameters_ada_3

        elif predictor == 'bag':
            if not all_mode:
                print('Training Bagging Classifier on Training Set [*]\n')
            classifier = BaggingClassifier(**params)
            if tune_mode == 1:
                parameters = parameters_bag_1
            elif tune_mode == 2:
                parameters = parameters_bag_2
            elif tune_mode == 3:
                parameters = parameters_bag_3

        elif predictor == 'extc':
            if not all_mode:
                print('Training Extra Trees Classifier on Training Set [*]\n')
            classifier = ExtraTreesClassifier(**params)
            if tune_mode == 1:
                parameters = parameters_extc_1
            elif tune_mode == 2:
                parameters = parameters_extc_2
            elif tune_mode == 3:
                parameters = parameters_extc_3

        elif predictor == 'lgbm':
            if not all_mode:
                print('Training LightGBM on Training Set [*]\n')
            classifier = LGBMClassifier(**params)
            if tune_mode == 1:
                parameters = parameters_lgbm_1
            elif tune_mode == 2:
                parameters = parameters_lgbm_2
            elif tune_mode == 3:
                parameters = parameters_lgbm_3

        elif predictor == 'cat':
            if not all_mode:
                print('Training CatBoostClassifier on Training Set [*]\n')
            params['verbose'] = verbose
            classifier = CatBoostClassifier(**params)
            if tune_mode == 1:
                parameters = parameters_cat_1
            elif tune_mode == 2:
                parameters = parameters_cat_2
            elif tune_mode == 3:
                parameters = parameters_cat_3

        elif predictor == 'xgb':
            if not all_mode:
                print('Training XGBClassifier on Training Set [*]\n')
            classifier = XGBClassifier(**params)
            if tune_mode == 1:
                parameters = parameters_xgb_1
            elif tune_mode == 2:
                parameters = parameters_xgb_2
            elif tune_mode == 3:
                parameters = parameters_xgb_3

        elif predictor == 'ann':
            def build_ann_model(input_units, optimizer, rate):
                try:
                    classifier = tf.keras.models.Sequential()
                    for i in range(0, hidden_layers):
                        classifier.add(tf.keras.layers.Dense(
                            units=input_units, activation=input_activation))
                        classifier.add(tf.keras.layers.Dropout(rate=rate))
                    classifier.add(tf.keras.layers.Dense(
                        units=output_units, activation=output_activation))
                    classifier.compile(optimizer=optimizer,
                                       loss=loss, metrics=metrics)
                    return classifier
                except Exception as error:
                    print('ANN Build Failed with error :', error, '\n')
            if not all_mode:
                print('Training ANN on Training Set [*]\n')
            classifier = build_ann_model(input_units, optimizer, dropout_rate)
            if verbose == False:
                verbose = 0
            if verbose == True:
                verbose = 'auto'
            ann_history = classifier.fit(
                X_train, y_train, validation_split=validation_split,
                validation_data=(
                    X_val, y_val), epochs=epochs, batch_size=batch_size, verbose=verbose)
            classifier_wrap = tf.keras.wrappers.scikit_learn.KerasClassifier(
                build_fn=build_ann_model, input_units=input_units,
                epochs=epochs, batch_size=batch_size, optimizer=optimizer, rate=dropout_rate,
            )
            if tune_mode == 1:
                parameters = parameters_ann_1
            elif tune_mode == 2:
                parameters = parameters_ann_2
            elif tune_mode == 3:
                parameters = parameters_ann_3
        if predictor == 'ann':
            return (parameters, classifier, classifier_wrap)
        return (parameters, classifier)
    except Exception as error:
        print('Model Build Failed with error :', error, '\n')
