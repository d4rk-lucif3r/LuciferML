import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import tensorflow as tf
from luciferml.supervised.utils.classification_params import *


def classificationPredictor(
        predictor, params, X_train, X_val, y_train, y_val, epochs, hidden_layers,
        input_activation, output_activation, loss,
        batch_size, metrics, validation_split, optimizer, output_units, input_units, tune_mode
):
    """
    Takes Predictor string , parameters , Training and Validation set and Returns a classifier for the Choosen Predictor.
    """
    try:
        if predictor == 'lr':
            print('Training Logistic Regression on Training Set [*]\n')

            classifier = LogisticRegression(**params)
            if tune_mode == 1:
                parameters = parameters_lin_1
            elif tune_mode == 2:
                parameters = parameters_lin_2
            elif tune_mode == 3:
                parameters = parameters_lin_3

        elif predictor == 'svm':
            print('Training Support Vector Machine on Training Set [*]\n')

            classifier = SVC(**params)
            if tune_mode == 1:
                parameters = parameters_svm_1
            elif tune_mode == 2:
                parameters = parameters_svm_2
            elif tune_mode == 3:
                parameters = parameters_svm_3

        elif predictor == 'knn':
            print('Training K-Nearest Neighbours on Training Set [*]\n')

            classifier = KNeighborsClassifier(**params)
            if tune_mode == 1:
                parameters = parameters_knn_1
            elif tune_mode == 2:
                parameters = parameters_knn_2
            elif tune_mode == 3:
                parameters = parameters_knn_3

        elif predictor == 'dt':
            print('Training Decision Tree Classifier on Training Set [*]\n')

            classifier = DecisionTreeClassifier(**params)

            if tune_mode == 1:
                parameters = parameters_dt_1
            elif tune_mode == 2:
                parameters = parameters_dt_2
            elif tune_mode == 3:
                parameters = parameters_dt_3

        elif predictor == 'nb':
            print('Training Naive Bayes Classifier on Training Set [*]\n')

            classifier = GaussianNB(**params)
            parameters = {}

        elif predictor == 'rfc':
            print('Training Random Forest Classifier on Training Set [*]\n')

            classifier = RandomForestClassifier(**params)
            if tune_mode == 1:
                parameters = parameters_rfc_1
            elif tune_mode == 2:
                parameters = parameters_rfc_2
            elif tune_mode == 3:
                parameters = parameters_rfc_3

        elif predictor == 'xgb':
            print('Training XGBClassifier on Training Set [*]\n')

            classifier = XGBClassifier(**params)

            if tune_mode == 1:
                parameters = parameters_xgb_1
            elif tune_mode == 2:
                parameters = parameters_xgb_2
            elif tune_mode == 3:
                parameters = parameters_xgb_3
            parameters = {
                'min_child_weight': [1, 5, 10],
                'gamma': [1e-3, 1e-4, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
                'subsample': [0.6, 0.8, 1.0],
                'colsample_bytree': [0.6, 0.8, 1.0],
                'max_depth': [4, 5, 6, 7, 8, 9, 10, 11, 12, 15, 20, 30, 40, 50, 70, 90, 120, 150],
                'learning_rate': [0.3, 0.1, 0.03],
            }
        elif predictor == 'ann':
            def build_ann_model(input_units):
                try:
                    classifier = tf.keras.models.Sequential()
                    for i in range(0, hidden_layers):
                        classifier.add(tf.keras.layers.Dense(
                            units=input_units, activation=input_activation))
                    classifier.add(tf.keras.layers.Dense(
                        units=output_units, activation=output_activation))
                    classifier.compile(optimizer=optimizer,
                                       loss=loss, metrics=metrics)
                    return classifier

                except Exception as error:
                    print('ANN Build Failed with error :', error, '\n')
            print('Training ANN on Training Set [*]\n')
            classifier = build_ann_model(input_units)

            ann_history = classifier.fit(
                X_train, y_train, validation_split=validation_split,
                validation_data=(
                    X_val, y_val), epochs=epochs, batch_size=batch_size
            )
            classifier_wrap = tf.keras.wrappers.scikit_learn.KerasClassifier(
                build_fn=build_ann_model, verbose=1, input_units=input_units,
                epochs=epochs, batch_size=batch_size
            )
            if tune_mode == 1:
                parameters = parameters_ann_1
            elif tune_mode == 2:
                parameters = parameters_ann_2
            elif tune_mode == 3:
                parameters = parameters_ann_3
        if predictor == 'ann':
            return (parameters, classifier,classifier_wrap)
        return (parameters, classifier)
    except Exception as error:
        print('Model Build Failed with error :', error, '\n')
