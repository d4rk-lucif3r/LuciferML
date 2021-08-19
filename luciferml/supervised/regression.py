import time

import numpy as np
import pandas as pd
from IPython.display import display
from luciferml.supervised.utils.configs import regressors
from luciferml.supervised.utils.dimensionalityReduction import \
    dimensionalityReduction
from luciferml.supervised.utils.encoder import encoder
from luciferml.supervised.utils.hyperTune import hyperTune
from luciferml.supervised.utils.intro import intro
from luciferml.supervised.utils.kfold import kfold
from luciferml.supervised.utils.predPreprocess import pred_preprocess
from luciferml.supervised.utils.regressionPredictor import regressionPredictor
from luciferml.supervised.utils.sparseCheck import sparseCheck
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


class Regression:

    def __init__(self,
                 predictor='lin',
                 params={},
                 tune=False,
                 test_size=.2,
                 cv_folds=10,
                 random_state=42,
                 pca_kernel='linear',
                 n_components_lda=1,
                 lda='n', pca='n',
                 n_components_pca=2,
                 hidden_layers=4,
                 output_units=1,
                 input_units=6,
                 input_activation='relu',
                 optimizer='adam',
                 loss='mean_squared_error',
                 validation_split=.20,
                 epochs=100,
                 batch_size=32,
                 tune_mode=1,
                 smote='n',
                 k_neighbors=1,
                 dropout_rate=0,
                 verbose=False
                 ):
        """
        Encodes Categorical Data then Applies SMOTE , Splits the features and labels in training and validation sets with test_size = .2
        scales X_train, X_val using StandardScaler.
        Fits every model on training set and predicts results,Finds R2 Score and mean square error
        finds accuracy of model applies K-Fold Cross Validation
        and stores its accuracies in a dictionary containing Model name as Key and accuracies as values and returns it
        Applies GridSearch Cross Validation and gives best params out from param list.


        Parameters:
            features : array
                        features array

            lables : array
                        labels array

            predictor : str
                    Predicting model to be used
                    Default 'lin'
                        Predictor Strings:
                                lin  - Linear Regression
                                sgd  - Stochastic Gradient Descent Regressor
                                elas - Elastic Net Regressor
                                krr  - Kernel Ridge Regressor
                                br   - Bayesian Ridge Regressor
                                svr  - Support Vector Regressor
                                knr  - K-Nearest Regressor
                                dt   - Decision Trees
                                rfr  - Random Forest Regressor
                                gbr  - Gradient Boost Regressor
                                ada  - AdaBoost Regressor,
                                bag  - Bagging Regressor,
                                extr - Extra Trees Regressor,
                                lgbm - LightGB Regressor
                                xgb  - XGBoost Regressor
                                cat  - Catboost Regressor
                                ann  - Artificial Neural Network
                                all  - Applies all above regressors
            params : dict
                        contains parameters for model
            tune : boolean
                    when True Applies GridSearch CrossValidation
                    Default is False

            test_size: float or int, default=.2
                        If float, should be between 0.0 and 1.0 and represent
                        the proportion of the dataset to include in
                        the test split.
                        If int, represents the absolute number of test samples.

            cv_folds : int
                    No. of cross validation folds. Default = 10
            pca : str
                if 'y' will apply PCA on Train and Validation set. Default = 'n'
            lda : str
                if 'y' will apply LDA on Train and Validation set. Default = 'n'
            pca_kernel : str
                    Kernel to be use in PCA. Default = 'linear'
            n_components_lda : int
                    No. of components for LDA. Default = 1
            n_components_pca : int
                    No. of components for PCA. Default = 2
            hidden_layers : int
                    No. of default layers of ann. Default = 4
            inputs_units : int
                    No. of units in input layer. Default = 6
            output_units : int
                    No. of units in output layer. Default = 6
            self.input_activation : str
                    Activation function for Hidden layers. Default = 'relu'
            optimizer: str
                    Optimizer for ann. Default = 'adam'
            loss : str
                    loss method for ann. Default = 'mean_squared_error'
            validation_split : float or int
                    Percentage of validation set splitting in ann. Default = .20
            epochs : int
                    No. of epochs for ann. Default = 100
            dropout_rate : int or float
                    rate for dropout layer. Default = 0 
            batch_size :
                    Batch Size for ANN. Default = 32
            tune_mode : int
                    HyperParam tune modes. Default = 1
                        Available Modes:
                            1 : Basic Tune
                            2 : Intermediate Tune
                            3 : Extreme Tune (Can Take Much Time)
            smote : str,
                Whether to apply SMOTE. Default = 'y'
            k_neighbors : int
                No. of neighbours for SMOTE. Default = 1
            verbose : boolean
                Verbosity of models. Default = False
        Returns:
        
            Dict Containing Name of Regressor, Its K-Fold Cross Validated Accuracy, RMSE, Prediction set
            
            Dataframe containing all the models and their accuracies when predictor is 'all'

        Example:

            from luciferml.supervised.regression import Regression

            dataset = pd.read_excel('examples\Folds5x2_pp.xlsx')

            X = dataset.iloc[:, :-1]

            y = dataset.iloc[:, -1]

            regressor = Regression(predictor = 'lin')

            regressor.fit(X, y)

            result = regressor.result()

        """

        self.predictor = predictor
        self.params = params
        self.tune = tune
        self.test_size = test_size
        self.cv_folds = cv_folds
        self.random_state = random_state
        self.pca_kernel = pca_kernel
        self.n_components_lda = n_components_lda
        self.lda = lda
        self.pca = pca
        self.n_components_pca = n_components_pca
        self.hidden_layers = hidden_layers
        self.output_units = output_units
        self.input_units = input_units
        self.input_activation = input_activation
        self.optimizer = optimizer
        self.loss = loss
        self.validation_split = validation_split
        self.epochs = epochs
        self.batch_size = batch_size
        self.dropout_rate = dropout_rate
        self.tune_mode = tune_mode
        self.rerun = False
        self.smote = smote
        self.k_neighbors = k_neighbors
        self.verbose = verbose

        self.accuracy_scores = {}
        self.reg_result = {}
        self.rm_squared_error = 0
        self.accuracy = 0
        self.y_pred = []
        self.kfold_accuracy = 0
        self.regressor_name = ''
        self.sc = 0

        self.kfoldacc = []
        self.acc = []
        self.mae = []
        self.rmse = []
        self.bestacc = []
        self.bestparams = []
        self.result_df = pd.DataFrame(index=regressors.values())

        self.pred_mode= ''

    def fit(self, features, labels):
        """[Takes Features and Labels and Encodes Categorical Data then Applies SMOTE , Splits the features and labels in training and validation sets with test_size = .2
        scales X_train, X_val using StandardScaler.
        Fits model on training set and predicts results, Finds R2 Score and mean square error
        finds accuracy of model applies K-Fold Cross Validation
        and stores its accuracies in a dictionary containing Model name as Key and accuracies as values and returns it
        Applies GridSearch Cross Validation and gives best params out from param list.]

        Args:
            features ([Pandas DataFrame]): [DataFrame containing Features]
            labels ([Pandas DataFrame]): [DataFrame containing Labels]
        """

        self.features = features
        self.labels = labels

        # Time Function ---------------------------------------------------------------------

        self.start = time.time()
        intro()
        print("Started Lucifer-ML \n")
        if not self.rerun:
            # CHECKUP ---------------------------------------------------------------------
            if not isinstance(self.features, pd.DataFrame) and not isinstance(self.labels, pd.Series):
                print('TypeError: This Function take features as Pandas Dataframe and labels as Pandas Series. Please check your implementation.\n')
                end = time.time()
                print(self.end - self.start)
                return

            # Encoding ---------------------------------------------------------------------

            self.features, self.labels = encoder(self.features, self.labels)

            # Sparse Check -------------------------------------------------------------
            self.features, self.labels = sparseCheck(
                self.features, self.labels)

            # Preprocessing ---------------------------------------------------------------------
            self.X_train, self.X_val, self.y_train, self.y_val, self.sc = pred_preprocess(
                self.features, self.labels, self.test_size, self.random_state, self.smote, self.k_neighbors)

            # Dimensionality Reduction---------------------------------------------------------------------
            self.X_train, self.X_val = dimensionalityReduction(
                self.lda, self.pca, self.X_train, self.X_val, self.y_train,
                self.n_components_lda, self.n_components_pca, self.pca_kernel, self.start)

        # Models ---------------------------------------------------------------------
        if self.predictor == 'all':
            self.pred_mode = 'all'
            self._fitall()
        if self.predictor == 'ann':
            self.parameters, self.regressor, self.regressor_wrap = regressionPredictor(
                self.predictor, self.params, self.X_train, self.X_val, self.y_train, self.y_val, self.epochs, self.hidden_layers,
                self.input_activation, self.loss,
                self.batch_size, self.validation_split, self.optimizer, self.output_units, self.input_units, self.tune_mode, self.dropout_rate, verbose=self.verbose
            )
        else:
            self.parameters, self.regressor = regressionPredictor(
                self.predictor, self.params, self.X_train, self.X_val, self.y_train, self.y_val, self.epochs, self.hidden_layers,
                self.input_activation,  self.loss,
                self.batch_size, self.validation_split, self.optimizer, self.output_units, self.input_units, self.tune_mode,
                verbose=self.verbose
            )

        try:

            if not self.predictor == 'ann':
                self.regressor.fit(self.X_train, self.y_train)
        except Exception as error:
            print('Model Train Failed with error: ', error, '\n')

        print('Model Training Done [', u'\u2713', ']\n')
        print('Predicting Data [*]\n')
        try:
            self.y_pred = self.regressor.predict(self.X_val)
            print('Data Prediction Done [', u'\u2713', ']\n')
        except Exception as error:
            print('Prediction Failed with error: ', error,  '\n')

        # Accuracy ---------------------------------------------------------------------
        print('''Evaluating Model Performance [*]''')
        try:
            self.accuracy = r2_score(self.y_val, self.y_pred)
            self.m_absolute_error = mean_absolute_error(
                self.y_val, self.y_pred)
            self.rm_squared_error = mean_squared_error(
                self.y_val, self.y_pred, squared=False)
            print('Validation R2 Score is {:.2f} %'.format(self.accuracy*100))
            print('Validation Mean Absolute Error is :',
                  self.m_absolute_error)
            print('Validation Root Mean Squared Error is :',
                  self.rm_squared_error, '\n')
            print('Evaluating Model Performance [', u'\u2713', ']\n')
        except Exception as error:
            print('Model Evaluation Failed with error: ', error, '\n')

        # K-Fold ---------------------------------------------------------------------
        if self.predictor == 'ann':
            self.regressor_name, self.kfold_accuracy = kfold(
                self.regressor_wrap,
                self.predictor, self.X_train, self.y_train, self.cv_folds, isReg=True
            )
        else:
            self.regressor_name, self.kfold_accuracy = kfold(
                self.regressor,
                self.predictor,
                self.X_train, self.y_train, self.cv_folds, isReg=True
            )

        # GridSearch ---------------------------------------------------------------------
        if not self.predictor == 'nb' and self.tune:
            self.__tuner()

        print('Complete [', u'\u2713', ']\n')
        self.end = time.time()
        print('Time Elapsed : ', self.end - self.start, 'seconds \n')

    def _fitall(self):
        print('Training All Regressors [*]')
        for index, self.predictor in enumerate(regressors):
            if not self.predictor == 'ann':
                self.parameters, self.regressor = regressionPredictor(
                    self.predictor, self.params, self.X_train, self.X_val, self.y_train, self.y_val, self.epochs, self.hidden_layers,
                    self.input_activation, self.loss,
                    self.batch_size, self.validation_split, self.optimizer, self.output_units, self.input_units, self.tune_mode, all_mode=True, verbose=self.verbose
                )
            elif self.predictor == 'ann':
                self.parameters, self.regressor, self.regressor_wrap = regressionPredictor(
                    self.predictor, self.params, self.X_train, self.X_val, self.y_train, self.y_val, self.epochs, self.hidden_layers,
                    self.input_activation, self.loss,
                    self.batch_size, self.validation_split, self.optimizer,
                    self.output_units, self.input_units, self.tune_mode, self.dropout_rate, all_mode=True, verbose=self.verbose
                )
            try:

                if not self.predictor == 'ann':
                    self.regressor.fit(self.X_train, self.y_train)
            except Exception as error:
                print(regressors[self.predictor],
                      'Model Train Failed with error: ', error, '\n')
            try:
                self.y_pred = self.regressor.predict(self.X_val)
            except Exception as error:
                print(regressors[self.predictor],
                      'Data Prediction Failed with error: ', error,  '\n')
            if self.predictor == 'ann':
                self.y_pred = (self.y_pred > 0.5).astype("int32")

            # Accuracy ---------------------------------------------------------------------
            try:
                self.accuracy = r2_score(self.y_val, self.y_pred)
                self.m_absolute_error = mean_absolute_error(
                    self.y_val, self.y_pred)
                self.rm_squared_error = mean_squared_error(
                    self.y_val, self.y_pred, squared=False)
                self.acc.append(self.accuracy*100)
                self.rmse.append(self.rm_squared_error)
                self.mae.append(self.m_absolute_error)
            except Exception as error:
                print(regressors[self.predictor],
                      'Evaluation Failed with error: ', error, '\n')

            # K-Fold ---------------------------------------------------------------------
            if self.predictor == 'ann':
                self.regressor_name, self.kfold_accuracy = kfold(
                    self.regressor_wrap,
                    self.predictor, self.X_train, self.y_train, self.cv_folds, all_mode=True, isReg=True
                )
            else:
                self.regressor_name, self.kfold_accuracy = kfold(
                    self.regressor,
                    self.predictor,
                    self.X_train, self.y_train, self.cv_folds, all_mode=True, isReg=True
                )
            self.kfoldacc.append(self.kfold_accuracy)
            # GridSearch ---------------------------------------------------------------------
            if not self.predictor == 'nb' and self.tune:
                self.__tuner(all_mode=True)
            if self.predictor == 'nb':
                self.best_params = ''
                self.best_accuracy = self.kfold_accuracy
        self.result_df['R2 Score'] = self.acc
        self.result_df['Mean Absolute Error'] = self.mae
        self.result_df['Root Mean Squared Error'] = self.rmse
        self.result_df['KFold Accuracy'] = self.kfoldacc
        if self.tune:
            self.result_df['Best Parameters'] = self.bestparams
            self.result_df['Best Accuracy'] = self.bestacc

        display(self.result_df)
        print('Complete [', u'\u2713', ']\n')
        self.end = time.time()
        print('Time Elapsed : ', self.end - self.start, 'seconds \n')
        return

    def __tuner(self, all_mode=False):
        if not all_mode:
            print(
                'Applying Grid Search Cross validation on Mode {} [*]'.format(self.tune_mode))
        if self.predictor == 'ann':
            self.best_params, self.best_accuracy = hyperTune(
                self.regressor_wrap, self.parameters, self.X_train, self.y_train, self.cv_folds, self.tune_mode, all_mode=all_mode, isReg=True)
        else:
            self.best_params, self.best_accuracy = hyperTune(
                self.regressor, self.parameters, self.X_train, self.y_train, self.cv_folds, self.tune_mode, all_mode=all_mode, isReg=True)
        self.bestparams.append(self.best_params)
        self.bestacc.append(self.best_accuracy*100)

    def result(self):
        """[Makes a dictionary containing Regressor Name, K-Fold CV Accuracy, RMSE, Prediction set.]

        Returns:
            [dict]: [Dictionary containing :
                        - "Regressor" - Regressor Name
                        - "Accuracy" - KFold CV Accuracy
                        - "RMSE" - Root Mean Square
                        - "YPred" - Array for Prediction set
                        ]
            [dataframe] : [Dataset containing accuracy and best_params 
                            for all predictors only when predictor = 'all' is used
                            ]
        """
        if not self.pred_mode == 'all':
            self.reg_result['Regressor'] = self.regressor_name
            self.reg_result['Accuracy'] = self.kfold_accuracy
            self.reg_result['RMSE'] = self.rm_squared_error
            self.reg_result['YPred'] = self.y_pred

            return self.reg_result
        if self.pred_mode == 'all':
            return self.result_df

    def predict(self, X_test):
        """[Takes test set and returns predictions for that test set]

        Args:
            X_test ([Array]): [Array Containing Test Set]

        Returns:
            [Array]: [Predicted set for given test set]
        """
        if not self.pred_mode == 'all':
            X_test = np.array(X_test)
            if X_test.ndim == 1:
                print('''Input Array has only 1-Dimension but 2-Dimension was expected. 
                    Reshaping it to 2-Dimension''')
                X_test = X_test.reshape(1, -1)

            y_test = self.regressor.predict(self.sc.transform(X_test))

            return y_test
        if self.pred_mode == 'all':
            print('[Error] This method is only applicable on single predictor')
            return
