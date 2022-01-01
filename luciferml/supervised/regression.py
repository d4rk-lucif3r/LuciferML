import copy
import os
import time
import warnings
from pickle import dump, load

import numpy as np
import optuna
import pandas as pd
from colorama import Fore
from IPython.display import display
from luciferml.supervised.utils.best import Best
from luciferml.supervised.utils.configs import *
from luciferml.supervised.utils.predictors import regression_predictor
from luciferml.supervised.utils.preprocesser import PreProcesser
from luciferml.supervised.utils.tuner.luciferml_tuner import luciferml_tuner
from luciferml.supervised.utils.validator import *
from optuna.samplers._tpe.sampler import TPESampler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


class Regression:
    def __init__(
        self,
        predictor=["lin"],
        params={},
        tune=False,
        test_size=0.2,
        cv_folds=10,
        random_state=42,
        pca_kernel="linear",
        n_components_lda=1,
        lda="n",
        pca="n",
        n_components_pca=2,
        loss="mean_squared_error",
        validation_split=0.20,
        smote="n",
        k_neighbors=1,
        verbose=False,
        exclude_models=[],
        path=None,
        optuna_sampler=TPESampler(),
        optuna_direction="maximize",
        optuna_n_trials=100,
        optuna_metric="r2",
    ):
        """
        Encodes Categorical Data then Applies SMOTE , Splits the features and labels in training and validation sets with test_size = .2\n
        scales X_train, X_val using StandardScaler.\n
        Fits every model on training set and predicts results,Finds R2 Score and mean square error\n
        finds accuracy of model applies K-Fold Cross Validation\n
        and stores its accuracies in a dictionary containing Model name as Key and accuracies as values and returns it\n
        Applies HyperParam Tuning and gives best params and accuracy.\n

        Parameters:

            features : array
                        features array
            lables : array
                        labels array
            predictor : list
                        Predicting models to be used
                        Default ['lin'] - 'Linear Regression'\n
                        Available Predictors:
                                lin  - Linear Regression\n
                                sgd  - Stochastic Gradient Descent Regressor\n
                                elas - Elastic Net Regressor\n
                                krr  - Kernel Ridge Regressor\n
                                br   - Bayesian Ridge Regressor\n
                                svr  - Support Vector Regressor\n
                                knr  - K-Nearest Regressor\n
                                dt   - Decision Trees\n
                                rfr  - Random Forest Regressor\n
                                gbr  - Gradient Boost Regressor\n
                                ada  - AdaBoost Regressor,\n
                                bag  - Bagging Regressor,\n
                                extr - Extra Trees Regressor,\n
                                lgbm - LightGB Regressor\n
                                xgb  - XGBoost Regressor\n
                                cat  - Catboost Regressor\n
                                ann  - Multi Layer Perceptron Regressor\n
                                all  - Applies all above regressors\n
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
            loss : str
                    loss method for ann. Default = 'mean_squared_error'
            smote : str,
                Whether to apply SMOTE. Default = 'y'
            k_neighbors : int
                No. of neighbours for SMOTE. Default = 1
            verbose : boolean
                Verbosity of models. Default = False
            exclude_models : list
                List of models to be excluded when using predictor = 'all' . Default = []
            path : list
                List containing path to saved model and scaler. Default = None
                Example: [model.pkl, scaler.pkl]
            random_state : int
                Random random_state for reproducibility. Default = 42
            optuna_sampler : Function
                Sampler to be used in optuna. Default = TPESampler()
            optuna_direction : str
                Direction of optimization. Default = 'maximize'
                Available Directions:
                    maximize : Maximize
                    minimize : Minimize
            optuna_n_trials : int
                No. of trials for optuna. Default = 100
            optuna_metric: str
                Metric to be used in optuna. Default = 'r2'
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
        self.preprocess = PreProcesser()
        if type(predictor) == list:
            self.predictor = predictor[0] if len(predictor) == 1 else predictor
        else:
            self.predictor = predictor
        bool_pred, pred = pred_check(predictor, pred_type="regression")
        if not bool_pred:
            raise ValueError(unsupported_pred_warning.format(pred))
        self.original_predictor = predictor
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
        self.loss = loss
        self.validation_split = validation_split
        self.rerun = False
        self.smote = smote
        self.k_neighbors = k_neighbors
        self.verbose = verbose
        self.exclude_models = exclude_models
        self.sampler = optuna_sampler
        self.direction = optuna_direction
        self.n_trials = optuna_n_trials
        self.metric = optuna_metric

        self.accuracy_scores = {}
        self.reg_result = {}
        self.rm_squared_error = 0
        self.accuracy = 0
        self.y_pred = []
        self.kfold_accuracy = 0
        self.regressor_name = ""
        self.sc = 0

        self.kfoldacc = []
        self.acc = []
        self.mae = []
        self.rmse = []
        self.bestacc = []
        self.bestparams = []
        self.regressor_model = []
        self.tuned_trained_model = []
        self.best_regressor_path = ""
        self.scaler_path = ""
        self.result_df = pd.DataFrame(index=None)
        self.regressors = copy.deepcopy(regressors)
        for i in self.exclude_models:
            self.regressors.pop(i)
        self.best_regressor = "First Run the Predictor in All mode"
        self.objective = None
        self.pred_mode = ""
        self.model_to_predict = None

        if path != None:
            try:
                self.regressor, self.sc = self.__load(path)
            except Exception as e:
                print(Fore.RED + e)
                print(Fore.RED + "Model not found")
        if not self.verbose:
            optuna.logging.set_verbosity(optuna.logging.WARNING)

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
        print(Fore.LIGHTBLACK_EX + intro, "\n")
        print(Fore.GREEN + "Started LuciferML [", "\u2713", "]\n")
        if not self.rerun:
            # CHECKUP ---------------------------------------------------------------------
            if not isinstance(self.features, pd.DataFrame) and not isinstance(
                self.labels, pd.Series
            ):
                print(
                    Fore.RED
                    + "TypeError: This Function take features as Pandas Dataframe and labels as Pandas Series. Please check your implementation.\n"
                )
                end = time.time()
                print(self.end - self.start)
                return
            print(Fore.YELLOW + "Preprocessing Started [*]\n")

            self.features, self.labels = self.preprocess.encoder(
                self.features, self.labels
            )
            self.features, self.labels = sparse_check(self.features, self.labels)
            (
                self.X_train,
                self.X_val,
                self.y_train,
                self.y_val,
                self.sc,
            ) = self.preprocess.data_preprocess(
                self.features,
                self.labels,
                self.test_size,
                self.random_state,
                self.smote,
                self.k_neighbors,
            )
            self.X_train, self.X_val = self.preprocess.dimensionality_reduction(
                self.lda,
                self.pca,
                self.X_train,
                self.X_val,
                self.y_train,
                self.n_components_lda,
                self.n_components_pca,
                self.pca_kernel,
                self.start,
            )

        print(Fore.GREEN + "Preprocessing Done [", "\u2713", "]\n")

        if self.original_predictor == "all" or type(self.predictor) == list:
            self.model_to_predict = (
                self.predictor if type(self.predictor) == list else [self.regressors]
            )
            self.result_df["Name"] = (
                list(self.regressors.values())
                if not type(self.predictor) == list
                else list(self.regressors[i] for i in self.predictor)
            )
            self.pred_mode = "all" if len(self.predictor) > 1 else "single"
            self.__fitall()
            return
        self.regressor, self.objective = regression_predictor(
            self.predictor,
            self.params,
            self.X_train,
            self.y_train,
            self.cv_folds,
            self.random_state,
            self.metric,
            verbose=self.verbose,
        )
        try:
            self.regressor.fit(self.X_train, self.y_train)
        except Exception as error:
            print(Fore.RED + "Model Train Failed with error: ", error, "\n")
        finally:
            print(Fore.GREEN + "Model Trained Successfully [", "\u2713", "]\n")

        try:
            print(Fore.YELLOW + "Evaluating Model Performance [*]\n")
            self.y_pred = self.regressor.predict(self.X_val)
            self.accuracy = r2_score(self.y_val, self.y_pred)
            self.m_absolute_error = mean_absolute_error(self.y_val, self.y_pred)
            self.rm_squared_error = mean_squared_error(
                self.y_val, self.y_pred, squared=False
            )
            print(
                Fore.CYAN
                + "        Validation R2 Score is {:.2f} %".format(self.accuracy * 100)
            )
            print(
                Fore.CYAN + "        Validation Mean Absolute Error is :",
                self.m_absolute_error,
            )
            print(
                Fore.CYAN + "        Validation Root Mean Squared Error is :",
                self.rm_squared_error,
            )
            self.regressor_name, self.kfold_accuracy = kfold(
                self.regressor,
                self.predictor,
                self.X_train,
                self.y_train,
                self.cv_folds,
                isReg=True,
            )
        except Exception as error:
            print(Fore.RED + "Model Evaluation Failed with error: ", error, "\n")
        finally:
            print(Fore.GREEN + "Model Evaluation Completed [", "\u2713", "]\n")

        if not self.predictor == "nb" and self.tune:
            self.__tuner()

        print(Fore.GREEN + "Completed LuciferML Run [", "\u2713", "]\n")
        self.end = time.time()
        final_time = self.end - self.start
        print(Fore.BLUE + "Time Elapsed : ", f"{final_time:.2f}", "seconds \n")

    def __fitall(self):
        print(Fore.YELLOW + "Training LuciferML [*]\n")
        if self.params != {}:
            warnings.warn(params_use_warning, UserWarning)
            self.params = {}
        for _, self.predictor in enumerate(self.model_to_predict):
            if not self.predictor in self.exclude_models:
                (self.regressor, self.objective,) = regression_predictor(
                    self.predictor,
                    self.params,
                    self.X_train,
                    self.y_train,
                    self.cv_folds,
                    self.random_state,
                    self.metric,
                    mode="multi",
                    verbose=self.verbose,
                )
                try:
                    self.regressor.fit(self.X_train, self.y_train)
                except Exception as error:
                    print(
                        Fore.RED + regressors[self.predictor],
                        "Model Train Failed with error: ",
                        error,
                        "\n",
                    )
                try:
                    self.y_pred = self.regressor.predict(self.X_val)
                    self.accuracy = r2_score(self.y_val, self.y_pred)
                    self.m_absolute_error = mean_absolute_error(self.y_val, self.y_pred)
                    self.rm_squared_error = mean_squared_error(
                        self.y_val, self.y_pred, squared=False
                    )
                    self.acc.append(self.accuracy * 100)
                    self.rmse.append(self.rm_squared_error)
                    self.mae.append(self.m_absolute_error)
                    self.regressor_name, self.kfold_accuracy = kfold(
                        self.regressor,
                        self.predictor,
                        self.X_train,
                        self.y_train,
                        self.cv_folds,
                        all_mode=True,
                        isReg=True,
                    )
                    self.kfoldacc.append(self.kfold_accuracy)
                    self.regressor_model.append(self.regressor)
                except Exception as error:
                    print(
                        Fore.RED + regressors[self.predictor],
                        "Evaluation Failed with error: ",
                        error,
                        "\n",
                    )

                if self.tune:
                    self.__tuner(all_mode=True, single_mode=False)
                if self.predictor == "nb":
                    self.best_params = ""
                    self.best_accuracy = self.kfold_accuracy
        self.result_df["R2 Score"] = self.acc
        self.result_df["Mean Absolute Error"] = self.mae
        self.result_df["Root Mean Squared Error"] = self.rmse
        self.result_df["KFold Accuracy"] = self.kfoldacc
        self.result_df["Model"] = self.regressor_model

        if self.tune:
            self.result_df["Best Parameters"] = self.bestparams
            self.result_df["Best Accuracy"] = self.bestacc
            self.result_df["Trained Model"] = self.tuned_trained_model
            self.best_regressor = Best(
                self.result_df.loc[self.result_df["Best Accuracy"].idxmax()],
                self.tune,
                isReg=True,
            )
        else:
            self.best_regressor = Best(
                self.result_df.loc[self.result_df["KFold Accuracy"].idxmax()],
                self.tune,
                isReg=True,
            )
        print(Fore.GREEN + "Training Done [", "\u2713", "]\n")
        print(Fore.CYAN + "Results Below\n")
        display(self.result_df)
        print(Fore.GREEN + "\nCompleted LuciferML Run [", "\u2713", "]\n")
        if len(self.model_to_predict) > 1:
            self.best_regressor_path, self.scaler_path = self.save(
                best=True, model=self.best_regressor.model, scaler=self.sc
            )
            print(
                Fore.CYAN
                + "Saved Best Model to {} and its scaler to {}".format(
                    self.best_regressor_path, self.scaler_path
                ),
                "\n",
            )
        self.end = time.time()
        final_time = self.end - self.start
        print(Fore.BLUE + "Time Elapsed : ", f"{final_time:.2f}", "seconds \n")
        return

    def __tuner(self, all_mode=False, single_mode=True):
        if not all_mode or single_mode:
            print(Fore.YELLOW + "Tuning Started [*]\n")
        if not self.predictor == "nb":
            (
                self.best_params,
                self.best_accuracy,
                self.best_trained_model,
            ) = luciferml_tuner(
                self.predictor,
                self.objective,
                self.n_trials,
                self.sampler,
                self.direction,
                self.X_train,
                self.y_train,
                self.cv_folds,
                self.random_state,
                self.metric,
                all_mode=all_mode,
                isReg=True,
            )
        if self.predictor == "nb":
            self.best_params = "Not Applicable"
            self.best_accuracy = 0
        self.bestparams.append(self.best_params)
        self.bestacc.append(self.best_accuracy * 100)
        self.tuned_trained_model.append(self.best_trained_model)
        if not all_mode or single_mode:
            print(Fore.GREEN + "Tuning Done [", "\u2713", "]\n")

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
        if not self.pred_mode == "all":
            self.reg_result["Regressor"] = self.regressor_name
            self.reg_result["Accuracy"] = self.kfold_accuracy
            self.reg_result["RMSE"] = self.rm_squared_error
            self.reg_result["YPred"] = self.y_pred

            return self.reg_result
        if self.pred_mode == "all":
            return self.result_df

    def predict(self, X_test):
        """[Takes test set and returns predictions for that test set]

        Args:
            X_test ([Array]): [Array Containing Test Set]

        Returns:
            [Array]: [Predicted set for given test set]
        """
        if not self.pred_mode == "all":
            X_test = np.array(X_test)
            if X_test.ndim == 1:
                X_test = X_test.reshape(1, -1)

            y_test = self.regressor.predict(self.sc.transform(X_test))

            return y_test
        if self.pred_mode == "all":
            raise TypeError("Predict is only applicable on single predictor")

    def save(self, path=None, best=False, **kwargs):
        """
        Saves the model and its scaler to a file provided with a path.
        If no path is provided will create a directory named
        lucifer_ml_info/models/ and lucifer_ml_info/scaler/ in current working directory

        Args:

            path ([list]): [List containing path to save the model and scaler.]
                Example: path = ["model.pkl", "scaler.pkl"]

        Returns:

            Path to the saved model and its scaler.
        """
        if not type(path) == list and path != None:
            raise TypeError("Path must be a list")
        if self.pred_mode == "all" and best == False:
            raise TypeError("Cannot save model for all predictors")
        dir_path_model = path[0] if path else "lucifer_ml_info/models/regression/"
        dir_path_scaler = path[1] if path else "lucifer_ml_info/scalers/regression/"
        model_name = regressors[self.predictor].replace(" ", "_")
        if best:
            dir_path_model = "lucifer_ml_info/best/regression/models/"
            dir_path_scaler = "lucifer_ml_info/best/regression/scalers/"
            model_name = self.best_regressor.name.replace(" ", "_")
        os.makedirs(dir_path_model, exist_ok=True)
        os.makedirs(dir_path_scaler, exist_ok=True)
        timestamp = str(int(time.time()))
        path_model = dir_path_model + model_name + "_" + timestamp + ".pkl"
        path_scaler = (
            dir_path_scaler + model_name + "_" + "Scaler" + "_" + timestamp + ".pkl"
        )
        if (
            not kwargs.get("model")
            and not kwargs.get("best")
            and not kwargs.get("scaler")
        ):
            dump(self.regressor, open(path_model, "wb"))
            dump(self.sc, open(path_scaler, "wb"))
        else:
            dump(kwargs.get("model"), open(path_model, "wb"))
            dump(kwargs.get("scaler"), open(path_scaler, "wb"))
        if not best:
            print("Model Saved at {} and Scaler at {}".format(path_model, path_scaler))
        return path_model, path_scaler

    def __load(self, path=None):
        """
        Loads model and scaler from the specified path

        Args:

            path ([list]): [List containing path to load the model and scaler.]
                Example: path = ["model.pkl", "scaler.pkl"]

        Returns:
            [Model] : [Loaded model]
            [Scaler] : [Loaded scaler]
        """
        model_path = path[0] if path[0] else None
        scaler_path = path[1] if path[1] else None
        if not ".pkl" in model_path and not model_path == None:
            raise TypeError(
                "[Error] Model Filetype not supported. Please use .pkl type "
            )
        if not ".pkl" in scaler_path and not scaler_path == None:
            raise TypeError(
                "[Error] Scaler Filetype not supported. Please use .pkl type "
            )
        if model_path != None and scaler_path != None:
            model = load(open(model_path, "rb"))
            scaler = load(open(scaler_path, "rb"))
            print(
                Fore.GREEN
                + "[Info] Model and Scaler Loaded from {} and {}".format(
                    model_path, scaler_path
                )
            )
            return model, scaler
        elif model_path != None and scaler_path == None:
            model = load(open(model_path, "rb"))
            print(Fore.GREEN + "[Info] Model Loaded from {}".format(model_path))
            return model
        elif model_path == None and scaler_path != None:
            scaler = load(open(scaler_path, "rb"))
            print(Fore.GREEN + "[Info] Scaler Loaded from {}".format(scaler_path))
            return scaler
        else:
            raise ValueError("No path specified.Please provide actual path\n")

    def imp_features(self, extensive=False, *args, **kwargs):
        """
        Returns the importance features of the dataset

        Args:

            extensive (bool): [If True shows the importance of all features exitensively and will take more time] [default = False]
            **args: [Additional arguments]
            **kwargs: [Additional keyword arguments]
        """
        if self.original_predictor == "all":
            raise TypeError(
                "[Error] This method is only applicable on single predictor"
            )
        if not extensive:
            self.preprocess.permutational_feature_imp(
                self.features, self.X_train, self.y_train, model=self.regressor
            )
        if extensive:
            self.preprocess.shap_feature_imp(
                self.features, self.X_train, model=self.regressor, *args, **kwargs
            )
