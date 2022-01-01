# Regression

Encodes Categorical Data then Applies SMOTE , Splits the features and labels in training and validation sets with test_size = .2
scales X_train, X_val using StandardScaler.
Fits every model on training set and predicts results,Finds R2 Score and mean square error
finds accuracy of model applies K-Fold Cross Validation
and stores its accuracies in a dictionary containing Model name as Key and accuracies as values and returns it
Applies HyperParam Tuning and gives best params and accuracy.

Parameters:

        features : array
                features array
        lables : array
                labels array
        predictor : str
                Predicting model to be used
                Default 'lin'
                Available Predictors:
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
                        ann  - Multi Layer Perceptron Regressor
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
