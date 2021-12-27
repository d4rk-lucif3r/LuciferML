from catboost import CatBoostClassifier, CatBoostRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from luciferml.supervised.utils.tuner.optuna.objectives.classification_objectives import (
    ClassificationObjectives,
)
from luciferml.supervised.utils.tuner.optuna.objectives.regression_objectives import (
    RegressionObjectives,
)
from sklearn.ensemble import (
    AdaBoostClassifier,
    AdaBoostRegressor,
    BaggingClassifier,
    BaggingRegressor,
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import (
    BayesianRidge,
    ElasticNet,
    LinearRegression,
    LogisticRegression,
    PassiveAggressiveClassifier,
    Perceptron,
    RidgeClassifier,
    SGDClassifier,
    SGDRegressor,
)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from xgboost import XGBClassifier, XGBRegressor


def classification_predictor(
    predictor,
    params,
    X_train,
    y_train,
    cv_folds,
    random_state,
    metric,
    all_mode=False,
    verbose=False,
    lgbm_objective="binary",
):
    """
    Takes Predictor string , parameters , Training and Validation set and Returns a classifier for the Choosen Predictor.
    """
    try:
        objective = ClassificationObjectives(
            X_train,
            y_train,
            cv=cv_folds,
            random_state=random_state,
            metric=metric,
            lgbm_objective=lgbm_objective,
        )
        if predictor == "lr":
            if not all_mode:
                print("Training Logistic Regression on Training Set [*]\n")
            classifier = LogisticRegression(**params)
            objective_to_be_tuned = objective.lr_classifier_objective

        elif predictor == "sgd":
            if not all_mode:
                print("Training Stochastic Gradient Descent on Training Set [*]\n")
            classifier = SGDClassifier(**params)
            objective_to_be_tuned = objective.sgd_classifier_objective

        elif predictor == "perc":
            if not all_mode:
                print("Training Perceptron on Training Set [*]\n")
            classifier = Perceptron(**params)
            objective_to_be_tuned = objective.perc_classifier_objective

        elif predictor == "pass":
            if not all_mode:
                print("Training Passive Aggressive on Training Set [*]\n")
            classifier = PassiveAggressiveClassifier(**params)
            objective_to_be_tuned = objective.pass_classifier_objective

        elif predictor == "ridg":
            if not all_mode:
                print("Training Ridge Classifier on Training Set [*]\n")
            classifier = RidgeClassifier(**params)
            objective_to_be_tuned = objective.ridg_classifier_objective

        elif predictor == "svm":
            if not all_mode:
                print("Training Support Vector Machine on Training Set [*]\n")
            classifier = SVC(**params)
            objective_to_be_tuned = objective.svm_classifier_objective

        elif predictor == "knn":
            if not all_mode:
                print("Training K-Nearest Neighbours on Training Set [*]\n")
            classifier = KNeighborsClassifier(**params)
            objective_to_be_tuned = objective.knn_classifier_objective

        elif predictor == "dt":
            if not all_mode:
                print("Training Decision Tree Classifier on Training Set [*]\n")
            classifier = DecisionTreeClassifier(**params)
            objective_to_be_tuned = objective.dt_classifier_objective

        elif predictor == "nb":
            if not all_mode:
                print("Training Naive Bayes Classifier on Training Set [*]\n")
            classifier = GaussianNB(**params)
            objective_to_be_tuned = None

        elif predictor == "rfc":
            if not all_mode:
                print("Training Random Forest Classifier on Training Set [*]\n")
            classifier = RandomForestClassifier(**params)
            objective_to_be_tuned = objective.rfc_classifier_objective

        elif predictor == "gbc":
            if not all_mode:
                print("Training Gradient Boosting Classifier on Training Set [*]\n")
            classifier = GradientBoostingClassifier(**params)
            objective_to_be_tuned = objective.gbc_classifier_objective

        elif predictor == "ada":
            if not all_mode:
                print("Training AdaBoost Classifier on Training Set [*]\n")
            classifier = AdaBoostClassifier(**params)
            objective_to_be_tuned = objective.ada_classifier_objective

        elif predictor == "bag":
            if not all_mode:
                print("Training Bagging Classifier on Training Set [*]\n")
            classifier = BaggingClassifier(**params)
            objective_to_be_tuned = objective.bag_classifier_objective

        elif predictor == "extc":
            if not all_mode:
                print("Training Extra Trees Classifier on Training Set [*]\n")
            classifier = ExtraTreesClassifier(**params)
            objective_to_be_tuned = objective.extc_classifier_objective

        elif predictor == "lgbm":
            if not all_mode:
                print("Training LightGBM on Training Set [*]\n")
            classifier = LGBMClassifier(**params)
            objective_to_be_tuned = objective.lgbm_classifier_objective

        elif predictor == "cat":
            if not all_mode:
                print("Training CatBoostClassifier on Training Set [*]\n")
            params["verbose"] = verbose
            classifier = CatBoostClassifier(**params)
            params.pop("verbose")
            objective_to_be_tuned = objective.cat_classifier_objective

        elif predictor == "xgb":
            if not all_mode:
                print("Training XGBClassifier on Training Set [*]\n")
            classifier = XGBClassifier(**params)
            objective_to_be_tuned = objective.xgb_classifier_objective

        elif predictor == "ann":
            classifier = MLPClassifier(**params)
            objective_to_be_tuned = objective.mlp_classifier_objective
        return (classifier, objective_to_be_tuned)
    except Exception as error:
        print("Model Build Failed with error :", error, "\n")


def regression_predictor(
    predictor,
    params,
    X_train,
    y_train,
    cv_folds,
    random_state,
    metric,
    all_mode=False,
    verbose=False,
):
    """
    Takes Predictor string , parameters , Training and Validation set and Returns a regressor for the Choosen Predictor.
    """
    try:
        objective = RegressionObjectives(
            X_train, y_train, cv=cv_folds, random_state=random_state, metric=metric
        )
        if predictor == "lin":
            if not all_mode:
                print("Training Logistic Regression on Training Set [*]\n")
            regressor = LinearRegression(**params)
            objective_to_be_tuned = objective.lin_regressor_objective
        elif predictor == "sgd":
            if not all_mode:
                print(
                    "Training Stochastic Gradient Descent Regressor on Training Set [*]\n"
                )
            regressor = SGDRegressor(**params)
            objective_to_be_tuned = objective.sgd_regressor_objective
        elif predictor == "krr":
            if not all_mode:
                print("Training Kernel Ridge Regressor on Training Set [*]\n")
            regressor = KernelRidge(**params)
            objective_to_be_tuned = objective.krr_regressor_objective
        elif predictor == "elas":
            if not all_mode:
                print("Training ElasticNet Regressor on Training Set [*]\n")
            regressor = ElasticNet(**params)
            objective_to_be_tuned = objective.elas_regressor_objective
        elif predictor == "br":
            if not all_mode:
                print("Training BayesianRidge Regressor on Training Set [*]\n")
            regressor = BayesianRidge(**params)
            objective_to_be_tuned = objective.br_regressor_objective
        elif predictor == "svr":
            if not all_mode:
                print("Training Support Vector Machine on Training Set [*]\n")
            regressor = SVR(**params)
            objective_to_be_tuned = objective.svr_regressor_objective
        elif predictor == "knr":
            if not all_mode:
                print("Training KNeighbors Regressor on Training Set [*]\n")
            regressor = KNeighborsRegressor(**params)
            objective_to_be_tuned = objective.knr_regressor_objective
        elif predictor == "dt":
            if not all_mode:
                print("Training Decision Tree regressor on Training Set [*]\n")
            regressor = DecisionTreeRegressor(**params)
            objective_to_be_tuned = objective.dt_regressor_objective
        elif predictor == "rfr":
            if not all_mode:
                print("Training Random Forest regressor on Training Set [*]\n")
            regressor = RandomForestRegressor(**params)
            objective_to_be_tuned = objective.rfr_regressor_objective
        elif predictor == "gbr":
            if not all_mode:
                print("Training Gradient Boosting Regressor  on Training Set [*]\n")
            regressor = GradientBoostingRegressor(**params)
            objective_to_be_tuned = objective.gbr_regressor_objective

        elif predictor == "ada":
            if not all_mode:
                print("Training AdaBoost Regressor on Training Set [*]\n")
            regressor = AdaBoostRegressor(**params)
            objective_to_be_tuned = objective.ada_regressor_objective
        elif predictor == "bag":
            if not all_mode:
                print("Training Bagging Regressor on Training Set [*]\n")
            regressor = BaggingRegressor(**params)
            objective_to_be_tuned = objective.bag_regressor_objective
        elif predictor == "extr":
            if not all_mode:
                print("Training Extra Trees Regressor on Training Set [*]\n")
            regressor = ExtraTreesRegressor(**params)
            objective_to_be_tuned = objective.extr_regressor_objective
        elif predictor == "xgb":
            if not all_mode:
                print("Training XGBregressor on Training Set [*]\n")
            regressor = XGBRegressor(**params)
            objective_to_be_tuned = objective.xgb_regressor_objective
        elif predictor == "lgbm":
            if not all_mode:
                print("Training LGBMRegressor on Training Set [*]\n")
            regressor = LGBMRegressor(**params)
            objective_to_be_tuned = objective.lgbm_regressor_objective
        elif predictor == "cat":
            if not all_mode:
                print("Training CatBoost Regressor on Training Set [*]\n")
            params["verbose"] = verbose
            regressor = CatBoostRegressor(**params)
            params.pop("verbose")
            objective_to_be_tuned = objective.cat_regressor_objective
        elif predictor == "ann":
            if not all_mode:
                print("Training Multi Layered Perceptron on Training Set [*]\n")
            regressor = MLPRegressor(**params)
            objective_to_be_tuned = objective.mlp_regressor_objective
        return (regressor, objective_to_be_tuned)
    except Exception as error:
        print("Model Build Failed with error :", error, "\n")
