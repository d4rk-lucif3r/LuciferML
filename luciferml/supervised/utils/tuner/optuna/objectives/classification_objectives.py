from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import (
    AdaBoostClassifier,
    BaggingClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import (
    LogisticRegression,
    PassiveAggressiveClassifier,
    Perceptron,
    RidgeClassifier,
    SGDClassifier,
)
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier


class ClassificationObjectives:
    def __init__(
        self, X, y, cv=5, random_state=42, metric="accuracy", lgbm_objective="binary"
    ):
        self.metric = metric
        self.cv = cv
        self.X = X
        self.y = y
        self.random_state = random_state
        self.lgbm_objective = lgbm_objective

    def lr_classifier_objective(self, trial):
        param = {
            "C": trial.suggest_loguniform("C", 1e-5, 1e5),
            "solver": trial.suggest_categorical(
                "solver", ["newton-cg", "lbfgs", "liblinear", "sag", "saga"]
            ),
            "max_iter": trial.suggest_int("max_iter", 1, 1000),
            "tol": trial.suggest_loguniform("tol", 1e-5, 1e-2),
            "random_state": self.random_state,
        }
        clf = LogisticRegression(**param)
        scores = cross_val_score(
            clf,
            self.X,
            self.y,
            cv=self.cv,
            scoring=self.metric,
            n_jobs=-1,
        )
        return scores.mean()

    def sgd_classifier_objective(self, trial):
        param = {
            "loss": trial.suggest_categorical(
                "loss",
                ["hinge", "log", "modified_huber", "squared_hinge", "perceptron"],
            ),
            "penalty": trial.suggest_categorical("penalty", ["l2", "l1", "elasticnet"]),
            "alpha": trial.suggest_loguniform("alpha", 1e-5, 1e5),
            "l1_ratio": trial.suggest_uniform("l1_ratio", 0, 1),
            "fit_intercept": trial.suggest_categorical("fit_intercept", [True, False]),
            "max_iter": trial.suggest_int("max_iter", 1, 1000),
            "tol": trial.suggest_loguniform("tol", 1e-5, 1e-2),
            "random_state": self.random_state,
        }
        clf = SGDClassifier(**param)
        scores = cross_val_score(
            clf, self.X, self.y, cv=self.cv, scoring=self.metric, n_jobs=-1
        )
        return scores.mean()

    def ridg_classifier_objective(self, trial):
        param = {
            "alpha": trial.suggest_loguniform("alpha", 1e-5, 1e5),
            "fit_intercept": trial.suggest_categorical("fit_intercept", [True, False]),
            "max_iter": trial.suggest_int("max_iter", 1, 1000),
            "tol": trial.suggest_loguniform("tol", 1e-5, 1e-2),
            "random_state": self.random_state,
        }
        clf = RidgeClassifier(**param)
        scores = cross_val_score(
            clf,
            self.X,
            self.y,
            cv=self.cv,
            scoring=self.metric,
            n_jobs=-1,
        )
        return scores.mean()

    def perc_classifier_objective(self, trial):
        param = {
            "penalty": trial.suggest_categorical("penalty", ["l2", "l1", "elasticnet"]),
            "alpha": trial.suggest_loguniform("alpha", 1e-5, 1e5),
            "fit_intercept": trial.suggest_categorical("fit_intercept", [True, False]),
            "max_iter": trial.suggest_int("max_iter", 1, 1000),
            "tol": trial.suggest_loguniform("tol", 1e-5, 1e-2),
            "random_state": self.random_state,
        }
        clf = Perceptron(**param)
        scores = cross_val_score(
            clf,
            self.X,
            self.y,
            cv=self.cv,
            scoring=self.metric,
            n_jobs=-1,
        )
        return scores.mean()

    def pass_classifier_objective(self, trial):
        param = {
            "C": trial.suggest_loguniform("C", 1e-5, 1e5),
            "fit_intercept": trial.suggest_categorical("fit_intercept", [True, False]),
            "max_iter": trial.suggest_int("max_iter", 1, 1000),
            "tol": trial.suggest_loguniform("tol", 1e-5, 1e-2),
            "random_state": self.random_state,
        }
        clf = PassiveAggressiveClassifier(**param)
        scores = cross_val_score(
            clf,
            self.X,
            self.y,
            cv=self.cv,
            scoring=self.metric,
            n_jobs=-1,
        )
        return scores.mean()

    def svm_classifier_objective(self, trial):
        param = {
            "C": trial.suggest_loguniform("C", 1e-5, 1e5),
            "kernel": trial.suggest_categorical(
                "kernel", ["rbf", "linear", "poly", "sigmoid"]
            ),
            "gamma": trial.suggest_loguniform("gamma", 1e-5, 1e5),
            "degree": trial.suggest_int("degree", 1, 10),
            "coef0": trial.suggest_loguniform("coef0", 1e-5, 1e5),
            "shrinking": trial.suggest_categorical("shrinking", [True, False]),
            "tol": trial.suggest_loguniform("tol", 1e-5, 1e-2),
            "random_state": self.random_state,
        }
        clf = SVC(**param)
        scores = cross_val_score(
            clf,
            self.X,
            self.y,
            cv=self.cv,
            scoring=self.metric,
            n_jobs=-1,
        )
        return scores.mean()

    def knn_classifier_objective(self, trial):
        param = {
            "n_neighbors": trial.suggest_int("n_neighbors", 1, 100),
            "weights": trial.suggest_categorical("weights", ["uniform", "distance"]),
            "algorithm": trial.suggest_categorical(
                "algorithm", ["auto", "ball_tree", "kd_tree", "brute"]
            ),
            "p": trial.suggest_int("p", 1, 10),
            "n_jobs": -1,
        }
        clf = KNeighborsClassifier(**param)
        scores = cross_val_score(
            clf,
            self.X,
            self.y,
            cv=self.cv,
            scoring=self.metric,
            n_jobs=-1,
        )
        return scores.mean()

    def dt_classifier_objective(self, trial):
        param = {
            "criterion": trial.suggest_categorical("criterion", ["gini", "entropy"]),
            "splitter": trial.suggest_categorical("splitter", ["best", "random"]),
            "max_depth": trial.suggest_int("max_depth", 1, 10),
            "min_samples_split": trial.suggest_int("min_samples_split", 1, 10),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
            "min_weight_fraction_leaf": trial.suggest_float(
                "min_weight_fraction_leaf", 0, 0.5
            ),
            "max_features": trial.suggest_categorical(
                "max_features", ["auto", "sqrt", "log2"]
            ),
            "random_state": self.random_state,
        }
        clf = DecisionTreeClassifier(**param)
        scores = cross_val_score(
            clf,
            self.X,
            self.y,
            cv=self.cv,
            scoring=self.metric,
            n_jobs=-1,
        )
        return scores.mean()

    def rfc_classifier_objective(self, trial):
        param = {
            "n_estimators": trial.suggest_int("n_estimators", 1, 100),
            "criterion": trial.suggest_categorical("criterion", ["gini", "entropy"]),
            "max_depth": trial.suggest_int("max_depth", 1, 10),
            "min_samples_split": trial.suggest_int("min_samples_split", 1, 10),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
            "min_weight_fraction_leaf": trial.suggest_float(
                "min_weight_fraction_leaf", 0, 0.5
            ),
            "max_features": trial.suggest_categorical(
                "max_features", ["auto", "sqrt", "log2"]
            ),
            "random_state": self.random_state,
            "n_jobs": -1,
        }
        clf = RandomForestClassifier(**param)
        scores = cross_val_score(
            clf,
            self.X,
            self.y,
            cv=self.cv,
            scoring=self.metric,
            n_jobs=-1,
        )
        return scores.mean()

    def gbc_classifier_objective(self, trial):
        param = {
            "n_estimators": trial.suggest_int("n_estimators", 1, 100),
            "learning_rate": trial.suggest_loguniform("learning_rate", 1e-5, 1e5),
            "max_depth": trial.suggest_int("max_depth", 1, 10),
            "min_samples_split": trial.suggest_int("min_samples_split", 1, 10),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
            "min_weight_fraction_leaf": trial.suggest_float(
                "min_weight_fraction_leaf", 0, 0.5
            ),
            "max_features": trial.suggest_categorical(
                "max_features", ["auto", "sqrt", "log2"]
            ),
        }
        clf = GradientBoostingClassifier(**param)
        scores = cross_val_score(
            clf,
            self.X,
            self.y,
            cv=self.cv,
            scoring=self.metric,
            n_jobs=-1,
        )
        return scores.mean()

    def ada_classifier_objective(self, trial):
        param = {
            "n_estimators": trial.suggest_int("n_estimators", 1, 100),
            "learning_rate": trial.suggest_loguniform("learning_rate", 1e-5, 1e5),
            "algorithm": trial.suggest_categorical("algorithm", ["SAMME", "SAMME.R"]),
            "random_state": self.random_state,
        }
        clf = AdaBoostClassifier(**param)
        scores = cross_val_score(
            clf,
            self.X,
            self.y,
            cv=self.cv,
            scoring=self.metric,
            n_jobs=-1,
        )
        return scores.mean()

    def bag_classifier_objective(self, trial):
        param = {
            "n_estimators": trial.suggest_int("n_estimators", 1, 100),
            "bootstrap": trial.suggest_categorical("bootstrap", [True, False]),
            "bootstrap_features": trial.suggest_categorical(
                "bootstrap_features", [True, False]
            ),
            "max_samples": trial.suggest_uniform("max_samples", 0.1, 1),
            "max_features": trial.suggest_uniform("max_features", 0.1, 1),
            "n_jobs": -1,
            "random_state": self.random_state,
        }
        clf = BaggingClassifier(**param)
        scores = cross_val_score(
            clf,
            self.X,
            self.y,
            cv=self.cv,
            scoring=self.metric,
            n_jobs=-1,
        )
        return scores.mean()

    def extc_classifier_objective(self, trial):
        param = {
            "n_estimators": trial.suggest_int("n_estimators", 1, 100),
            "criterion": trial.suggest_categorical("criterion", ["gini", "entropy"]),
            "max_depth": trial.suggest_int("max_depth", 1, 10),
            "min_samples_split": trial.suggest_int("min_samples_split", 1, 10),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
            "min_weight_fraction_leaf": trial.suggest_float(
                "min_weight_fraction_leaf", 0, 0.5
            ),
            "max_features": trial.suggest_categorical(
                "max_features", ["auto", "sqrt", "log2"]
            ),
            "random_state": self.random_state,
            "n_jobs": -1,
        }
        clf = ExtraTreesClassifier(**param)
        scores = cross_val_score(
            clf,
            self.X,
            self.y,
            cv=self.cv,
            scoring=self.metric,
            n_jobs=-1,
        )
        return scores.mean()

    def lgbm_classifier_objective(self, trial):
        param = {
            "n_estimators": trial.suggest_int("n_estimators", 1, 1000),
            "learning_rate": trial.suggest_loguniform("learning_rate", 1e-5, 1e5),
            "max_depth": trial.suggest_int("max_depth", 1, 10),
            "min_child_samples": trial.suggest_int("min_child_samples", 1, 10),
            "min_child_weight": trial.suggest_uniform("min_child_weight", 0, 0.5),
            "subsample": trial.suggest_uniform("subsample", 0.1, 1),
            "subsample_freq": trial.suggest_int("subsample_freq", 1, 10),
            "colsample_bytree": trial.suggest_uniform("colsample_bytree", 0.1, 1),
            "random_state": self.random_state,
            "objective": self.lgbm_objective,
            "n_jobs": -1,
        }
        clf = LGBMClassifier(**param)
        scores = cross_val_score(
            clf,
            self.X,
            self.y,
            cv=self.cv,
            scoring=self.metric,
            n_jobs=-1,
        )
        return scores.mean()

    def cat_classifier_objective(self, trial):
        param = {
            "objective": trial.suggest_categorical("objective", ["Logloss", "CrossEntropy"]),
            "iterations": trial.suggest_int("iterations", 100, 3000),
            "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.01, 0.1),
            "depth": trial.suggest_int("depth", 1, 12),
            "boosting_type": trial.suggest_categorical("boosting_type", ["Ordered", "Plain"]),
            "bootstrap_type": trial.suggest_categorical(
                "bootstrap_type", ["Bayesian", "Bernoulli", "MVS"]
            ),
            "used_ram_limit": "3gb",
            "learning_rate": trial.suggest_loguniform("learning_rate", 1e-5, 1e5),
            "random_state": self.random_state,
        }

        if param["bootstrap_type"] == "Bayesian":
            param["bagging_temperature"] = trial.suggest_float("bagging_temperature", 0, 10)
        elif param["bootstrap_type"] == "Bernoulli":
            param["subsample"] = trial.suggest_float("subsample", 0.1, 1)
        clf = CatBoostClassifier(**param)
        scores = cross_val_score(
            clf,
            self.X,
            self.y,
            cv=self.cv,
            scoring=self.metric,
            n_jobs=-1,
        )
        return scores.mean()

    def xgb_classifier_objective(self, trial):
        param = {
            "n_estimators": trial.suggest_int("n_estimators", 1, 100),
            "max_depth": trial.suggest_int("max_depth", 1, 10),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "learning_rate": trial.suggest_uniform("learning_rate", 0.01, 1),
            "gamma": trial.suggest_uniform("gamma", 0, 1),
            "colsample_bytree": trial.suggest_uniform("colsample_bytree", 0.1, 1),
            "subsample": trial.suggest_uniform("subsample", 0.1, 1),
            "reg_alpha": trial.suggest_uniform("reg_alpha", 0, 1),
            "reg_lambda": trial.suggest_uniform("reg_lambda", 0, 1),
            "random_state": self.random_state,
            "n_jobs": -1,
            
        }
        clf = XGBClassifier(**param)
        scores = cross_val_score(
            clf,
            self.X,
            self.y,
            cv=self.cv,
            scoring=self.metric,
            n_jobs=-1,
        )
        return scores.mean()

    def mlp_classifier_objective(self, trial):
        param = {
            "hidden_layer_sizes": trial.suggest_int("hidden_layer_sizes", 1, 10),
            "activation": trial.suggest_categorical(
                "activation", ["identity", "logistic", "tanh", "relu"]
            ),
            "solver": trial.suggest_categorical("solver", ["lbfgs", "sgd", "adam"]),
            "alpha": trial.suggest_uniform("alpha", 0, 1),
            "learning_rate": trial.suggest_categorical(
                "learning_rate", ["constant", "invscaling", "adaptive"]
            ),
            "learning_rate_init": trial.suggest_uniform("learning_rate_init", 0, 1),
            "max_iter": trial.suggest_int("max_iter", 1, 2000),
            "random_state": self.random_state,
        }
        clf = MLPClassifier(**param)
        scores = cross_val_score(
            clf,
            self.X,
            self.y,
            cv=self.cv,
            scoring=self.metric,
            n_jobs=-1,
        )
        return scores.mean()
