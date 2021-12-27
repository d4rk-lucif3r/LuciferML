import traceback

from luciferml.supervised.utils.predictors import classification_predictor
from luciferml.supervised.utils.predictors import regression_predictor
from luciferml.supervised.utils.tuner.optuna.optuna_base import Tuner


def luciferml_tuner(
    predictor,
    objective,
    n_trials,
    sampler,
    direction,
    X_train,
    y_train,
    cv_folds,
    random_state,
    metric,
    all_mode=False,
    verbose=False,
    isReg=False,
):
    """
    Takes classifier, tune-parameters, Training Data and no. of folds as input and Performs GridSearch Crossvalidation.
    """
    tuner = Tuner(n_trials=n_trials, sampler=sampler, direction=direction)
    try:
        best_params, best_score = tuner.tune(objective)
        if isReg:
            model, _ = regression_predictor(
                predictor,
                best_params,
                X_train,
                y_train,
                cv_folds,
                random_state,
                metric,
                all_mode=all_mode,
                verbose=verbose,
            )
        if not isReg:
            model, _ = classification_predictor(
                predictor,
                best_params,
                X_train,
                y_train,
                cv_folds,
                random_state,
                metric,
                all_mode=all_mode,
                verbose=verbose,
            )
        if not all_mode:
            print("Best Params: ", best_params)
            print("Best Score: ", best_score * 100, "\n")
        return best_params, best_score, model
    except Exception as error:
        print("HyperParam Tuning Failed with Error: ", error, "\n")
        traceback.print_exc()
