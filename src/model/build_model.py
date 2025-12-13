from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from src.config import config
from sklearn.metrics import make_scorer, recall_score, precision_score, f1_score, roc_auc_score
from sklearn.dummy import DummyClassifier
import pandas as pd
import numpy as np
import time
from src.preprocessing import preprocessing
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from typing import Any, Tuple, Dict
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

def build_baseline(X_train: np.ndarray, 
    y_train: np.ndarray) -> np.ndarray:
    """
    Build and train a baseline DummyClassifier using a stratified strategy.

    Parameters
    ----------
    X_train : np.ndarray
        Training features.
    y_train : np.ndarray
        Training labels.
s
    Returns
    -------
    np.ndarray
        Predictions from the baseline model on training data.
    """
    
    # create baseline object 
    base_model = DummyClassifier(strategy='stratified')

    # fit object to train data
    base_model.fit(X_train, y_train)

    # predict train data
    y_pred = base_model.predict(X_train)

    return y_pred

def build_cv_train(
        estimator: Any, 
        preprocessor: Any, 
        params: Dict[str, Any], 
        X_train: np.ndarray, 
        y_train: np.ndarray, 
        is_smote: bool = False) -> Any:
    """
    Perform cross-validated model training with preprocessing and SMOTE pipeline.
    Evaluates the best model on training data and returns predictions + best model.

    Parameters
    ----------
    estimator : Any
        Machine learning estimator to train.
    preprocessor : Any
        Preprocessing transformer.
    params : dict
        Hyperparameter search space for RandomizedSearchCV.
    X_train : np.ndarray
        Training input features.
    y_train : np.ndarray
        Training labels.
    is_smote : bool
        Option whether using smote or non-smote in fitting the models

    Returns
    -------
    tuple
        cv_model : Any  
            Group of CV models to be selected as a best model in the next of evaluation process.
    """

    # define start time process
    start_time = time.time()

    # define cv score metrics
    scoring = {
        'recall': make_scorer(recall_score),
        'precision': make_scorer(precision_score),
        'f1': make_scorer(f1_score),
        'roc_auc': make_scorer(roc_auc_score),
    }

    # build model pipeline
    steps = [('preprocessing', preprocessor)]
    if is_smote:
        steps.append(('SMOTE', SMOTE(random_state=config.RANDOM_STATE)))
    steps.append(('model', estimator))

    model = ImbPipeline(steps=steps)

    # if the model is random forest, we will train using grid search
    if isinstance(estimator, (RandomForestClassifier, CatBoostClassifier, XGBClassifier)):
        cv_model = RandomizedSearchCV(
            estimator=model,
            param_distributions=params,
            n_iter=config.N_ITER,
            scoring=scoring,
            refit=config.REFIT,
            n_jobs=config.N_JOBS,
        )

    else:
        cv_model = GridSearchCV(
            estimator=model,
            param_grid=params,
            scoring=scoring,
            refit=config.REFIT,
            n_jobs=config.N_JOBS,
        )

    # fit cv and train
    cv_model.fit(X_train, y_train)
    
    # define end time process
    end_time = time.time() - start_time
    end_time = round(end_time/60, 2)
    print(f'Model {estimator.__class__.__name__} has been created succesfully, time elapsed: {end_time} minutes.')

    return cv_model

def build_test(
        estimator: Any, 
        X_test: np.ndarray) -> np.ndarray:
    
    """
    Generate predictions from the final trained estimator on the test data.

    Parameters
    ----------
    estimator : Any
        Trained model used to generate predictions.
    X_test : pd.DataFrame
        Test features.

    Returns
    -------
    np.ndarray
        Predicted labels on the test set.
    """

    # define start time
    start_time = time.time()

    # predict models on test data
    y_pred = estimator.predict(X_test)
    
    # calculate processing time
    end_time = time.time() - start_time
    end_time = round(end_time/60, 2)

    return y_pred