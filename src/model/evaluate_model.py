from sklearn.metrics import recall_score, precision_score, f1_score, roc_auc_score, ConfusionMatrixDisplay, confusion_matrix
import numpy as np
import pandas as pd
from typing import Any, Dict, Tuple
import matplotlib.pyplot as plt

def evaluate_baseline(y_train: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray) -> Tuple[float, float, float, float]:
    """
    Calculate baseline evaluation metrics (recall, precision, f1-score).

    Parameters
    ----------
        y_train (np.ndarray): True labels from the training set.
        y_pred (np.ndarray): Predicted labels from the model.
        y_proba (np.ndarray): Predicted probabilities for positive class.

    Returns
    -------
        Tuple[float, float, float, float]: Recall, precision, F1-score, and ROC-AUC score.
    """

    # calculate baseline recall
    recall_base = recall_score(y_train, y_pred)
    precision_base = precision_score(y_train, y_pred)
    f1_base = f1_score(y_train, y_pred)
    roc_auc_base = roc_auc_score(y_train, y_proba)

    return recall_base, precision_base, f1_base, roc_auc_base

def evaluate_cv_train(
    estimator: Any,
    X_train: np.ndarray,
    y_train: np.ndarray
) -> Tuple[Any, Any, float, float, float, float, float, float, float, float]:
    """
    Evaluate both cross-validation performance and training performance.

    Parameters
    ----------
        estimator (Any): A fitted sklearn CV model (e.g., GridSearchCV, RandomizedSearchCV).
        x_train (np.ndarray): Feature labels from the training set.
        y_train (np.ndarray): True labels from the training set.

    Returns
    -------
        Tuple[float, float, float, float, float, float, float, float]:
            - CV best params
            - CV best model
            - CV recall
            - CV precision
            - CV F1 score
            - CV AUC
            - Train recall
            - Train precision
            - Train F1 score
            - Train AUC
    """

    # pick best param and best model
    best_param = estimator.best_params_
    best_model = estimator.best_estimator_

    # generate cv scores
    recall_cv = estimator.cv_results_['mean_test_recall'].max()
    precision_cv = estimator.cv_results_['mean_test_precision'].max()
    f1_cv = estimator.cv_results_['mean_test_f1'].max()
    roc_auc_cv = estimator.cv_results_['mean_test_roc_auc'].max()

    # fitting models to train
    best_model.fit(X_train, y_train)

    # predict models to train
    y_pred = best_model.predict(X_train)
    y_proba = best_model.predict_proba(X_train)[:, 1]

    # generate train scores
    recall_train = recall_score(y_train, y_pred)
    precision_train = precision_score(y_train, y_pred)
    f1_train = f1_score(y_train, y_pred)
    roc_auc_train = roc_auc_score(y_train, y_proba)

    return best_param, best_model, recall_cv, precision_cv, f1_cv, roc_auc_cv, \
    recall_train, precision_train, f1_train, roc_auc_train

def evaluate_test(y_test: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray) -> Tuple[float, float, float, float]:
    """
    Evaluate model performance on the test set.

    Parameters
    ----------
        y_test (np.ndarray): True labels from the test set.
        y_pred (np.ndarray): Predicted labels from the model.
        y_proba (np.ndarray): Predicted probabilities for positive class.

    Returns
    -------
        Tuple[float, float, float, float]: Test recall, precision, F1 score, ROC-AUC score.
    """
    
    # generate metric scores
    recall_test = recall_score(y_test, y_pred)
    precision_test = precision_score(y_test, y_pred)
    f1_test = f1_score(y_test, y_pred)
    roc_auc_test = roc_auc_score(y_test, y_proba)

    return recall_test, precision_test, f1_test, roc_auc_test

def confusion(
    y_test: np.ndarray,
    y_pred: np.ndarray
) -> Tuple[ConfusionMatrixDisplay, int, int, int, int]:
    """
    Generate and display a confusion matrix plot, and return the underlying values.

    Parameters
    ----------
        y_test (np.ndarray): True labels from the test set.
        y_pred (np.ndarray): Predicted labels from the model.

    Returns
    -------
        Tuple[
            ConfusionMatrixDisplay,
            int, int, int, int
        ]:
            - ConfusionMatrixDisplay object
            - True Negative (TN)
            - False Positive (FP)
            - False Negative (FN)
            - True Positive (TP)
    """

    # define cm object
    cm = confusion_matrix(y_test, y_pred)
    display = ConfusionMatrixDisplay(confusion_matrix=cm)

    # plot the cm display
    display.plot()

    # unpack the each component value
    tn, fp, fn, tp = cm.ravel()

    # show the plot
    plt.show()

    return display, tn, fp, fn, tp