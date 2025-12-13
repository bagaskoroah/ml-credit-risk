import numpy as np
from sklearn.metrics import recall_score, precision_score
import matplotlib.pyplot as plt
from typing import Any

class ThresholdClassifier:
    """
    Wrapper classifier to apply a custom probability threshold
    on top of a fitted probabilistic estimator.

    This class does NOT retrain the model. It only modifies the
    decision rule during prediction time.
    """

    def __init__(self, estimator: Any, threshold_point: float) -> None:
        """
        Initialize the threshold-based classifier.

        Parameters
        ----------
        estimator : Any
            A fitted estimator that implements `predict_proba`.
        threshold_point : float
            Probability threshold used to convert probabilities
            into binary predictions.
        """

        self.estimator = estimator
        self.threshold = threshold_point

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        return (self.estimator.predict_proba(X_test)[:, 1] >= self.threshold).astype(int)
    
def visualize(estimator: Any, X_test: np.ndarray, y_test: np.ndarray) -> None:
    """
    Predict binary class labels using a custom probability threshold.

    Parameters
    ----------
    X_test : np.ndarray
        Input feature matrix.

    Returns
    -------
    np.ndarray
        Binary predictions (0 or 1) after applying the threshold.
    """
    thresh_list = np.arange(0.1, 0.9, 0.01)

    recall_list = []
    precision_list = []

    for t in thresh_list:
        y_pred = (estimator.predict_proba(X_test)[:, 1] >= t).astype(int)
        recall_list.append(recall_score(y_test, y_pred))
        precision_list.append(precision_score(y_test, y_pred))

    plt.plot(thresh_list, recall_list, label='Recall')
    plt.plot(thresh_list, precision_list, label='Precision')
    plt.xlabel('Threshold')
    plt.ylabel('Score')
    plt.title('Recall-Precision Scores vs Threshold')
    plt.legend()
    return plt.show()