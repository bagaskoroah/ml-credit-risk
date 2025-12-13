import shap
import pandas as pd
from typing import Any, Literal

def explain_model(
        model: Any,
        X_test: pd.DataFrame,
        mode: Literal['summary', 'bar', 'both'] = 'both') -> shap.Explanation:
    """
    Generate SHAP-based interpretability visualization for a trained model.
    Automatically handles models inside a pipeline and distinguishes 
    between tree-based and non-tree-based algorithms.

    Parameters
    ----------
    model : Any
        The fitted model or pipeline containing a preprocessor 
        and estimator under 'model' step.
    X_test : pd.DataFrame
        Input feature data used to compute SHAP values.
    mode : {'summary', 'bar', 'both'}, default='both'
        Determines which visualization to display:
            - 'summary' : SHAP summary plot
            - 'bar'     : SHAP feature importance bar plot
            - 'both'    : Shows both summary and bar plots

    Returns
    -------
    shap_values : shap.Explanation
        SHAP values generated for the given model and dataset.
    """

    # extract only model from the model pipeline
    if hasattr(model, 'named_steps'):
        # get the preprocessor only
        preprocessor = model.named_steps['preprocessing']
        # get the estimator only
        estimator = model.named_steps['model']

        # preprocess X_test
        X_test_transformed = preprocessor.transform(X_test)

        # get feature names from column transformer
        feature_names = preprocessor.get_feature_names_out()

    # if the model is not in pipeline
    else:
        estimator = model
        X_test_transformed = X_test
        feature_names = X_test.columns.tolist()
    
    # detect whether the model is tree or non-tree
    is_tree = {
        'XGB' in estimator.__class__.__name__ or
        'CatBoost' in estimator.__class__.__name__ or
        'RandomForest' in estimator.__class__.__name__
    }

    # select SHAP explainer for tree-based models or non-tree
    if is_tree:
        explainer = shap.TreeExplainer(estimator)
    else:
        explainer = shap.Explainer(estimator, X_test_transformed)
    
    # compute SHAP values
    shap_values = explainer(X_test_transformed)

    # visualization
    if mode in ['summary', 'both']:
        shap.summary_plot(shap_values, X_test_transformed, feature_names=feature_names)
    
    if mode in ['bar', 'both']:
        shap.summary_plot(shap_values, feature_names=feature_names, plot_type='bar')

    return shap_values