# data path
RAW_DATA = 'data/raw/credit_risk_dataset.csv'
CLEAN_DATA = 'data/processed/credit_risk_cleaned.csv'

# numerical and categorical features
NUM_COLS = ['person_age', 'person_income', 'person_emp_length', 'loan_amnt', 'loan_int_rate', 'loan_percent_income', 'cb_person_cred_hist_length']
CAT_COLS = ['loan_intent', 'loan_grade', 'cb_person_default_on_file', 'person_home_ownership']

# splitting data
TARGET = 'loan_status'

# train test split
RANDOM_STATE = 123
TEST_SIZE = 0.2

# cv arguments
CV = 5
N_ITER = 40
REFIT = 'recall'
VERBOSE = 0
N_JOBS = -1

# model parameters
XGBOOST_PARAMS = {
    'model__n_estimators': [100, 200, 500, 750, 1000],
    'model__learning_rate': [0.01, 0.05, 0.1, 0.2],
    'model__max_depth': [3, 5, 7, 10],
    'model__reg_lambda': [0, 0.1, 0.25, 0.5, 1, 2],
    'model__reg_alpha': [0, 0.1, 0.5]
    }

CATBOOST_PARAMS = {
    'model__iterations': [200, 500, 800],
    'model__learning_rate': [0.01, 0.05, 0.1, 0.2],
    'model__depth': [4, 6, 8, 10]
    }

KNN_PARAMS = {
    'model__n_neighbors': list(range(3, 26, 2))
}

RF_PARAMS = {
    'model__n_estimators': [100, 200, 500],
    'model__max_depth': [None, 5, 10, 20, 50],
    'model__min_samples_split': [2, 5, 10],
    'model__min_samples_leaf': [1, 2, 4],
    "model__max_features": ['sqrt', 'log2']
}

# model threshold config
THRESHOLD_PROBA = 0.2

# model output directory path
MODEL_PATH = 'artifacts/model/best_model.pkl'
MODEL_THRESHOLD_PATH = 'artifacts/model/threshold_model.pkl'