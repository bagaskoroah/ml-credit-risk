from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from src.config import config
from sklearn.compose import ColumnTransformer

def build_pipeline():
    # build numerical features pipeline
    num_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    # build categorical features pipeline
    cat_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(sparse_output=False, handle_unknown='ignore'))
    ])

    return num_pipeline, cat_pipeline
    
def build_preprocessing(num_pipe, cat_pipe):
    # combine num and cat pipeline
    preprocessor = ColumnTransformer(transformers=[
        ('num', num_pipe, config.NUM_COLS),
        ('cat', cat_pipe, config.CAT_COLS)
    ])

    return preprocessor