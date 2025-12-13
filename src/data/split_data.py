from sklearn.model_selection import train_test_split
from src.config import config
import pandas as pd
from typing import Tuple

def split_train_test(data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    '''
    Split the data into training and testing sets.
    
    Params:
    - data (pd.DataFrame): The input data to be split.

    Returns:
    - Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]: X_train, X_test, y_train, y_test
    '''
    # split into x and y
    X = data.drop(columns=[config.TARGET])
    y = data[config.TARGET]

    # split train and test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE
    )

    # print data shapes
    print('X_train shape:', X_train.shape)
    print('X_test shape:', X_test.shape)
    print('y_train shape:', y_train.shape)
    print('y_test shape:', y_test.shape)

    return X_train, X_test, y_train, y_test