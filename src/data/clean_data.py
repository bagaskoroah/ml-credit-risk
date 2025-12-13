import pandas as pd
from src.config import config

def remove_duplicates(data: pd.DataFrame) -> pd.DataFrame:
    """
    Remove duplicate rows from a DataFrame and print diagnostic information.

    Parameters
    ----------
    data : pd.DataFrame
        The input dataset that may contain duplicate rows.

    Returns
    -------
    pd.DataFrame
        The DataFrame after removing duplicate rows.
    """
    # check total duplicates
    print('Data shape before dropping duplicates:', data.shape)
    print('Total duplicates in dataset:', data.duplicated().sum())

    # remove duplicate
    data.drop_duplicates(keep='last', inplace=True)

    # check shape after dropping
    print('\nRemoving duplicates....')
    print(f'Duplicates have dropped. Data shape after dropping duplicates: {data.shape}')
    
    return data

def cast_type_target(data: pd.DataFrame) -> pd.DataFrame:
    """
    Cast the target column into integer type if it is not already numeric.

    Parameters
    ----------
    data : pd.DataFrame
        Input dataset that contains the target column defined in config.TARGET.

    Returns
    -------
    pd.DataFrame
        The DataFrame with the target column casted into integer type (if needed).
    """
    if data[config.TARGET].dtype not in [int, float]:
        print(f'{config.TARGET} data type before casting: {data[config.TARGET].dtype}')
        print('Casting target data type....')
        data[config.TARGET] = data[config.TARGET].astype(int)
        print(f'\n{config.TARGET} column data type has been casted into desired type: {data[config.TARGET].dtype}.')

    return data