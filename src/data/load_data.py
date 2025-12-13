import pandas as pd

def read_data(file_path: str) -> pd.DataFrame:
    '''
    Load data from a CSV file into a pandas DataFrame.

    Params:
    - file_path (str): The path to the CSV file.

    Returns:
    - pd.Dataframe: Loaded data as a Pandas DataFrame.
    '''

    # checking file extensions
    if file_path.endswith('.csv'):
        try:
            data = pd.read_csv(file_path)
            print(f'Data loaded succesfully from {file_path}!')
            print('Data shape:', data.shape)
            return data
        except FileNotFoundError:
            print(f'Error: The .csv file at {file_path} was not found.')
            return pd.DataFrame()
        except pd.errors.EmptyDataError:
            print(f'Error: The .csv file at {file_path} is empty.')
            return pd.DataFrame()
        except pd.errors.ParserError:
            print(f"Error: The .csv file at {file_path} couldn't be parsed.")
            return pd.DataFrame()
    else:
        print('Error: Unsupported file format. Please provide your file in .csv format.')