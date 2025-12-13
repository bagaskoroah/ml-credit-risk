import joblib
from typing import Any

def save_object(obj: Any, path:str) -> None:
    '''Save trained any object into disk.'''
    joblib.dump(obj, path)
    print('Saving object. . . .')
    print('Your object has been saved succesfully and stored into:', path)

def load_object(path:str) -> Any:
    '''Load any object from disk and return it.'''
    return joblib.load(path)