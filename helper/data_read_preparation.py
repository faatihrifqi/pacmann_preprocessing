import pandas as pd

def read_data(PATH):
    '''
    Read data from dataset from path
   
    Parameters
    ----------
    PATH : str
        path source of training data, csv.
    
    Returns
    -------
    data : pd.DataFrame
        Data for modeling
    '''
    data = pd.read_csv(PATH, encoding='latin-1')
    
    return data

def read_and_check_data(path):
    """Read data."""
    print("start import data")
    df = read_data(path)
    print("done import data")
    
    return df