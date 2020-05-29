''' A collection of auxiliary functions for the Entity Embedding Encoder
'''

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler 

def replace_in_df(df, mapping):
    """ Replaces categories by numbers according to the mapping
        If a category is not in mapping, it gets a random code
        mapping: dictionary from categories to codes
    """
    # Ensure df has the right type
    if not(isinstance(df,(pd.DataFrame))):
        try:
            df = pd.DataFrame(df)
        except:
            raise Exception('Cannot convert to pandas.DataFrame')

    cat_cols = categorical_cols(df)

    # Updates the mapping with random codes for categories not
    # previously in the mapping
    for x in cat_cols:
            cats = np.unique(df[x])
            for x in cats:
                if not(x in mapping):
                    mapping[x] = np.random.uniform(0,1)

    return df.replace(mapping)

def is_categorical(array):
    """ Tests if the column is categorical
    """
    return array.dtype.name == 'category' or array.dtype.name == 'object'

def categorical_cols(df): 
    """ Return the column numbers of the categorical variables in df
    """
    cols = []
    # Rename columns as numbers
    df.columns = range(len(df.columns))
    
    for x in df.columns: 
        if is_categorical(df[x]):
            cols.append(x)
    return cols

def categorical_instances(df):
    """ Returns an array with all the categorical instances in df
    """
    instances = []
    cols = categorical_cols(df)
    for x in cols:
        instances = instances + list(np.unique(df[x]))
        
    return instances

def num_categorical_instances(df):
    """ Returns the total number of categorical instances in df
    """
    return len(categorical_instances(df))

def scale_df(df):
    """ Scales numerical variables to [0,1]
    """
    scaler = MinMaxScaler()
    for x in df.columns:
        if not(is_categorical(df[x])):
            df[x] = scaler.fit_transform(df[x].values.reshape(-1,1))
    return df
