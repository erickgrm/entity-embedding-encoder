''' A collection of auxiliary functions for categorical encoders
'''

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler

def replace_in_df(df, mapping): 
    """ Replaces categories by numbers according to the mapping
        If a category is not in mapping, it gets a random code
        mapping: dictionary from categories to codes
    """
    # Updates the mapping with random codes for categories not 
    # previously in the mapping
    for x in df.columns:
        if is_categorical(df[x]):
            cats = np.unique(df[x])
            for x in cats:
                if not(x in mapping):
                    mapping[x] = np.random.uniform(0,1)

    return df.replace(mapping)


def codes_to_dictionary(L):
    """ L: list of strings of the form str + ": " + float
        RETURNS dictionary with elements str : float
    """
    dict = {}
    for x in L:
        k, v = split_str(x)
        dict[k] = v
    return dict

def split_str(s):
    """ Splits str + ": " + float into str, float
    """
    i=0
    while s[i] != ':':
        i+=1
    return s[:i], s[i+2:]

def is_categorical(array):
    """ Tests if the column is categorical
    """
    return array.dtype.name == 'category' or array.dtype.name == 'object'

def scale_df(df):
    """ Scales numerical variables to [0,1]
    """
    scaler = MinMaxScaler()
    for x in df.columns:
        if not(is_categorical(df[x])):
            df[x] = scaler.fit_transform(df[x].values.reshape(-1,1))
    return df

def standardize_df(df):
    """ Removes mean and scales numerical variables
    """
    scaler = StandardScaler()
    for x in df.columns:
        if not(is_categorical(df[x])):
            df[x] = scaler.fit_transform(df[x].values.reshape(-1,1))
    return df

def random_encoding_of_categories(df):
    """ Encodes the categorical variables with random numbers in [0,1]
    """
    for x in df.columns:
        if is_categorical(df[x]):
            np.random.seed()
            k = len(np.unique(df[x]))
            codes = np.random.uniform(0,1,k)
            dictionary = dict(zip(np.unique(df[x]),codes))
            df[x] = df[x].replace(dictionary)
    return df

def seeded_random_encoding_of_variable(var,seed):
    """ Encodes the target variable with random numbers in [0,1]
        var can be a pandas DataFrame or a numpy array
        returns the encoded target as a pandas DataFrame
    """
    k = len(np.unique(var))
    np.random.seed(seed)
    codes = np.random.uniform(0,1,k)
    dictionary = dict(zip(np.unique(var),codes))

    return pd.DataFrame(var).replace(dictionary)

def categorical_instances(df):
    """ Returns an array with all the categorical instances in df
    """
    instances = []
    for x in df.columns:
        if is_categorical(df[x]):
            instances = instances + list(np.unique(df[x]))
    return instances

def num_categorical_instances(df):
    """ Returns the total number of categorical instances in df
    """
    k = 0
    for x in df.columns:
        if is_categorical(df[x]):
            k += len(np.unique(df[x]))
    return k

