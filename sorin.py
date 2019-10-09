import pandas as pd
import numpy as np
import random



def import_worldbank_indicator(data_path=None, meta_path=None):
    
    # read the indicator
    data_df = pd.read_csv(data_path, skiprows=4)
    
    # retain only select columns
    columns_to_keep = ["Country Code", "Country Name", "2010", "2011", "2012", "2013", "2014", "2015"]
    columns_to_drop = set.difference(set(data_df.columns), set(columns_to_keep))
    data_df.drop(columns=list(columns_to_drop), inplace=True)
    data_df.dropna(thresh=5, inplace=True)
    
    # read the country meta
    meta_df = pd.read_csv(meta_path)
    
    # merge the dataframes
    df = pd.merge(data_df, meta_df[["Country Code", "Region", "IncomeGroup"]], how="left")

    return df



def calculate_correlation(data=None):

    # calculate the Pearson correlation coefficient
    # input is of type [(x1,y1), (x2,y2), (x3,y3), ...]
    # which can be converted to a np.ndarray of dtype float64

    # convert to numpy
    try:
        data = list(data)
        data = np.array(data, dtype=np.float64)
    except:
        raise
    
    # extract x and y 
    xs = data[:, 0]
    ys = data[:, 1]
    
    # calculate Pearson correlation coefficient
    corr = np.corrcoef(xs, ys)
    corr = corr[0,1]
    
    return corr



def resample_correlation_bootstrap(data=None, iterations=1):

    # resample method for the correlation coefficient
    # BOOTSTRAP algorithm for calculating the confidence interval
    #    N.B. it maintains correlations in the data
    # input is of type [(x1,y1), (x2,y2), (x3,y3), ...]
    # which can be converted to a np.ndarray of dtype float64
    # returns the corresponding correlation coefficients

    # convert to numpy
    try:
        data = list(data)
        data = np.array(data, dtype=np.float64)
    except:
        raise
    
    # resampled correlations
    corrs = []
    
    # iterate
    for index in range(iterations):
        data_resampled = random.choices(list(data), k=len(list(data)))
        corrs.append(calculate_correlation(data_resampled))
    
    # calculate 95 confidence interval
    low_end  = np.percentile(corrs, 2.5)
    high_end = np.percentile(corrs, 97.5)
    
    return low_end, high_end



def resample_correlation_randomize(data=None, iterations=1):

    # resample method for the correlation coefficient
    # RANDOMIZE algorithm for calculating the confidence interval
    #    N.B. it breaks correlations in the data
    # input is of type [(x1,y1), (x2,y2), (x3,y3), ...]
    # which can be converted to a np.ndarray of dtype float64
    # returns the corresponding correlation coefficients

    # convert to numpy
    try:
        data = list(data)
        data = np.array(data, dtype=np.float64)
    except:
        raise
    
    # resampled correlations
    corrs = []
    
    # extract x and y 
    xs = data[:, 0]
    ys = data[:, 1]
        
    # iterate
    for index in range(iterations):
        data_resampled = zip(xs, np.random.permutation(ys))
        corrs.append(calculate_correlation(data_resampled))
    
    # calculate p-value
    corrs = np.array(corrs)
    pval = len(corrs[corrs > calculate_correlation(data)]) / len(corrs)
    
    return pval
