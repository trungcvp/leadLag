import numpy as np
import pandas as pd
import itertools
import dcor
import statsmodels.api as sm
import multiprocessing as mp
import matplotlib.pyplot as plt
import iisignature
from tqdm import tqdm
from sklearn.feature_selection import mutual_info_classif
from multiprocessing import Pool, cpu_count
from tqdm.contrib.concurrent import process_map

# Thay đổi cách tính cross_correlation tích hợp 2 yếu tố thay vì chỉ 1 là returns 
# cor_new = alpha*cor_returns + (1-alpha)*cor_volume
def cross_correlation(data_1, data_2, lag, correlation_method='pearson', volume_1=None, volume_2=None, alpha=0.5):
    data_1_lag = data_1.shift(lag)
    data_2_lag = data_2.copy()

    if lag >= 0:
        data_1_lag = data_1_lag[lag:]
        data_2_lag = data_2_lag[lag:]
        if volume_1 is not None:
            volume_1 = volume_1[lag:]
            volume_2 = volume_2[lag:]
    else:
        data_1_lag = data_1_lag[:lag]
        data_2_lag = data_2_lag[:lag]
        if volume_1 is not None:
            volume_1 = volume_1[:lag]
            volume_2 = volume_2[:lag]

    df = pd.DataFrame({
        'ret1': data_1_lag,
        'ret2': data_2_lag
    })
    if volume_1 is not None and volume_2 is not None:
        df['vol1'] = volume_1
        df['vol2'] = volume_2

    df = df.dropna()
    if len(df) == 0:
        return 0

    if correlation_method == 'distance':
        return dcor.distance_correlation(df['ret1'], df['ret2'])

    elif correlation_method == 'distance_combined':
        dcor_ret = dcor.distance_correlation(df['ret1'], df['ret2'])
        dcor_vol = dcor.distance_correlation(df['vol1'], df['vol2'])
        return alpha * dcor_ret + (1 - alpha) * dcor_vol

    elif correlation_method in ['pearson', 'kendall', 'spearman']:
        return df[['ret1', 'ret2']].corr(method=correlation_method).iloc[0, 1]

    else:
        raise NotImplementedError(f'Correlation method {correlation_method} not supported.')

def compute_lead_lag(data_subset, method, **kwargs):
    volume_df = kwargs.get('volume_df', None)
    alpha = kwargs.get('alpha', 0.5)

    if method == 'ccf_auc':
        lags = np.arange(1, kwargs['max_lag'] + 1)
        lags = np.r_[-lags, lags]
        cross_correlation_measure = dict()
        for lag in lags:
            cross_correlation_measure[lag] = cross_correlation(
                data_subset.iloc[:, 0],
                data_subset.iloc[:, 1],
                lag=lag,
                correlation_method=kwargs['correlation_method'],
                volume_1=volume_df[data_subset.columns[0]] if volume_df is not None else None,
                volume_2=volume_df[data_subset.columns[1]] if volume_df is not None else None,
                alpha=alpha
            )
        cross_correlation_measure = pd.Series(cross_correlation_measure)
        A = cross_correlation_measure[cross_correlation_measure.index > 0].abs().sum()
        B = cross_correlation_measure[cross_correlation_measure.index < 0].abs().sum()
        lead_lag_measure = np.array([A, B]).max() / (A + B) * np.sign(A - B)

    elif method == 'signature':
        path = data_subset.cumsum()
        path /= path.std()
        signature = iisignature.sig(path, 2, 1)
        lead_lag_measure = signature[1][1] - signature[1][2]

    else:
        raise NotImplementedError

    return lead_lag_measure

def lead_lag_given_pair(args):
    pair, data, method, kwargs = args
    data_subset = data.loc[:, pair]
    return compute_lead_lag(data_subset, method=method, **kwargs)

def get_lead_lag_matrix(data, method, **kwargs):
    pair_list = list(itertools.combinations(data.columns, 2))
    args = list(itertools.product(pair_list, [data], [method], [kwargs]))

    lead_lag_measure = process_map(
        lead_lag_given_pair,
        args,
        max_workers=4,
        chunksize=1,
        desc="⏳ Calculating lead–lag matrix"
    )

    lead_lag_measure = {pair: lead_lag_measure[num] for num, pair in enumerate(pair_list)}
    lead_lag_measure = pd.Series(lead_lag_measure).unstack()
    lead_lag_measure = lead_lag_measure.reindex(index=data.columns, columns=data.columns)
    lead_lag_measure = lead_lag_measure.fillna(0)
    lead_lag_measure = lead_lag_measure - lead_lag_measure.T

    return lead_lag_measure