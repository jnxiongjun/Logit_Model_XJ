# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from math import *
import matplotlib.pyplot as plt
#是数据清洗，删除空值、单一值过多的变量
def variable_cleaning(df, keep_list, null_cutoff = 0.7, value_cutoff = 0.8, level_cutoff = 40):
    '''
    Function:
    =========
    Delete varaibles:
    1. with too many missings;
    2. with dominant values;
    3. with too many levels.

    Parameters:
    ===========
    df          : DataFrame, original data set
    keep_list   :      list, necessary variables to keep (like target and primary keys)
    null_cutoff :     float, cutoff for variables with missing values (default 0.7)
    value_cutoff:     float, cutoff for variables with dominant values(default 0.8)
    level_cutoff:   integer, cutoff for variables with multiple levels(default 40)

    Output:
    =======
    DataFrame: Cleaned data

    '''
    n_sample = df.shape[0]

    # Step 1: drop variables with too many missing values
    var_list_s1 = [x for x in df.columns if x not in keep_list]
    var_cal_s1 = df[var_list_s1]

    var_missing = var_cal_s1.apply(lambda x: x.isnull().sum(), axis = 0)/n_sample
    missing_list = list(var_missing[var_missing >= null_cutoff].index)

    df = df.drop(missing_list, axis = 1)

    # Step 2: drop variables with dominant values
    var_list_s2 = [x for x in df.columns if x not in keep_list]
    var_cal_s2 = df[var_list_s2]

    var_dom = var_cal_s2.apply(lambda x: (x.value_counts()/n_sample).max(), axis = 0)
    dom_list = list(var_dom[var_dom >= value_cutoff].index)

    df = df.drop(dom_list, axis = 1)

    # Step 3: deal with vairables with too many levels
    object_list = list(df.dtypes[df.dtypes == 'object'].index)
    var_list_s3 = [x for x in object_list if x not in keep_list]

    var_cal_s3 = df[var_list_s3]

    var_level = var_cal_s3.apply(lambda x: x.value_counts().count(), axis = 0)
    level_list = list(var_level[var_level >= level_cutoff].index)

    df = df.drop(level_list, axis = 1)

    return df

def cat_factorize(df, keep_list):
    '''
    Function:
    =========
    Transform categorical variables into integer
    '''

    cat_list = df.dtypes[(df.dtypes == 'object') | (df.dtypes == 'bool')]
    cat_var_list = list(cat_list.index)

    for c in [x for x in cat_var_list if x not in keep_list]:
        df[c] = df[c].factorize()[0]

    return df

def binning_plot(df):
    '''
    plot Y_rate and bin_percent curves
    '''
    plt.figure(figsize=(9, 4))
    plt.subplot(121)
    ax1a = df['Y_rate'].plot.line(style='r')

    plt.subplot(122)
    ax1b = df['PctTotal'].plot.bar(secondary_y=True,alpha = 0.3, grid = False)

    plt.tight_layout()
    plt.show()

#相关性删除
def corr_delete(df, var, var_iv, corr_value=0.7):
    '''
    delete correlated variables for a given variable
    '''
    select_list = []
    for c in var_iv.index.values:
        if abs(df[var].corr(df[c]))< corr_value:
            select_list.append(c)
    return var_iv[select_list]



def variable_binning(df, target, var, bins, return_type, side = "left"):
    '''
    binning for numerical variables
    '''
    # Preparation: some statistics and calculation
    df = df[[target, var]]
    n_sample = df.shape[0]
    y_count = df[target].sum()

    if type(bins) == type(1):
        df[var].fillna(-9999, inplace = True)
        percentile = 1.0/ bins
        value_percent = (df[var].value_counts().sort_index().cumsum()/n_sample)
        map_dict = dict((value_percent/percentile).apply(lambda x: ceil(x)))
        df['qt_binning'] = df[var].apply(lambda x: map_dict[x])

    elif type(bins) == type([]):
        df[var].fillna(-9999, inplace = True)
        df['qt_binning'] = var_bins_series(df[var], bins, side = side)

    else: raise NameError("'bins' must be integer or list")

    gb_var = df.groupby('qt_binning')
    binning_result = gb_var[target].agg({'Totalcnt': 'count',
                                         'Y_rate':np.mean,
                                         'Y_count': np.sum,
                                         'n_Y_count': lambda x: np.sum(1-x),
                                         'Y_pct' : lambda x: np.sum(x / y_count),
                                         'n_Y_pct': lambda x: np.sum((1-x) / (n_sample-y_count))})

    binning_bound = gb_var[var].agg({'Lbound': 'min', 'Ubound': 'max'})

    binning_result['Lbound'] = binning_bound['Lbound']
    binning_result['Ubound'] = binning_bound['Ubound']
    binning_result['PctTotal'] = binning_result['Totalcnt']/n_sample
    binning_result['Y_pct'] = binning_result['Y_pct'].replace(0, 10e-7)

    binning_result['WOE'] = (binning_result['Y_pct']/binning_result['n_Y_pct']).map(lambda x:log(x))
    binning_result['IV'] = (binning_result['Y_pct'] - binning_result['n_Y_pct']) * binning_result['WOE']

    var_iv = binning_result['IV'].sum()

    if return_type == 'iv':
        return pd.Series({var: var_iv})

    elif return_type == 'bins':
        binning_result = binning_result[['Lbound','Ubound','Totalcnt','PctTotal','Y_count','n_Y_count',
                                    'Y_pct','n_Y_pct','Y_rate','WOE','IV']].sort_values('Lbound')
    else: raise ValueError("return_type should either be 'iv' or 'bins'")

    return binning_result


def KS_calculation(y, y_pred, bin_num=20):
    '''
    Calculate KS
    '''
    df = pd.DataFrame()
    df['y'] = y
    df['y_pred'] = y_pred
    n_sample = df.shape[0]
    y_cnt = df['y'].sum()
    #向上取整
    bucket_sample = ceil(n_sample/bin_num)
    #df按y_pred进行排序
    df = df.sort_values('y_pred', ascending = False)
    df['group'] = [ceil(x/bucket_sample) for x in range(1, n_sample+1)]

    grouped = df.groupby('group')['y'].agg({'Totalcnt': 'count',
                                            'Y_rate': np.mean,
                                            'Y_pct': lambda x: np.sum(x / y_cnt),
                                            'n_Y_pct': lambda x: np.sum((1-x) / (n_sample-y_cnt))})
    grouped['Cum_Y_pct'] = grouped['Y_pct'].cumsum()
    grouped['Cum_nY_pct'] = grouped['n_Y_pct'].cumsum()
    grouped['KS'] = (grouped['Cum_Y_pct'] - grouped['Cum_nY_pct']).map(lambda x: abs(x))

    KS = grouped['KS'].max()

    return KS

def _format_bucket(bins, side = "left"):
    '''
    Return successive buckets based on cut-off values specified by 'bins'.

    Parameters
    ==========
    bins: an iterable object (lists, tuples, arrays etc.)
    side: 'left'/'right', the bucket would be left-sided open or right-sided open

    Return
    ======
    levels: list

    Example
    =======
    bins = [0, 5, 10] / side = 'left'
    return = ['(-inf, 0]', '(0, 5]', '(5, 10]', '(10, +inf]']
    '''
    if np.iterable(bins):
        if (np.diff(bins) < 0).any():
            raise ValueError('bins must increase monotonically.')
        else:
            if side == "left":
                levels = ["({0}, {1}]".format(bins[i-1],bins[i]) for i in range(1,len(bins))]
                levels.insert(0, "(-inf, {0}]".format(bins[0]))
                levels.append("({0}, +inf)".format(bins[-1]))
            elif side == "right":
                levels = ["[{0}, {1})".format(bins[i-1],bins[i]) for i in range(1,len(bins))]
                levels.insert(0, "(-inf, {0})".format(bins[0]))
                levels.append("[{0}, +inf)".format(bins[-1]))
            else: raise ValueError("side should either be 'right' or 'left'.")
    else: raise ValueError("bins should be iterables.")

    return levels

def var_bins_series(sr, bins, side = "left"):
    var_index = sr.index
    bins = np.asarray(bins)

    ist_position = bins.searchsorted(sr, side = side)
    na_mask = sr.isnull()
    has_na = na_mask.any()

    buckets = np.asarray(pd.Categorical(ist_position, _format_bucket(bins, side = side), ordered = True, fastpath = True))
    if has_na:
        np.putmask(buckets, na_mask, np.nan)

    return pd.Series(buckets, index = var_index)

def var_bin_woe(df, target, var, bins, side = "left"):

    df[var].fillna(-9999, inplace = True)

    var_series = var_bins_series(df[var], bins, side)
    var_woe_dict = dict(variable_binning(df, target, var, bins , 'bins')["WOE"])

    var_woe_series = var_series.map(lambda x: var_woe_dict.get(x))

    return var_woe_series

def WOE_mapping(df, select_var, target, side = "left"):
    var_list = list(select_var.keys())
    output_df = pd.DataFrame()
    output_df[target] = df[target]

    for i in var_list:
        output_df[i] = var_bin_woe(df, target, i, select_var.get(i), side = side)

    return output_df

#混淆矩阵
def confusion_matrix(true,predicty):
    a=[[0,0],[0,0]]
    for i in range(0,predicty.shape[0]):
        if predicty[i]==0 and true[i]==0:
            a[0][0]=a[0][0]+1
        elif predicty[i]==1 and true[i]==0:
            a[1][0]=a[1][0]+1
        elif predicty[i]==0 and true[i]==1:
            a[0][1]=a[0][1]+1
        else:
            a[1][1]=a[1][1]+1
    FN=a[0][0]
    FP=a[1][0]
    TN=a[0][1]
    TP=a[1][1]
    return  (FN,TN,FP,TP)
