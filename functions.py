import torch
import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import inv_boxcox
import matplotlib.pyplot as plt

def get_wc_ref_cross(wc_ref):
    '''returns the list of 276 WCs and cross terms evalulated
    at the initial WC values provided from wc_ref'''

    wc_ref = [1] + wc_ref

    wc_ref_cross=[]
    for i in range(len(wc_ref)):
        for j in range(i+1):
            term = wc_ref[i]*wc_ref[j]
            wc_ref_cross.append(term)

    return wc_ref_cross

def get_wc_names_cross(wc_names_lst):
    '''returns the list of names for 276 WCs and cross terms
    with the same ordering as get_wc_ref_cross'''

    wc_names_lst = ["SM"] + wc_names_lst

    wc_names_cross=[]
    for i in range(len(wc_names_lst)):
        for j in range(i+1):
            term = wc_names_lst[i]+"*"+wc_names_lst[j]
            wc_names_cross.append(term)

    return wc_names_cross

def box_cox(array):
    if np.min(array) < 0:
        array = array - np.min(array)*2
    lmbda = stats.boxcox_normmax(array, method='pearsonr')
    out = stats.boxcox(array, lmbda=lmbda, alpha=None)
    return (out, lmbda)
   
def norm(df):
    mean = df.mean()
    std  = df.std()
    out  = (df-mean)/std
    return (out, std, mean)

def norm_inv(df, std, mean):
    out = df*std+mean
    return out

def box_cox_inv(array, lmbda):
    out = inv_boxcox(array, lmbda)
    return out

def log_tf(df):
    df['r_c'] = np.log((df['r_c'] - df['r_c'].min()) + (df['r_c'] - df['r_c'].min()).nsmallest(2).iloc[-1])
    return df

def hist(df, title, filename, datatype, coef, log=True):
    stat = df.agg(['skew', 'kurtosis']).transpose()
    k2 = stats.normaltest(df['r_c'].to_numpy())[0]

    plt.rcParams['figure.figsize'] = [15, 8]

    n, bins, patches = plt.hist(df['r_c'], 100, density=False, alpha=0.75)

    xlim = np.max(bins)
    ylim = np.max(n)

    plt.xlabel('$\mathregular{r_c}$', fontsize=14)
    plt.ylabel('Frequency', fontsize=14)
    plt.title(str(coef + ' Distribution of Outputs' + title), fontsize=16)
    plt.grid(True)
    if log == True:
        plt.yscale('log')
    plt.savefig(str('plots/' + coef + '/' + datatype + '/histograms/' + filename + '.png'))
    plt.show()
    return(bins, n)

def scatter(df, name, kind, axt, title, subset, datatype, coef, log=False, color='b.'):
    plt.plot(df[name].to_numpy(), df['r_c'].to_numpy(), color)
    plt.grid(b=True, color='grey', alpha=0.2, linestyle=':', linewidth=2)
    plt.xlabel(axt, fontsize=15)
    plt.xticks(fontsize=13)
    plt.ylabel('$r_c$', fontsize=15)
    plt.yticks(fontsize=13)
    plt.title(str( coef +  title + ' Dataset Output vs Input'), fontsize=16)
    if log == True:
        plt.yscale('log')
    plt.savefig(str('plots/' + coef + '/' + datatype + '/scatter_plots/' + subset + '/rc_vs_' + kind + '.png'))
    plt.show()