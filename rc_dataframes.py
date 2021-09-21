import pandas as pd
import numpy as np
from tqdm import tqdm


fname = '/scratch365/cmcgrad2/data/data.feather'

print('Loading dataframe...')
df = pd.read_feather(fname)
print('dataframe loaded!')

wc_names_lst = ['cpt', 'ctp', 'cptb', 'cQlMi', 'cQq81', 'cQq11', 'cQl3i',
                'ctq8', 'ctlTi', 'ctq1', 'ctli', 'cQq13','cbW', 'cpQM', 
                'cpQ3', 'ctei', 'cQei', 'ctW', 'ctlSi', 'cQq83', 'ctZ', 'ctG']

wc_ref = [12.88, -0.8, 16.53, 100., 0.99, -0.72, 100., 
          0.93, 100.0, 0.7, 100., 0.68, 7.34, -11.14, 
          5.79, 100., 100., 1., 100., 1.54, -1.25, 0.09]

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

wc_ref_cross = np.array(get_wc_ref_cross(wc_ref), dtype=np.float32)
wc_names_cross = np.array(get_wc_names_cross(wc_names_lst))

ref_weight_avg = df.iloc[:, 12:].dot(wc_ref_cross).mean()
print('Creating final dataframe...')
df.iloc[:, 12:] = df.iloc[:, 12:].multiply(ref_weight_avg)
print('Final dataframe generated!')

print('Saving dataframes...')
for i in tqdm(range(len(wc_names_cross))):
    file = '/scratch365/cmcgrad2/data/dataframes/rc_' + str(wc_names_cross[i])+'.feather'
    df.iloc[:, [0,1,2,3,4,5,6,7,8,9,10,11,i+12]].to_feather(file)
print('Done!')
