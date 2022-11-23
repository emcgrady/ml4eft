import pandas as pd
import numpy as np
from tqdm import tqdm
from functions import get_wc_ref_cross, get_wc_names_cross


fname = '/scratch365/cmcgrad2/data/lhe/gg/px_py_pz/w_jets/gg.feather'

print('Loading dataframe...')
df = pd.read_feather(fname)
print('dataframe loaded!')

wc_names_lst = ['cpt', 'ctp', 'cptb', 'cQlMi', 'cQq81', 'cQq11', 'cQl3i',
                'ctq8', 'ctlTi', 'ctq1', 'ctli', 'cQq13','cbW', 'cpQM', 
                'cpQ3', 'ctei', 'cQei', 'ctW', 'ctlSi', 'cQq83', 'ctZ', 'ctG']

wc_ref = [12.88, -0.8, 16.53, 100., 0.99, -0.72, 100., 
          0.93, 100.0, 0.7, 100., 0.68, 7.34, -11.14, 
          5.79, 100., 100., 1., 100., 1.54, -1.25, 0.09]

wc_ref_cross = np.array(get_wc_ref_cross(wc_ref), dtype=np.float32)
wc_names_cross = np.array(get_wc_names_cross(wc_names_lst))

end = len(df.columns) - len(wc_ref_cross)
ins = list(range(end))
ins.append(len(ins))

ref_weight_avg = df.iloc[:, end:].dot(wc_ref_cross).mean()
print('Creating final dataframe...')
df.iloc[:, end:] = df.iloc[:, end:].multiply(1/ref_weight_avg)
print('Final dataframe generated!')

print('Saving dataframes...')
for i in tqdm(range(len(wc_names_cross))):
    ins[len(ins) - 1] = end + i
    file = '/scratch365/cmcgrad2/data/lhe/gg/px_py_pz/w_jets/dataframes/' + str(wc_names_cross[i])+'.feather'
    df.iloc[:, ins].to_feather(file)
print('Done!')
