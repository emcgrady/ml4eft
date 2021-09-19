import math
import pandas as pd
import numpy as np
import awkward as ak
from coffea.nanoevents import NanoEventsFactory, NanoAODSchema
from df_accumulator import DataframeAccumulator
from pyarrow import feather

fname = '/scratch365/hnelson2/data/data.feather'

#entire dataframe from feather file
df = feather.read_feather(fname)
#dataframe of just EFTFitCoefficients, in current setup, EFT starts in col12
df_eft = df.iloc[:, 12:]
#dataframe of just parton level information "z"
df_z = df.iloc[:, 0:12]

#standard listing of WC's with the corresponding WC reference values
#WC reference values are the values of MGSTART from
#startpts_scale_by_1p1_ttHJet.txt in TopEft/GridpackGeneration
wc_dict = {'cpt':12.88, 'ctp':-0.8, 'cptb':16.53, 'cQlMi':100.0,
    'cQq81':0.99, 'cQq11':-0.72, 'cQl3i':100.0, 'ctq8':0.93, 'ctlTi':100.0,
    'ctq1':0.7, 'ctli':100.0, 'cQq13':0.68, 'cbW':7.34, 'cpQM':-11.14,
    'cpQ3':5.79, 'ctei':100.0, 'cQei':100.0, 'ctW':1.0, 'ctlSi':100.0,
    'cQq83':1.54, 'ctZ':-1.25, 'ctG':0.09}

wc_names_lst = list(wc_dict.keys())
wc_ref = list(wc_dict.values())

def get_wc_ref_cross(wc_ref):
    '''returns the list of 276 WCs and cross terms evalulated
    at the initial WC values provided from wc_ref'''

    # The order of the wilson coeff array is the "lower triangle" of the matrix
    # I.e. if the list of WC names is [c1,c2,...,cn], the order of the quad terms is:
    #     quad terms = [
    #         sm*sm,
    #         c1*sm, c1*c1,
    #         c2*sm, c2*c1, c2*c2,
    #         ...
    #         cn*sm,  ... , cn*cn]

    # Prepend SM value to wc_ref
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

    #Prepend "SM" to wc_names_lst
    wc_names_lst = ["SM"] + wc_names_lst

    wc_names_cross=[]
    for i in range(len(wc_names_lst)):
        for j in range(i+1):
            term = wc_names_lst[i]+"*"+wc_names_lst[j]
            wc_names_cross.append(term)

    return wc_names_cross

wc_ref_cross = np.array(get_wc_ref_cross(wc_ref))
wc_names_cross = np.array(get_wc_names_cross(wc_names_lst))

#to calculate the mean reference weight, take the dot product of the
#EFTFitCoefficients and WC reference values for each event, and average
ref_weight = np.dot(df_eft,wc_ref_cross)
ref_weight_avg = np.mean(ref_weight)

#data frame of each r_c value for each event
df_r = df_eft.multiply(ref_weight_avg)

#make a dataframe for each WC in wc_names_lst where each df contains df_z
#plus the column of r_c for chosen WC, save to a HDF5 file with the df
#key corresponding to the name of the chosen WC
for i in range(len(wc_names_cross)):
    #we could save each df into one hdf5 file with a unique key, but when I did this 
    #the size of the file was ~100GB for just 1/3 of the data, so we will need to save
    #to feather files to keep the memory size managable

    #df_rc = df_z.join(df_r[wc_names_cross[i]])
    #df_rc.to_hdf('/scratch365/hnelson2/hdf/rc_dfs.h5', key = wc_names_cross[i], mode='a')

    file = '/scratch365/hnelson2/feather/rc_' + str(wc_names_cross[i])+'.feather'
    df_rc.to_feather(file)
