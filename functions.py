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