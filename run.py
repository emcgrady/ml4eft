#!/usr/bin/env python
import lz4.frame as lz4f
import pickle
import json
import time
import cloudpickle
import gzip
import os
from optparse import OptionParser

import uproot
from coffea import hist, processor
from coffea.util import load, save
from coffea.nanoevents import NanoAODSchema

import pandas as pd
import numpy as np
from tqdm import tqdm
from functions import get_wc_ref_cross, get_wc_names_cross

import gen_processor
from topcoffea.modules import samples
from topcoffea.modules import fileReader

if __name__ == '__main__':
    
    import argparse
    parser = argparse.ArgumentParser(description='You can customize your run')
    parser.add_argument('jsonFiles', nargs='?', default='', help = 'Json file(s) containing files and metadata')
    parser.add_argument('--prefix', '-r', nargs='?', default='', help = 'Prefix or redirector to look for the files')
    parser.add_argument('--test', '-t', action='store_true', help = 'To perform a test, run over a few events in a couple of chunks')
    parser.add_argument('--nworkers','-n'   , default=8  , help = 'Number of workers')
    parser.add_argument('--chunksize','-s'   , default=100000  , help = 'Number of events per chunk')
    parser.add_argument('--nchunks','-c'   , default=None  , help = 'You can choose to run only a number of chunks')
    parser.add_argument('--treename'   , default='Events', help = 'Name of the tree inside the files')
    parser.add_argument('--wc-list', action='extend', nargs='+', help = 'Specify a list of Wilson coefficients to use in filling histograms.')
    parser.add_argument('--int_part', default='all', help='Initial particles')
    
    args = parser.parse_args()
    jsonFiles  = args.jsonFiles
    prefix     = args.prefix
    nworkers   = int(args.nworkers)
    chunksize  = int(args.chunksize)
    nchunks    = int(args.nchunks) if not args.nchunks is None else args.nchunks
    treename   = args.treename
    wc_lst = args.wc_list if args.wc_list is not None else []
    int_part = args.int_part

    ### Load samples from json
    samplesdict = {}
    allInputFiles = []

    def LoadJsonToSampleName(jsonFile, prefix):
        sampleName = jsonFile if not '/' in jsonFile else jsonFile[jsonFile.rfind('/')+1:]
        if sampleName.endswith('.json'): sampleName = sampleName[:-5]
        with open(jsonFile) as jf:
            samplesdict[sampleName] = json.load(jf)
            samplesdict[sampleName]['redirector'] = prefix

    if   isinstance(jsonFiles, str) and ',' in jsonFiles: jsonFiles = jsonFiles.replace(' ', '').split(',')
    elif isinstance(jsonFiles, str)                     : jsonFiles = [jsonFiles]
    for jsonFile in jsonFiles:
        if os.path.isdir(jsonFile):
            if not jsonFile.endswith('/'): jsonFile+='/'
            for f in os.path.listdir(jsonFile):
                if f.endswith('.json'): allInputFiles.append(jsonFile+f)
        else:
            allInputFiles.append(jsonFile)

    # Read from cfg files
    for f in allInputFiles:
        if not os.path.isfile(f):
            print('[WARNING] Input file "%s% not found!'%f)
            continue
        # This input file is a json file, not a cfg
        if f.endswith('.json'): 
            LoadJsonToSampleName(f, prefix)
        # Open cfg files
        else:
            with open(f) as fin:
                print(' >> Reading json from cfg file...')
                lines = fin.readlines()
                for l in lines:
                    if '#' in l: l=l[:l.find('#')]
                    l = l.replace(' ', '').replace('\n', '')
                    if l == '': continue
                    if ',' in l:
                        l = l.split(',')
                        for nl in l:
                            if not os.path.isfile(l): prefix = nl
                            else: LoadJsonToSampleName(nl, prefix)
                    else:
                        if not os.path.isfile(l): prefix = l
                        else: LoadJsonToSampleName(l, prefix)

    flist = {};
    for sname in samplesdict.keys():
        redirector = samplesdict[sname]['redirector']
        flist[sname] = [(redirector+f) for f in samplesdict[sname]['files']]
        samplesdict[sname]['year'] = int(samplesdict[sname]['year'])
        samplesdict[sname]['xsec'] = float(samplesdict[sname]['xsec'])
        samplesdict[sname]['nEvents'] = int(samplesdict[sname]['nEvents'])
        samplesdict[sname]['nGenEvents'] = int(samplesdict[sname]['nGenEvents'])
        samplesdict[sname]['nSumOfWeights'] = float(samplesdict[sname]['nSumOfWeights'])

        # Print file info
        print('>> '+sname)
        print('   - isData?      : %s'   %('YES' if samplesdict[sname]['isData'] else 'NO'))
        print('   - year         : %i'   %samplesdict[sname]['year'])
        print('   - xsec         : %f'   %samplesdict[sname]['xsec'])
        print('   - histAxisName : %s'   %samplesdict[sname]['histAxisName'])
        print('   - options      : %s'   %samplesdict[sname]['options'])
        print('   - tree         : %s'   %samplesdict[sname]['treeName'])
        print('   - nEvents      : %i'   %samplesdict[sname]['nEvents'])
        print('   - nGenEvents   : %i'   %samplesdict[sname]['nGenEvents'])
        print('   - SumWeights   : %f'   %samplesdict[sname]['nSumOfWeights'])
        print('   - Prefix       : %s'   %samplesdict[sname]['redirector'])
        print('   - nFiles       : %i'   %len(samplesdict[sname]['files']))
        for fname in samplesdict[sname]['files']: print('     %s'%fname)

  # Extract the list of all WCs, as long as we haven't already specified one.
    if len(wc_lst) == 0:
        for k in samplesdict.keys():
            for wc in samplesdict[k]['WCnames']:
                if wc not in wc_lst:
                    wc_lst.append(wc)

    if len(wc_lst) > 0:
        if len(wc_lst) == 1:
            wc_print = wc_lst[0]
        elif len(wc_lst) == 2:
            wc_print = wc_lst[0] + ' and ' + wc_lst[1]
        else:
            wc_print = ', '.join(wc_lst[:-1]) + ', and ' + wc_lst[-1]
        print('Wilson Coefficients: {}.'.format(wc_print))
    else:
        print('No Wilson coefficients specified')
        
    print('wc_lst = ' + str(wc_lst))
 
    processor_instance = gen_processor.AnalysisProcessor(samplesdict, wc_lst)

  # Run the processor and get the output
    tstart = time.time()
    output = processor.run_uproot_job(flist, treename=treename, processor_instance=processor_instance,
                                      executor=processor.futures_executor, 
                                      executor_args={"schema": NanoAODSchema,'workers': nworkers},
                                      chunksize=chunksize, maxchunks=nchunks)
    dt = time.time() - tstart
    print("Processing time: %1.2f s with %i workers (%.2f s cpu overall)" % (dt, nworkers, dt*nworkers, ))
    print('Done!')
    
    output = output.get()

    wc_ref = [12.88, -0.8, 16.53, 100., 0.99, -0.72, 100., 
              0.93, 100.0, 0.7, 100., 0.68, 7.34, -11.14, 
              5.79, 100., 100., 1., 100., 1.54, -1.25, 0.09]

    wc_ref_cross = np.array(get_wc_ref_cross(wc_ref), dtype=np.float32)
    wc_names_cross = np.array(get_wc_names_cross(wc_lst))

    end = len(output.columns) - len(wc_ref_cross)
    ins = list(range(end))
    ins.append(len(ins))

    ref_weight_avg = output.iloc[:, end:].dot(wc_ref_cross).mean()
    print('Creating final dataframe...')
    output.iloc[:, end:] = output.iloc[:, end:].multiply(1/ref_weight_avg)
    print('Final dataframe generated!')
    
    print('Saving dataframes...')
    for i in tqdm(range(len(wc_names_cross))):
        ins[len(ins) - 1] = end + i
        file = '/scratch365/cmcgrad2/data/lhe/' + str(wc_names_cross[i])+'.feather'
        output.iloc[:, ins].to_feather(file)
    print('Done!')