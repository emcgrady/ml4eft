import pandas as pd
import awkward as ak
import numpy as np
import math

from coffea import processor
from df_accumulator import DataframeAccumulator
from functions import get_wc_names_cross

class AnalysisProcessor(processor.ProcessorABC):
    def __init__(self, samples, wc_names_lst=[], dtype=np.float32):
        self._samples = samples
        self._wc_names_lst = wc_names_lst
        self._dtype = dtype
        
        self._accumulator = DataframeAccumulator(pd.DataFrame())
        
    def accumulator(self):
        return self._accumulator
        
    
    def process(self, events):
        
        dfa  = self._accumulator
        df = dfa.get()
        
        dataset = events.metadata['dataset']
        
        eft_coeffs = ak.to_numpy(events['EFTfitCoefficients']) if hasattr(events, "EFTfitCoefficients") else None
        if eft_coeffs is not None:
            if self._samples[dataset]['WCnames'] != self._wc_names_lst:
                eft_coeffs = efth.remap_coeffs(self._samples[dataset]['WCnames'], self._wc_names_lst, eft_coeffs)
        
        WC = get_wc_names_cross(self._wc_names_lst)

        eft_coeffs = pd.DataFrame(data = eft_coeffs, columns = WC)
        
        higgs    = events.GenPart[((events.GenPart.pdgId == 25)) & events.GenPart.hasFlags('isLastCopy')]
        top      = events.GenPart[((events.GenPart.pdgId == 6))  & events.GenPart.hasFlags('isLastCopy')]
        anti_top = events.GenPart[((events.GenPart.pdgId == -6)) & events.GenPart.hasFlags('isLastCopy')]
        
        '''df['Higgs px']   = ak.to_pandas(ak.flatten(higgs.pt*np.cos(higgs.phi)))
        df['Higgs py']  = ak.to_pandas(ak.flatten(higgs.pt*np.sin(higgs.phi)))
        df['Higgs pz']  = ak.to_pandas(ak.flatten(higgs.pt*np.sinh(higgs.eta)))
        
        df['Top px']   = ak.to_pandas(ak.flatten(top.pt*np.cos(top.phi)))
        df['Top py']  = ak.to_pandas(ak.flatten(top.pt*np.sin(top.phi)))
        df['Top pz']  = ak.to_pandas(ak.flatten(top.pt*np.sinh(top.eta)))
        
        df['Anti-Top px']   = ak.to_pandas(ak.flatten(anti_top.pt*np.cos(anti_top.phi)))
        df['Anti-Top py']  = ak.to_pandas(ak.flatten(anti_top.pt*np.sin(anti_top.phi)))
        df['Anti-Top pz']  = ak.to_pandas(ak.flatten(anti_top.pt*np.sinh(anti_top.eta)))'''
        
        df['Higgs pt']   = ak.to_pandas(ak.flatten(higgs.pt))
        df['Higgs eta']  = ak.to_pandas(ak.flatten(higgs.eta))
        df['Higgs phi']  = ak.to_pandas(ak.flatten(higgs.phi))
        
        df['Top pt']   = ak.to_pandas(ak.flatten(top.pt))
        df['Top eta']  = ak.to_pandas(ak.flatten(top.eta))
        df['Top phi']  = ak.to_pandas(ak.flatten(top.phi))
        
        df['Anti-Top pt']   = ak.to_pandas(ak.flatten(anti_top.pt))
        df['Anti-Top eta']  = ak.to_pandas(ak.flatten(anti_top.eta))
        df['Anti-Top phi']  = ak.to_pandas(ak.flatten(anti_top.phi))
        
        dfa = dfa.concat(eft_coeffs)
        
        return dfa
    
    def postprocess(self, accumulator):
        return accumulator

    