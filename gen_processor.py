import pandas as pd
import awkward as ak
import numpy as np
import math

from coffea import processor
from df_accumulator import DataframeAccumulator

class AnalysisProcessor(processor.ProcessorABC):
    def __init__(self, samples, wc_names_lst=[], do_errors=False, do_systematics=False, dtype=np.float32):
        self._samples = samples
        self._wc_names_lst = wc_names_lst
        self._dtype = dtype
        self._do_errors = do_errors
        
        self._accumulator = DataframeAccumulator(pd.DataFrame())
        
    def accumulator(self):
        return self._accumulator
        
    
    def process(self, events):
        df = self._accumulator.identity()
        
        dataset = events.metadata['dataset']
        
        eft_coeffs = ak.to_numpy(events['EFTfitCoefficients']) if hasattr(events, "EFTfitCoefficients") else None
        if eft_coeffs is not None:
            if self._samples[dataset]['WCnames'] != self._wc_names_lst:
                eft_coeffs = efth.remap_coeffs(self._samples[dataset]['WCnames'], self._wc_names_lst, eft_coeffs)
                
        WC = []
        self._wc_names_lst.insert(0,'SM')
        
        for i in range(276):
            WC.append(self._wc_names_lst[math.floor((-1+math.sqrt(9-8*(1-i)))/2)] + '*' 
                      + self._wc_names_lst[i - int((math.floor((-1+math.sqrt(9-8*(1-i)))/2)+1)*
                                        math.floor((-1+math.sqrt(9-8*(1-i)))/2)/2)])
        
        eft_coeffs = pd.DataFrame(data = eft_coeffs, columns = [WC])
        
        higgs    = events.GenPart[((events.GenPart.pdgId == 25)) & events.GenPart.hasFlags('isLastCopy')]
        top      = events.GenPart[((events.GenPart.pdgId == 6)) & events.GenPart.hasFlags('isLastCopy')]
        anti_top = events.GenPart[((events.GenPart.pdgId == -6)) & events.GenPart.hasFlags('isLastCopy')]
        
        df['Higgs pt']   = ak.to_pandas(higgs.pt)
        df['Higgs eta']  = ak.to_pandas(higgs.eta)
        df['Higgs phi']  = ak.to_pandas(higgs.phi)
        df['Higgs mass'] = ak.to_pandas(higgs.mass)
        
        df['Top pt']   = ak.to_pandas(top.pt)
        df['Top eta']  = ak.to_pandas(top.eta)
        df['Top phi']  = ak.to_pandas(top.phi)
        df['Top mass'] = ak.to_pandas(top.mass)
        
        df['Anti-Top pt']   = ak.to_pandas(anti_top.pt)
        df['Anti-Top eta']  = ak.to_pandas(anti_top.eta)
        df['Anti-Top phi']  = ak.to_pandas(anti_top.phi)
        df['Anti-Top mass'] = ak.to_pandas(anti_top.mass)
        
        df = df.reset_index(level=1, drop=True).reset_index(level=0, drop=True)
        df = pd.concat((df, eft_coeffs), axis=1)
        
        return df
    
    def postprocess(self, accumulator):
        return accumulator

    