import pandas as pd
import awkward as ak
import numpy as np
import math

from coffea import processor
# from df_accumulator import DataframeAccumulator
from functions import get_wc_names_cross
import topcoffea.modules.eft_helper as efth


from coffea.processor import AccumulatorABC
class DataframeAccumulator(AccumulatorABC):
    def __init__(self, df: pd.DataFrame):
        self._df = df
        
    def add(self, other: "DataframeAccumulator") -> "DataframeAccumulator":
        return DataframeAccumulator(pd.concat([self._df, other._df], ignore_index = True))
    
    def __add__(self, other):
        return self.add(other)

    def __iadd__(self, other):
        return self.add(other)
        
    def get(self) -> pd.DataFrame:
        return self._df

    def identity(self):
        return DataframeAccumulator(pd.DataFrame())
    
    def concat(self, df: pd.DataFrame):
        return DataframeAccumulator(pd.concat([self._df, df], axis=1))


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
        df = self._accumulator.get()

        higgs    = events.LHEPart[(events.LHEPart.pdgId == 25)]
        top      = events.LHEPart[(events.LHEPart.pdgId == 6)]
        anti_top = events.LHEPart[(events.LHEPart.pdgId == -6)]
        
        df['Particle 1']  = ak.to_pandas(events.LHEPart.pdgId[:, 0]).astype('int8')
        df['Particle 2']  = ak.to_pandas(events.LHEPart.pdgId[:, 1]).astype('int8')
        df['No. of Jets'] = ak.to_pandas(events.LHE.Njets).astype('int8')
        
        df['shat'] = ak.to_pandas(np.sqrt(-4*events.LHEPart.incomingpz[:,0]*events.LHEPart.incomingpz[:,1])).astype('float32')
        
        df['Higgs px']   = ak.to_pandas(ak.flatten(higgs.pt*np.cos(higgs.phi))).astype('float32')
        df['Higgs py']  = ak.to_pandas(ak.flatten(higgs.pt*np.sin(higgs.phi))).astype('float32')
        df['Higgs pz']  = ak.to_pandas(ak.flatten(higgs.pt*np.sinh(higgs.eta))).astype('float32')
        
        df['Top px']   = ak.to_pandas(ak.flatten(top.pt*np.cos(top.phi))).astype('float32')
        df['Top py']  = ak.to_pandas(ak.flatten(top.pt*np.sin(top.phi))).astype('float32')
        df['Top pz']  = ak.to_pandas(ak.flatten(top.pt*np.sinh(top.eta))).astype('float32')
        
        df['Anti-Top px']   = ak.to_pandas(ak.flatten(anti_top.pt*np.cos(anti_top.phi))).astype('float32')
        df['Anti-Top py']  = ak.to_pandas(ak.flatten(anti_top.pt*np.sin(anti_top.phi))).astype('float32')
        df['Anti-Top pz']  = ak.to_pandas(ak.flatten(anti_top.pt*np.sinh(anti_top.eta))).astype('float32')
        
        df['T-H d eta'] = ak.to_pandas(ak.flatten(top.eta - higgs.eta)).astype('float32')
        df['A-H d eta'] = ak.to_pandas(ak.flatten(anti_top.eta - higgs.eta)).astype('float32')
        df['T-A d eta'] = ak.to_pandas(ak.flatten(top.eta - anti_top.eta)).astype('float32')
        
        df['T-H d phi'] = ak.to_pandas(ak.flatten(top.phi - higgs.phi)).astype('float32')
        df['A-H d phi'] = ak.to_pandas(ak.flatten(anti_top.phi - higgs.phi)).astype('float32')
        df['T-A d phi'] = ak.to_pandas(ak.flatten(top.phi - anti_top.phi)).astype('float32')
        

        dataset = events.metadata['dataset']
        
        eft_coeffs = ak.to_numpy(events['EFTfitCoefficients']) if hasattr(events, "EFTfitCoefficients") else None
        if eft_coeffs is not None:
            if self._samples[dataset]['WCnames'] != self._wc_names_lst:
                eft_coeffs = efth.remap_coeffs(self._samples[dataset]['WCnames'], self._wc_names_lst, eft_coeffs)
        
        WC = get_wc_names_cross(self._wc_names_lst)

        eft_coeffs = pd.DataFrame(data = eft_coeffs, columns = WC)
        
        dfa = dfa.concat(eft_coeffs)
        
        return dfa

    
    def postprocess(self, accumulator):
        return accumulator

    