import pandas as pd
import numpy as np
import glob
import os
from craco.datadirs import ScanDir

def format_sbid(sbid, padding=True, prefix=True):
    """
    format sbid into a desired format
    """
    if isinstance(sbid, str): sbid = int(
        "".join([i for i in sbid if i.isdigit()])
    )

    sbidstr = ""
    if prefix: sbidstr += "SB"
    if padding: sbidstr += f"{sbid:0>6}"
    else: sbidstr += f"{sbid}"

    return sbidstr

def parse_candfile(candfile, sep="\t", skiprows=0, skipfooter = 0):

    with open(candfile, 'r') as ff:
        while True:
            line = ff.readline()
            if line.strip() == "":
                continue
            HDR_keys = line.strip().strip('#').strip().split(sep)
            break
    #print(f"Header keys = {HDR_keys}")
    df = pd.read_csv(candfile, skiprows=skiprows, skipfooter=skipfooter, sep=sep, header = 0, names = HDR_keys)
    return df

def load_cands(sbid, scanid="*", tstart="*", beam=None, runname="results"):
    sb = format_sbid(sbid)
    if beam is None:
        beam = "*"
    else:
        b = int(beam)
        assert b <= 36, f"Request beam > 36 = {b}"
        beam = f"{b:02g}"
        
    raw_candpath = f"/CRACO/DATA_??/craco/{sb}/scans/{scanid}/{tstart}/{runname}/candidates.b{beam}.txt"
    clustered_raw_candpath = f"/CRACO/DATA_??/craco/{sb}/scans/{scanid}/{tstart}/{runname}/clustering_output/candidates.b{beam}.*rawcat.csv"
    clustered_rfi_candpath = f"/CRACO/DATA_??/craco/{sb}/scans/{scanid}/{tstart}/{runname}/clustering_output/candidates.b{beam}.*rfi.csv"
    clustered_uniq_candpath = f"/CRACO/DATA_??/craco/{sb}/scans/{scanid}/{tstart}/{runname}/clustering_output/candidates.b{beam}.*uniq.csv"
    clustered_inj_candpath = f"/CRACO/DATA_??/craco/{sb}/scans/{scanid}/{tstart}/{runname}/clustering_output/candidates.b{beam}.*inject.cand.csv"
    #TODO - add injection file here
    
    raw_candfiles = glob.glob(raw_candpath)
    clustered_raw_candfiles = glob.glob(clustered_raw_candpath)
    clustered_rfi_candfiles = glob.glob(clustered_rfi_candpath)
    clustered_uniq_candfiles = glob.glob(clustered_uniq_candpath)
    clustered_inj_candfiles = glob.glob(clustered_inj_candpath)

    return raw_candfiles, clustered_raw_candfiles, clustered_rfi_candfiles, clustered_uniq_candfiles, clustered_inj_candfiles
    

def parse_candpath(fname):
    sections = fname.strip().split("/")
    print(sections)
    assert sections[0] == "" and sections[1] == "CRACO" and sections[3] == "craco", f"Doesn't look like a correction path - {fname}"
    sbid = sections[4]
    scanid = sections[6]
    tstart = sections[7]
    final = sections[-1]
    if final.startswith("candidates"):
        final_parts = final.strip().split(".")
        beamid = final_parts[1].strip("b")
        if len(final_parts) == 3:
            filetype = "raw"
        elif len(final_parts) > 3:
            filetype = "clustered_" + final_parts[3]

    return sbid, scanid, tstart, beamid, filetype


class Candfile:
    def __init__(self, candfile):
        self.fname = candfile
        self.sbid, self.scanid, self.tstart, self.beamid, self.type = parse_candpath(self.fname)
        if self.type == "raw":
            self.sep = "\t"
            self.skipfooter = 1
        else:
            self.sep = ","
            self.skipfooter = 0
        
        self.cands = parse_candfile(self.fname, sep=self.sep, skipfooter=self.skipfooter)
        


    def __str__(self):
        return f"Candfile object of {self.fname}"

    __repr__ = __str__
    
    @property
    def ncands(self):
        return len(self.cands)

    @property
    def header(self):
        return list(self.cands.keys())

    @property
    def nclusters(self):
        return self.cands['cluster_id'].nunique()
        


class SBCandsManager:
    def __init__(self, sbname, runname = "results"):
        self.sb = format_sbid(sbname)
        self._raw_candfile_paths, self._clustered_raw_candfile_paths, self._clustered_rfi_candfile_paths, self._clustered_uniq_candfile_paths, self._clustered_inj_candfile_paths = load_cands(self.sb, runname=runname)
        self._candfile_paths = self._raw_candfile_paths + \
                                self._clustered_raw_candfile_paths +\
                                self._clustered_rfi_candfile_paths + \
                                self._clustered_uniq_candfile_paths +\
                                self._clustered_inj_candfile_paths
        
        self.all_candfiles = []
        self.raw_candfiles = []
        self.clustered_raw_candfiles = []
        self.clustered_rfi_candfiles = []
        self.clustered_uniq_candfiles = []
        self.clustered_inj_candfiles = []
        
        for f in self._candfile_paths:           
            if not os.path.exists(f):
                continue
            if f in self._raw_candfile_paths:
                cf = Candfile(f)
                self.raw_candfiles.append(cf)
            if f in self._clustered_raw_candfile_paths:
                cf = Candfile(f)
                self.clustered_raw_candfiles.append(cf)
            if f in self._clustered_rfi_candfile_paths:
                cf = Candfile(f)
                self.clustered_rfi_candfiles.append(cf)
            if f in self._clustered_uniq_candfile_paths:
                cf = Candfile(f)
                self.clustered_uniq_candfiles.append(cf)
            if f in self._clustered_inj_candfile_paths:
                cf = Candfile(f)
                self.clustered_inj_candfiles.append(cf)

            self.all_candfiles.append(cf)

    @property
    def n_candfiles(self):
        return len(self.all_candfiles)

    @property
    def n_rawcandfiles(self):
        return len(self.raw_candfiles)

    @property
    def n_clusteredrawcandfiles(self):
        return len(self.clustered_raw_candfiles)

    @property
    def n_clusteredrficandfiles(self):
        return len(self.clustered_rfi_candfiles)

    @property
    def n_clustereduniqcandfiles(self):
        return len(self.clustered_uniq_candfiles)
        
    @property
    def n_clusteredinjcandfiles(self):
        return len(self.clustered_inj_candfiles)
        
    def filter_candfiles(self, candfiles, scanid=None, tstart=None, beamid=None, selection='AND'):
        '''
        Returns a subset of candfiles which belong to the given scanid/tstart/beamid. 
        The selection can be an intersection (AND) of the three options, or a combination OR)
        candfiles: list of Canfile objects
        '''
        n_candfiles = len(candfiles)
        mask1 = np.zeros(n_candfiles, dtype='bool')
        mask2 = np.zeros(n_candfiles, dtype='bool')
        mask3 = np.zeros(n_candfiles, dtype='bool')
        
        for i, f in enumerate(candfiles):
            if not scanid:
                mask1[:] = True
            elif int(f.scanid) == int(scanid):
                mask1[i] = True
                
            if not tstart:
                mask2[:] = True
            elif int(f.tstart) == int(tstart):
                mask2[i] = True

            if not beamid:
                mask3[:] = True
            elif int(f.beamid) == int(beamid):
                mask3[i] = True

        if selection.upper() == "AND":
            mask = mask1 & mask2 & mask3
        elif selection.upper() == "OR":
            mask = mask1 | mask2 | mask3
        else:
            raise ValueError(f"selection can only be AND or OR, given - {selection}")

        #print(type(mask), mask.dtype, mask)
        return list(np.array(candfiles)[mask])
            
        
class ScanCandsManager:
    def __init__(self, sbname, scanid, tstart, runname = "results"):
        self.sb = format_sbid(sbname, padding=True, prefix=True)
        self.scandir = ScanDir(self.sb, f"{scanid}/{tstart}")
        self._raw_candfile_paths, self._clustered_raw_candfile_paths, self._clustered_rfi_candfile_paths, self._clustered_uniq_candfile_paths, self._clustered_inj_candfile_paths = load_cands(self.sb, runname=runname)
        self._candfile_paths = self._raw_candfile_paths + \
                                self._clustered_raw_candfile_paths +\
                                self._clustered_rfi_candfile_paths + \
                                self._clustered_uniq_candfile_paths +\
                                self._clustered_inj_candfile_paths
        
        self.all_candfiles = []
        self.raw_candfiles = []
        self.clustered_raw_candfiles = []
        self.clustered_rfi_candfiles = []
        self.clustered_uniq_candfiles = []
        self.clustered_inj_candfiles = []
        
        for f in self._candfile_paths:           
            if not os.path.exists(f):
                continue
            if f in self._raw_candfile_paths:
                cf = Candfile(f)
                self.raw_candfiles.append(cf)
            if f in self._clustered_raw_candfile_paths:
                cf = Candfile(f)
                self.clustered_raw_candfiles.append(cf)
            if f in self._clustered_rfi_candfile_paths:
                cf = Candfile(f)
                self.clustered_rfi_candfiles.append(cf)
            if f in self._clustered_uniq_candfile_paths:
                cf = Candfile(f)
                self.clustered_uniq_candfiles.append(cf)
            if f in self._clustered_inj_candfile_paths:
                cf = Candfile(f)
                self.clustered_inj_candfiles.append(cf)

            self.all_candfiles.append(cf)

    @property
    def n_candfiles(self):
        return len(self.all_candfiles)

    @property
    def n_rawcandfiles(self):
        return len(self.raw_candfiles)

    @property
    def n_clusteredrawcandfiles(self):
        return len(self.clustered_raw_candfiles)

    @property
    def n_clusteredrficandfiles(self):
        return len(self.clustered_rfi_candfiles)

    @property
    def n_clustereduniqcandfiles(self):
        return len(self.clustered_uniq_candfiles)
        
    @property
    def n_clusteredinjcandfiles(self):
        return len(self.clustered_inj_candfiles)
        
    def filter_candfiles(self, candfiles, scanid=None, tstart=None, beamid=None, selection='AND'):
        '''
        Returns a subset of candfiles which belong to the given scanid/tstart/beamid. 
        The selection can be an intersection (AND) of the three options, or a combination OR)
        candfiles: list of Canfile objects
        '''
        n_candfiles = len(candfiles)
        mask1 = np.zeros(n_candfiles, dtype='bool')
        mask2 = np.zeros(n_candfiles, dtype='bool')
        mask3 = np.zeros(n_candfiles, dtype='bool')
        
        for i, f in enumerate(candfiles):
            if not scanid:
                mask1[:] = True
            elif int(f.scanid) == int(scanid):
                mask1[i] = True
                
            if not tstart:
                mask2[:] = True
            elif int(f.tstart) == int(tstart):
                mask2[i] = True

            if not beamid:
                mask3[:] = True
            elif int(f.beamid) == int(beamid):
                mask3[i] = True

        if selection.upper() == "AND":
            mask = mask1 & mask2 & mask3
        elif selection.upper() == "OR":
            mask = mask1 | mask2 | mask3
        else:
            raise ValueError(f"selection can only be AND or OR, given - {selection}")

        #print(type(mask), mask.dtype, mask)
        return list(np.array(candfiles)[mask])
            
        
