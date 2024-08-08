

from craco.datadirs import SchedDir, ScanDir
from craco.metadatafile import MetadataFile
from craco.candidate_manager import SBCandsManager

class ObsInfo:

    def __init__(self, sbid:str, scanid:str, tstart:str, runname:str = 'results'):
        '''
        sbid: str, SBIDs - Can accept SB0xxxxx, 0xxxxx, xxxx formats
        scanid: str, scanid - needs to be in 00 format
        tstart: str, tstart - nees to be in 20240807121212 format
        runname: str, runname - results/inj_r1 etc

        This class cannot summarise injections yet!

        '''
        self.scandir = ScanDir(sbid, f"{scanid}/{tstart}")
        self.sbid = self.scandir.scheddir.sbid
        self.scanid = self.scandir.scan
        self.runname = runname

        self.cands_manager = SBCandsManager(self.sbid, runname=self.runname)

        self._dict = {}

    def run_candpipe(self):
        '''
        YM: implement calling of the candpipe from this function       
        from craco.cadpipe import Candpipe, get_parser() etc
        '''


    def run_with_tsp(self):
        '''
        VG: implement a job that can be scheduled via TSP

        '''


    def get_candidates_info(self):
        '''
        To be implemented by YM

        Candidate properties -
            Total number of raw candidates
            Total number of clustered candidates
            Total number of candidates cross-matched as [Pulsars, RACS, RFI, OTHER KNOWN SOURCES (like repeating FRBs), CRACO discoveries, and UNKNOWNS]
            List of pulsars detected in the obs
            List of custom sources detected
            List (URLs) of unknown sources detected

        Add all these values to the self._dict
        '''
        pass

    def plot_candidates(self):
        '''
        Make a plot of all raw candidates coloured by their classification
        X-axis - RA
        Y-axis - Dec
        '''
        pass


    def get_rfi_stats(self):
        '''
        To be implemented by VG
        Diagnostics -
            RFI statistics
            Dropped packet statistics

        '''
        pass

    def get_search_pipeline_params(self):
        '''
        To be implemented by Keith And VG 
        
        '''
        pass

    def get_sbid_info(self):
        '''
        SBID related info -
            SBID
            Scan ID
            Tstart
            Tobs
            Beamformer weights used by ASKAP
        
        '''
        pass

    def get_observation_params(self):
        '''
        Observation params -
                Beam footprint
                Central freq
                Bandwidth
                Time resolution
                Number of channels
                RA, DEC of every beam center
                Alt, Az of all antenna (mean)
                Gl, Gb of every beam center
                Coordinates of sun
                Guest science data requested (True/False)
        '''

        
        pass