

from craco.datadirs import SchedDir, ScanDir, format_sbid
from craco.metadatafile import MetadataFile as MF
from craco.candidate_manager import SBCandsManager
import logging
import os
import subprocess
import argparse
from collections import defaultdict

log = logging.getLogger(__name__)
logging.basicConfig(filename="/CRACO/SOFTWARE/craco/craftop/logs/summarise_scan.log",
                    format='[%(asctime)s] %(levelname)s: %(message)s',
                    level=logging.DEBUG)
stdout_handler = logging.StreamHandler(sys.stdout)
log.addHandler(stdout_handler)


def parse_scandir_env(path):
    parts = path.strip().split("/")
    if len(parts) > 0:
        for ip, part in enumerate(parts):
            if part.startswith("SB0"):
                sbid = part
                scanid = parts[ip + 2]
                tstart = parts[ip + 3]
                
                if len(sbid) == 8 and len(scanid) == 2 and len(tstart) == 14:
                    return sbid, scanid, tstart

    raise RuntimeError(f"Could not parse sbid, scanid and tstart from {path}")


def run_with_tsp():
    log.info(f"Queuing up summarise scan")
    EOS_TS_SOCKET = "/data/craco/craco/tmpdir/queues/end_of_scan"
    TMPDIR = "/data/craco/craco/tmpdir"
    environment = {
        "TS_SOCKET": EOS_TS_SOCKET,
        "TMPDIR": TMPDIR,
    }
    ecopy = os.environ.copy()
    ecopy.update(environment)

    try:
        scan_dir = os.environ['SCAN_DIR']
        sbid, scanid, tstart = parse_scandir_env(scan_dir)
    except Exception as KE:
        log.critical(f"Could not fetch the scan directory from environment variables!!")
        log.critical(KE)
        return
    else:
        sbid, scanid, tstart = parse_scandir_env(scan_dir)
        cmd = f"""summarise_scan -sbid {sbid} -scanid {scanid} -tstart {tstart}"""

        subprocess.run(
            [f"tsp {cmd}"], shell=True, capture_output=True,
            text=True, env=ecopy,
        )
        log.info(f"Queued summarise scan job - with command - {cmd}")


def get_metadata_info(scan):
    '''
    Parses the metadata file for a given scan.
    If found and parseable, it returns the list of source names as a string
    If not, returns None

    Arguments
    ---------

    scan: ScanDir() object

    Returns
    -------
    source_names: str or None
                  A string containing a list of source names observed in this scan
                  None if it was unable to retrieve this info
    '''

    metapath = os.path.join(scan.scan_head_dir, "metafile.json")
    if not os.path.exists(metapath):
        emsg = f"Metafile not found at path - {metapath}"
        log.critical(emsg)
    else:
        try:
           mf = MF(metapath)
           source_names = str(list(mf.sources(0).keys()))
           return source_names
           #msg = f"Beam 0 source names - {source_names}\n"
        except Exception as E:
            emsg = f"Could not load the metadata info from {metapath} due to this error - {E}"
            log.critical(emsg)
            pass
            #msg = emsg
    return None


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
        self.sbid = format_sbid(self.scandir.scheddir.sbid, padding=True, prefix=True)
        self.scanid = self.scandir.scan
        self.runname = runname
        self.tstart = tstart

        self.cands_manager = SBCandsManager(self.sbid, runname=self.runname)

        self._dict = {}

    def run_candpipe(self):
        '''
        YM: implement calling of the candpipe from this function       
        from craco.cadpipe import Candpipe, get_parser() etc
        '''
        pass

    def _get_num_candidates(self, candfiles, snr=None):
        num_cands = 0
        num_cands_per_beam = {}

        for candfile in candfiles:
            if snr is None:
                num_cands += candfile.ncands
                num_cands_per_beam[candfile.beamid] = candfile.ncands
            else:
                try:
                    ncands = sum( candfile.cands['snr'] >= snr )
                except KeyError:
                    ncands = sum( candfile.cands['SNR'] >= snr )
                num_cands += ncands
                num_cands_per_beam[candfile.beamid] = ncands
                
        return num_cands, num_cands_per_beam

    
    def _get_classified_candidates(self, candfiles, label='PSR', snr=None):
        pulsar_cands = []
        pulsar_cands_per_beam = {}
        
        for clustered_uniq_candfile in self.cands_manager.clustered_uniq_candfiles:
            if snr is None:
                cands = clustered_uniq_candfile.cands
            else:
                cands = clustered_uniq_candfile.cands[ clustered_uniq_candfile.cands['snr'] >= snr ]

            psr_rows = cands[ cands['LABEL'] == label ]
            pulsar_cands_per_beam[clustered_uniq_candfile.beamid] = psr_rows['MATCH_name'].unique().tolist()
            pulsar_cands += psr_rows['MATCH_name'].unique().tolist()
            
        return list(set(pulsar_cands)), pulsar_cands_per_beam
        

    def get_candidates_info(self, snr=9):
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
        candidates_info = {}
        # total number of raw candidates (and in each beam)
        num_raw_cands, num_raw_cands_per_beam   = self._get_num_candidates(self.cands_manager.raw_candfiles)
        candidates_info['num_raw_cands']             = num_raw_cands
        candidates_info['num_raw_cands_per_beam']    = num_raw_cands_per_beam

        # total number of bright raw candidates (and in each beam)
        num_raw_cands_bright, num_raw_cands_bright_per_beam = self._get_num_candidates(self.cands_manager.raw_candfiles, snr=snr)
        candidates_info['num_raw_cands_bright']          = num_raw_cands_bright
        candidates_info['num_raw_cands_bright_per_beam'] = num_raw_cands_bright_per_beam

        # total number of clustered unique candidates (and in each beam)
        num_clustered_uniq_cands, num_clustered_uniq_cands_per_beam = self._get_num_candidates(self.cands_manager.clustered_uniq_candfiles)
        candidates_info['num_clustered_uniq_cands']          = num_clustered_uniq_cands
        candidates_info['num_clustered_uniq_cands_per_beam'] = num_clustered_uniq_cands_per_beam

        # total number of bright clustered unique candidates (and in each beam)
        num_clustered_uniq_cands_bright, num_clustered_uniq_cands_bright_per_beam = self._get_num_candidates(self.cands_manager.clustered_uniq_candfiles, snr=snr)
        candidates_info['num_clustered_uniq_cands_bright']          = num_clustered_uniq_cands_bright
        candidates_info['num_clustered_uniq_cands_bright_per_beam'] = num_clustered_uniq_cands_bright_per_beam

        # total number of clustered unique candidates (and in each beam)
        num_clustered_rfi_cands, num_clustered_rfi_cands_per_beam = self._get_num_candidates(self.cands_manager.clustered_rfi_candfiles)
        candidates_info['num_clustered_rfi_cands']          = num_clustered_rfi_cands
        candidates_info['num_clustered_rfi_cands_per_beam'] = num_clustered_rfi_cands_per_beam

        # total number of bright clustered unique candidates (and in each beam)
        num_clustered_rfi_cands_bright, num_clustered_rfi_cands_bright_per_beam = self._get_num_candidates(self.cands_manager.clustered_rfi_candfiles, snr=snr)
        candidates_info['num_clustered_rfi_cands_bright']          = num_clustered_rfi_cands_bright
        candidates_info['num_clustered_rfi_cands_bright_per_beam'] = num_clustered_rfi_cands_bright_per_beam

        # total number of clustered candidates (and in each beam) 
        num_clustered_cands = 0
        num_clustered_cands_per_beam = {}
        for clustered_raw_candfile in self.cands_manager.clustered_raw_candfiles:
            num_clustered_cands += clustered_raw_candfile.nclusters
            num_clustered_cands_per_beam[clustered_raw_candfile.beamid] = clustered_raw_candfile.nclusters
        candidates_info['num_clustered_cands']          = num_clustered_cands
        candidates_info['num_clustered_cands_per_beam'] = num_clustered_cands_per_beam

        # total number of candidates for each classification 
        num_classified_cands = defaultdict(int)
        num_classified_cands_per_beam = {}
        for clustered_uniq_candfile in self.cands_manager.clustered_uniq_candfiles:
            value_counts = clustered_uniq_candfile.cands['LABEL'].value_counts()
            num_classified_cands_per_beam[clustered_uniq_candfile.beamid] = value_counts.to_dict()
            for key, count in value_counts.items():
                num_classified_cands[key] += count
            
        candidates_info['num_classified_cands'] = dict(num_classified_cands)
        candidates_info['num_classified_cands_per_beam'] = num_classified_cands_per_beam

        # total of PSR (names) detected in obs 
        pulsar_cands, pulsar_cands_per_beam = self._get_classified_candidates(self.cands_manager.clustered_uniq_candfiles, label='PSR')
        candidates_info['pulsar_cands'] = pulsar_cands 
        candidates_info['pulsar_cands_per_beam'] = pulsar_cands_per_beam

        # total of bright PSR (names) detected in obs 
        pulsar_cands_bright, pulsar_cands_bright_per_beam = self._get_classified_candidates(self.cands_manager.clustered_uniq_candfiles, label='PSR', snr=snr)
        candidates_info['pulsar_cands_bright'] = pulsar_cands_bright 
        candidates_info['pulsar_cands_bright_per_beam'] = pulsar_cands_bright_per_beam
        
        # total of CUSTOM sources (names)
        custom_cands, custom_cands_per_beam = self._get_classified_candidates(self.cands_manager.clustered_uniq_candfiles, label='CUSTOM')
        candidates_info['custom_cands'] = custom_cands 
        candidates_info['custom_cands_per_beam'] = custom_cands_per_beam

        # total of bright CUSTOM sources (names)
        custom_cands_bright, custom_cands_bright_per_beam = self._get_classified_candidates(self.cands_manager.clustered_uniq_candfiles, label='CUSTOM', snr=snr)
        candidates_info['custom_cands_bright'] = custom_cands_bright
        candidates_info['custom_cands_bright_per_beam'] = custom_cands_bright_per_beam
            
        
        self._dict['candidates_info'] = candidates_info
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

    def get_scan_info(self):
        '''
        SBID related info -
            SBID
            Scan ID
            Tstart
            Tobs
            Beamformer weights used by ASKAP
        
        '''
        scan_info = {}
        scan_info['sbid'] = self.sbid
        scan_info['scan_id'] = self.scanid
        scan_info['tstart'] = self.tstart
        


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


    def post_slack_message(self):
        '''
        Compose a nice slack message using the self._dict and the plot
                
        '''

    def run(self):
        self.get_scan_info()
        self.get_observation_params()
        self.get_candidates_info()
        self.get_rfi_stats()
        self.get_search_pipeline_params()
        self.plot_candidates()
        self.post_slack_message()

def main(args):
    obsinfo = ObsInfo(sbid = args.sbid,
                      scanid = args.scanid,
                      tstart = args.tstart)
    obsinfo.run()




if __name__ == '__main__':
    a = argparse.ArgumentParser()
    a.add_argument("-sbid", type=str, help="SBID", required=True)
    a.add_argument("-scanid", type=str, help="scanid", required=True)
    a.add_argument("-tstart", type=str, help="tstart", required=True)

    args = a.parse_args()
    main(args)