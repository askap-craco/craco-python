

from craco.datadirs import SchedDir, ScanDir, format_sbid
from craco.metadatafile import MetadataFile as MF
from craco.candidate_manager import SBCandsManager
from craft.sigproc import SigprocFile as SF
import logging
import os
import subprocess
import argparse
import numpy as np
import sys
from collections import defaultdict

log = logging.getLogger(__name__)
logging.basicConfig(filename="/CRACO/SOFTWARE/craco/craftop/logs/summarise_scan.log",
                    format='[%(asctime)s] %(levelname)s: %(message)s',
                    level=logging.DEBUG)
stdout_handler = logging.StreamHandler(sys.stdout)
log.addHandler(stdout_handler)



def dec_to_dms(deg:float) -> str:
    '''
    Converts dec angle in degrees (float) to DD:MM:SS.ss (string)
    '''
    dd = int(deg)
    signum = np.sign(deg)
    if signum >= 0:
        prefix = '+'
    else:
        prefix = '-'

    rem = np.abs(deg - dd)
    mm = int(rem * 60)
    ss = (rem * 60  - mm) * 60
    ss_int = int(ss)
    ss_frac = int((ss - ss_int) * 100)

    return f"{prefix}{np.abs(dd):02g}:{mm:02g}:{ss_int:02g}.{ss_frac:02g}"

def ra_to_hms(ha:float) -> str:
    '''
    Converts ra angle in ha (float) to HH:MM:SS.ss (string)
    '''
    ha /= 15
    hh = int(ha)
    rem = ha - hh
    mm = int(rem * 60)
    ss = (rem*60  - mm) * 60
    ss_int = int(ss)
    ss_frac = int((ss - ss_int) * 100)

    return f"{hh:02g}:{mm:02g}:{ss_int:02g}.{ss_frac:02g}"

def read_filterbank_stats(filpath):
    try:
        f = SF(filpath)
        dur = f.nsamples * f.tsamp / 60         #minutes
        bw = np.abs(f.foff) * f.nchans
        fcen = f.fch1 + bw / 2
        ra = f.src_raj_deg
        dec = f.src_dej_deg
        #coord_string = f"{ra_to_hms(ra)}, {dec_to_dms(dec)} ({ra:.4f}, {dec:.4f})"
        fil_info = {
            'tobs': dur,
            'BW': bw,
            'fcen': fcen,
            'beam0_coords_ra_hms': ra_to_hms(ra),
            'beam0_coords_ra_deg': ra,
            'beam0_coords_dec_dms': dec_to_dms(dec),
            'beam0_coords_dec_deg': dec,
        }
    except:
        log.critical(f"!Error: Could not read filterbank information from path - {filpath}!")
        fil_info = {}
    finally:
        return fil_info


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


def get_num_candidates(candfiles, snr=None):
    '''
    Get number of candidates 

    Arguments
    ---------
    candfiles: list
        a list of Candfile() object 
    snr: float or None
        snr lower limit, return all candidates if None
        return number of candidates brighter than this value if float 

    Returns
    -------
    num_cands: int
        total number of candidates for the whole list 
    num_cands_per_beam: dict 
        number of candidates for each beam/Candfile() with beamid as keys 
    '''
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


def get_num_clusters(candfiles, snr=None):
    '''
    Get number of clusters for candfiles 

    Arguments
    ---------
    candfiles: list
        a list of Candfile() object 
    snr: float or None
        snr lower limit, return all candidates if None
        return number of candidates brighter than this value if float 

    Returns
    -------
    num_clusters: int
        total number of clusters for the whole Candfile() list 
    num_clusters_per_beam: dict 
        number of clusters for each beam/Candfile() with beamid as keys 
    '''
    num_clusters = 0
    num_clusters_per_beam = {}

    for candfile in candfiles:
        if snr is None:
            num_clusters += candfile.nclusters
            num_clusters_per_beam[candfile.beamid] = candfile.nclusters
        else:
            try:
                cands = candfile.cands[ candfile.cands['snr'] >= snr ]
            except KeyError:
                cands = candfile.cands[ candfile.cands['SNR'] >= snr ]
            nclusters = cands['cluster_id'].nunique()
            num_clusters += nclusters
            num_clusters_per_beam[candfile.beamid] = nclusters
            
    return num_clusters, num_clusters_per_beam


def get_num_classified_candidates(candfiles, snr=None):
    '''
    Get a list of candidates names for a specific class (label)

    Arguments
    ---------
    candfiles: list
        a list of Candfile() object, clustered_uniq_candfile (with labels)
    snr: float or None
        snr lower limit, return all candidates if None
        return candidates brighter than this value if float 

    Returns
    -------
    num_classified_cands: dict
        number of classified candidates for each category/label  
    num_classified_cands_per_beam: dict
        number of classified candidates for each category/label per beam (with beamid as keys) 
    '''
    num_classified_cands = defaultdict(int)
    num_classified_cands_per_beam = {}
    
    for candfile in candfiles:
        if snr is None:
            cands = candfile.cands
        else:
            try:
                cands = candfile.cands[ candfile.cands['snr'] >= snr ]
            except KeyError:
                cands = candfile.cands[ candfile.cands['SNR'] >= snr ]

        value_counts = cands['LABEL'].value_counts()
        num_classified_cands_per_beam[candfile.beamid] = value_counts.to_dict()
        for key, count in value_counts.items():
            num_classified_cands[key] += count
        
    return dict(num_classified_cands), num_classified_cands_per_beam



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
        self.tstart = tstart
        self.runname = runname
        self.tstart = tstart

        self.cands_manager = SBCandsManager(self.sbid, runname=self.runname)

        self.filterbank_stats = read_filterbank_stats()
        self._dict = {}

    def run_candpipe(self):
        '''
        YM: implement calling of the candpipe from this function       
        from craco.cadpipe import Candpipe, get_parser() etc
        '''
        pass

    def _form_url(self, cands, beamid):
        '''
        example url:
        http://localhost:8024/candidate?
        sbid=64401
        &beam=17
        &scan=00
        &tstart=20240805162545
        &runname=results
        &dm=46.16743469238281
        &boxcwidth=1
        &lpix=182
        &mpix=167
        &totalsample=11090
        &ra=268.24566650390625
        &dec=-28.110868453979492
        
        General urls based on candidates 

        Arguments
        ---------
        cands: pandas.DataFrame
            a pandas table of candidates that needs to general url 

        Returns
        -------
        urls: list
            a list of urls based on input 
        '''
        urls = 'http://localhost:8024/candidate?' + \
                'sbid=' + format_sbid(self.sbid, padding=False, prefix=False) + \
                '&beam=' + str(beamid) + \
                '&scan=' + str(self.scanid) + \
                '&tstart=' + str(self.tstart) + \
                '&runname=' + str(self.runname) + \
                '&dm=' + cands['dm_pccm3'].astype(str) + \
                '&boxcwidth=' + cands['boxc_width'].astype(str) + \
                '&lpix=' + cands['lpix'].astype(str) + \
                '&mpix=' + cands['mpix'].astype(str) + \
                '&totalsample=' + cands['total_sample'].astype(str) + \
                '&ra=' + cands['ra_deg'].astype(str) + \
                '&dec=' + cands['dec_deg'].astype(str) 

        return urls.tolist()


    def _get_classified_candidates(self, candfiles, label='PSR', snr=None):
        '''
        Get a list of candidates names for a specific class (label)

        Arguments
        ---------
        candfiles: list
            a list of Candfile() object, clustered_uniq_candfile (with labels)
        snr: float or None
            snr lower limit, return all candidates if None
            return candidates brighter than this value if float 
        label: str
            classification name, will return candidates matches this label 
            can be PSR, CUSTOM, RACS, UNKNOWN 

        Returns
        -------
        classified_cands: list
            a list of unique classified (crossmatched) names  
        classified_cands_per_beam: dict
            a list of unique classified (crossmatched) names for each beam with beamid as keys 
        '''
        classified_cands = []
        classified_cands_per_beam = {}
        
        for candfile in candfiles:
            if snr is None:
                cands = candfile.cands
            else:
                try:
                    cands = candfile.cands[ candfile.cands['snr'] >= snr ]
                except KeyError:
                    cands = candfile.cands[ candfile.cands['SNR'] >= snr ]

            classified_rows = cands[ cands['LABEL'] == label ]
            if label != 'UNKNOWN':
                # return crossmatched names for known objects 
                classified_cands_per_beam[candfile.beamid] = classified_rows['MATCH_name'].unique().tolist()
                classified_cands += classified_rows['MATCH_name'].unique().tolist()
            else:
                # return a list of urls for unknown candidates 
                classified_cands_per_beam[candfile.beamid] = self._form_url(classified_rows, candfile.beamid)
                classified_cands += self._form_url(classified_rows, candfile.beamid)
            
        return list(set(classified_cands)), classified_cands_per_beam
        

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
        num_raw_cands, num_raw_cands_per_beam   = get_num_candidates(self.cands_manager.raw_candfiles)
        candidates_info['num_raw_cands']             = num_raw_cands
        candidates_info['num_raw_cands_per_beam']    = num_raw_cands_per_beam

        # total number of bright raw candidates (and in each beam)
        num_raw_cands_bright, num_raw_cands_bright_per_beam = get_num_candidates(self.cands_manager.raw_candfiles, snr=snr)
        candidates_info['num_raw_cands_bright']          = num_raw_cands_bright
        candidates_info['num_raw_cands_bright_per_beam'] = num_raw_cands_bright_per_beam

        # total number of clustered unique candidates (and in each beam)
        num_clustered_uniq_cands, num_clustered_uniq_cands_per_beam = get_num_candidates(self.cands_manager.clustered_uniq_candfiles)
        candidates_info['num_clustered_uniq_cands']          = num_clustered_uniq_cands
        candidates_info['num_clustered_uniq_cands_per_beam'] = num_clustered_uniq_cands_per_beam

        # total number of bright clustered unique candidates (and in each beam)
        num_clustered_uniq_cands_bright, num_clustered_uniq_cands_bright_per_beam = get_num_candidates(self.cands_manager.clustered_uniq_candfiles, snr=snr)
        candidates_info['num_clustered_uniq_cands_bright']          = num_clustered_uniq_cands_bright
        candidates_info['num_clustered_uniq_cands_bright_per_beam'] = num_clustered_uniq_cands_bright_per_beam

        # total number of clustered unique candidates (and in each beam)
        num_clustered_rfi_cands, num_clustered_rfi_cands_per_beam = get_num_candidates(self.cands_manager.clustered_rfi_candfiles)
        candidates_info['num_clustered_rfi_cands']          = num_clustered_rfi_cands
        candidates_info['num_clustered_rfi_cands_per_beam'] = num_clustered_rfi_cands_per_beam

        # total number of bright clustered unique candidates (and in each beam)
        num_clustered_rfi_cands_bright, num_clustered_rfi_cands_bright_per_beam = get_num_candidates(self.cands_manager.clustered_rfi_candfiles, snr=snr)
        candidates_info['num_clustered_rfi_cands_bright']          = num_clustered_rfi_cands_bright
        candidates_info['num_clustered_rfi_cands_bright_per_beam'] = num_clustered_rfi_cands_bright_per_beam

        # total number of clustered candidates (and in each beam) 
        num_clustered_cands, num_clustered_cands_per_beam = get_num_clusters(self.cands_manager.clustered_raw_candfiles)
        candidates_info['num_clustered_cands']          = num_clustered_cands
        candidates_info['num_clustered_cands_per_beam'] = num_clustered_cands_per_beam

        # total number of bright clustered candidates (and in each beam) 
        num_clustered_cands_bright, num_clustered_cands_bright_per_beam = get_num_clusters(self.cands_manager.clustered_raw_candfiles, snr=snr)
        candidates_info['num_clustered_cands_bright']          = num_clustered_cands_bright
        candidates_info['num_clustered_cands_bright_per_beam'] = num_clustered_cands_bright_per_beam

        # total number of candidates for each classification 
        num_classified_cands, num_classified_cands_per_beam = get_num_classified_candidates(self.cands_manager.clustered_uniq_candfiles)
        candidates_info['num_classified_cands'] = num_classified_cands
        candidates_info['num_classified_cands_per_beam'] = num_classified_cands_per_beam

        # total number of bright candidates for each classification 
        num_classified_cands_bright, num_classified_cands_bright_per_beam = get_num_classified_candidates(self.cands_manager.clustered_uniq_candfiles, snr=snr)
        candidates_info['num_classified_cands_bright'] = num_classified_cands_bright
        candidates_info['num_classified_cands_bright_per_beam'] = num_classified_cands_bright_per_beam

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

        # total of UNKNOWN sources (urls)
        unknown_cands, unknown_cands_per_beam = self._get_classified_candidates(self.cands_manager.clustered_uniq_candfiles, label='UNKNOWN')
        candidates_info['unknown_cands'] = unknown_cands 
        candidates_info['unknown_cands_per_beam'] = unknown_cands_per_beam

        # total of bright UNKNOWN sources (urls)
        unknown_cands_bright, unknown_cands_bright_per_beam = self._get_classified_candidates(self.cands_manager.clustered_uniq_candfiles, label='UNKNOWN', snr=snr)
        candidates_info['unknown_cands_bright'] = unknown_cands_bright
        candidates_info['unknown_cands_bright_per_beam'] = unknown_cands_bright_per_beam
            
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
        scan_info['scanid'] = self.scanid
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