

from craco.datadirs import SchedDir, ScanDir, format_sbid
from craco.metadatafile import MetadataFile as MF
from craco.candidate_manager import SBCandsManager
from craft.sigproc import SigprocFile as SF
from astropy.coordinates import get_sun, get_body
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

def pcb_path_to_beamid(pcb_path):
    pcb_name = os.path.basename(pcb_path)
    beamid = int(pcb_name.strip("pcbb").strip(".fil"))
    return beamid

def get_last_uncommented_line(file_path):
    '''
    Reads the last uncommented line from the file without looping through all lines
    Thanks Chat-GPT
    '''
    with open(file_path, 'rb') as file:
        file.seek(0, 2)  # Move the pointer to the end of the file
        buffer = b''
        while file.tell() > 0:
            file.seek(-2, 1)  # Move the pointer back two characters
            new_byte = file.read(1)
            if new_byte == b'\n' and buffer:
                line = buffer[::-1].decode().strip()
                if line and not line.startswith("#"):
                    return line
                buffer = b''
            else:
                buffer += new_byte
        # Check the first line in case it’s uncommented
        line = buffer[::-1].decode().strip()
        if line and not line.startswith("#"):
            return line
    return None

def parse_flagging_statistics_line(line):
    '''
    Parses the values in a given line of the flagging statistics file

    Arguments
    ---------
    line: str
        The contents of the line 
    
    Returns
    -------
    flagging_stats: dict
        A dict containing useful info derived from that line
    '''
    parts = line.strip().split("\t")
    info = {}
    info['nblks']= float(parts[0])
    info['num_good_bls_pre'] = float(parts[1])
    info['num_good_cells_pre'] = float(parts[2])
    info['num_good_bls_post'] = float(parts[3])
    info['num_good_cells_post'] = float(parts[4])
    #num_bad_cells_pre = float(parts[5])    We don't need these two values
    #num_bad_cells_post = float(parts[6])
    info['blk_shape'] = tuple(int(x) for x in parts[7][1::-1].split(','))
    info['tot_num_cells'] = float(parts[8])
    info['num_fixed_good_chans'] = float(parts[9])
    return info
    

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

def read_pcb_stats(scandir):
    '''
    Loops over all beam nodes in a scan, and parses the information in each of the pcb filterbank

    Arguments
    ---------
    scandir: Object of datadirs.Scandir() class
    
    Returns
    -------
    filinfo: dict
        A dictionary containing information extracted from pcb headers - keyed by beamid
    '''
    pcbinfo = {}
    for beamid in range(36):
        try:
            filpath = scandir.beam_pcb_path(beamid)
            f = SF(filpath)
            dur = f.nsamples * f.tsamp / 60         #minutes
            bw = np.abs(f.foff) * f.nchans
            fcen = f.fch1 + bw / 2
            ra = f.src_raj_deg
            dec = f.src_dej_deg
            #coord_string = f"{ra_to_hms(ra)}, {dec_to_dms(dec)} ({ra:.4f}, {dec:.4f})"
            beaminfo = {
                'tobs': dur,
                'BW': bw,
                'fcen': fcen,
                'coords_ra_hms': ra_to_hms(ra),
                'coords_ra_deg': ra,
                'coords_dec_dms': dec_to_dms(dec),
                'coords_dec_deg': dec,
            }
        except:
            log.critical(f"!Error: Could not read filterbank information from path - {filpath}!")
            beaminfo = {}
        finally:
            pcbinfo[beamid] = beaminfo

    return pcbinfo


def read_plan_info(scandir):
    '''
    Reads pickled plans for all beams in the given scan and extract all the relevant info

    Arguments
    ---------
    scandir: Object of datadir.Scandir() class

    Returns
    -------

    planinfo: dict
        A dictionary containing relevant info from plans of all beams - keyed by beamid
    
    '''

    planinfo = {}
    for beamid in range(36):
        try:
            planfile = scandir.beam_plan0_path(beamid)
            plan0 = np.load(planfile, allow_pickle=True)
            
            if beamid == 0:
                planinfo['values'] = vars(plan0.values)
                
                freq_info = {}
                freq_info['fmin'] = plan0.fmin
                freq_info['fmax'] = plan0.fmax
                freq_info['nchan'] = plan0.nf
                freq_info['foff'] = plan0.foff
                freq_info['fch1'] = plan0.freqs[0]

                planinfo['freq_info'] = freq_info

                uf = plan0.useful_info()
                planinfo['tsamp_s'] = uf['TSAMP']
                planinfo['nant'] = uf['NANT']
                planinfo['nbl'] = uf['NBL']
                planinfo['start_mjd'] = uf['STARTMJD']
                planinfo['epoch'] = uf['EPOCH']
                planinfo['nowtai'] = uf['NOWTAI']

                planinfo['tstart'] = plan0.tstart
            
            
            beaminfo = {}
            beaminfo['beamid'] = plan0.beamid
            beaminfo['target'] = plan0.target_name
            beaminfo['solar_elong_deg'] = plan0.phase_center.separation(get_sun(plan0.tstart)).deg
            beaminfo['lunar_elong_deg'] = plan0.phase_center.separation(get_body("moon", plan0.tstart)).deg

            wcsinfo = {}
            wcsinfo['coords_ra_deg'] = plan0.ra.deg
            wcsinfo['coords_ra_hms'] = plan0.ra.to_string(sep=":", precision=2)
            wcsinfo['coords_dec_deg'] = plan0.dec.deg
            wcsinfo['coords_dec_dms'] = plan0.dec.to_string(sep=":", precision=2)
            wcsinfo['gl_deg'] = plan0.phase_center.galactic.l.deg
            wcsinfo['gb_deg'] = plan0.phase_center.galactic.b.deg
            wcsinfo['az_deg'] = plan0.craco_wcs.altaz.az.deg
            wcsinfo['alt_deg'] = plan0.craco_wcs.altaz.alt.deg
            wcsinfo['npix'] = plan0.craco_wcs.npix
            wcsinfo['cellsize1_deg'] = plan0.craco_wcs.cellsize[0].deg
            wcsinfo['cellsize2_deg'] = plan0.craco_wcs.cellsize[1].deg
            wcsinfo['fov1_deg'] = (plan0.craco_wcs.cellsize[0] * plan0.craco_wcs.npix).deg
            wcsinfo['fov2_deg'] = (plan0.craco_wcs.cellsize[1] * plan0.craco_wcs.npix).deg
            wcsinfo['hourangle_hr'] = plan0.craco_wcs.hour_angle.hour
            wcsinfo['lst_hr'] = plan0.craco_wcs.lst.hour

            beaminfo['wcs'] = wcsinfo

            planinfo[f'beam_{int(beamid):0>2}'] = beaminfo
        except Exception as E:
            beaminfo = {}
        finally:
            planinfo[f'beam_{int(beamid):0>2}'] = beaminfo
        
    return planinfo


def read_rfi_stats_info(scandir):
    '''
    Loops through all beams in given scandir and reads the RFI statistics logfile, parsing some useful info

    Arguments
    ---------
    scandir: Object of datadirs.ScanDir() class

    Returns
    -------
    rfiinfo: dict
        Dictionary containing the RFI statistics keyed by beamid
    '''

    rfiinfo = {}
    for beamid in range(36):
        beaminfo = {}
        try:
            rfi_stats_path = scandir.beam_rfi_stats_path(beamid)
            last_line = get_last_uncommented_line(rfi_stats_path)
            final_values = parse_flagging_statistics_line(last_line)
            nbl, nf, nt = final_values['blk_shape']

            beaminfo['avg_frac_good_cells'] = final_values['num_good_cells_post'] / final_values['tot_num_cells']
            beaminfo['frac_fixed_good_chans'] = final_values['num_fixed_good_chans'] / nf
            beaminfo['avg_frac_good_pre_flagging'] = final_values['num_good_cells_pre'] / final_values['tot_num_cells']
            beaminfo['avg_flagging_frac'] = (final_values['num_good_cells_pre'] - final_values['num_good_cells_post']) / final_values['tot_num_cells']

        except Exception as E:
            pass
        finally:
            rfiinfo[beamid] = beaminfo

    return rfiinfo



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

        self.pcb_stats = read_pcb_stats(self.scandir)
        self.plan_info = read_plan_info(self.scandir)
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

        # total number of clustered rfi candidates (and in each beam)
        num_clustered_rfi_cands, num_clustered_rfi_cands_per_beam = get_num_candidates(self.cands_manager.clustered_rfi_candfiles)
        candidates_info['num_clustered_rfi_cands']          = num_clustered_rfi_cands
        candidates_info['num_clustered_rfi_cands_per_beam'] = num_clustered_rfi_cands_per_beam

        # total number of bright clustered rfi candidates (and in each beam)
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