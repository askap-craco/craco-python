

from craco.datadirs import SchedDir, ScanDir, format_sbid
from craco.metadatafile import MetadataFile as MF
from craco.candidate_manager import SBCandsManager, ScanCandsManager
from craco.craco_run.auto_sched import SlackPostManager
from craft.sigproc import SigprocFile as SF
from astropy.coordinates import get_sun, get_body
from astropy import time as T
import logging
import os
import subprocess
import datetime
import argparse
import numpy as np
import sys
from collections import defaultdict
import glob
import traceback
import IPython
import warnings
import json

warnings.filterwarnings("ignore")

log = logging.getLogger(__name__)
username = os.environ['USER']

if username == 'gup037':
    logname = "/tmp/tmp.tmp"
elif username == 'craftop':
    logname = "/CRACO/SOFTWARE/craco/craftop/logs/summarise_scan.log"
else:
    #raise RuntimeError("Oi - what are you doing with my code -- shoo!")
    # Need to remove this runtime error as it breaks unit test discover running as ban115
    logname = 'summarise_scan.log'

logging.basicConfig(filename=logname,
                    format='[%(asctime)s] %(levelname)s: %(message)s',
                    level=logging.DEBUG)
stdout_handler = logging.StreamHandler(sys.stdout)
log.addHandler(stdout_handler)

null_value = None

'''
        spatial_pixels_beam0 = planinfo['beam_00']['wcs']['npix']
        spatial_pixels_min = min(planinfo[key]['wcs']['npix'] for key in planinfo if key.startswith('beam_'))
        spatial_pixels_max = max(planinfo[key]['wcs']['npix'] for key in planinfo if key.startswith('beam_'))

'''

class TrivialEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, T.Time):
            d = o.iso
        elif isinstance(o, np.int64):
            d = int(o)
        else:
            d = super().default(o)
        return d
    
class ReadInfoException(Exception):
    def __init__(self, error_msg, exc):
        self.my_message = error_msg
        self.exc = exc
        super().__init__(exc)

def search_dict(d, key):
    '''
    Traverses through the dictionary d and finds the value for key key.
    If not found, returns None
    '''
    if key in d:
        return d[key]
    for k, v in d.items():
        if isinstance(v, dict):
            return search_dict(v, key)
    return None
    
def parse_tstart_as_ISO_time(tstart):
    '''
    Takes tstart string in the format YYYYMMDDHHMMSS
    and converts it into the ISO time format string
    i.e. yyyy-MM-ddTHH:mm:ss.SSSXXX
    '''
    time = datetime.datetime.strptime(tstart, "%Y%m%d%H%M%S")
    isot = time.isoformat()
    return isot

def find_beam0_min_max_values(d, key):
    '''
    Takes a dictionary d and a key and finds the value of key for beam0,
    and its min and max values across all beams

    Assumes that all beam sub-dicts are indexed using the pattern beam_00, beam_24, etc

    If a certain beam is not found, it simply ignores it
    If beam 0 is not found, it reports None for the beam 0 value
    '''
    values = []

    beam0_val = search_dict(d['beam_00'], key)
    for beamid in range(36):
        val = search_dict(d[f'beam_{beamid:0>2}'], key)
        if val is not None:
            values.append(val)

    if len(values) == 0:
        return None, None, None

    return beam0_val, min(values), max(values)


def pcb_path_to_beamid(pcb_path):
    pcb_name = os.path.basename(pcb_path)
    beamid = int(pcb_name.strip("pcbb").strip(".fil"))
    return beamid

def get_last_uncommented_line(file_path):
    '''
    Reads the last uncommented line from the file without looping through all lines
    Thanks Chat-GPT
    '''
    if file_path and os.path.exists(file_path):
        with open(file_path, 'rb') as file:
            file.seek(0, 2)  # Move the pointer to the end of the file
            buffer = b''
            while file.tell() > 1:
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
    info['avg_num_good_bls_pre_flagging'] = float(parts[1])
    info['avg_num_good_cells_pre_flagging'] = float(parts[2])
    info['avg_num_good_bls_post_flagging'] = float(parts[3])
    info['avg_num_good_cells_post_flagging'] = float(parts[4])
    #num_bad_cells_pre = float(parts[5])    We don't need these two values
    #num_bad_cells_post = float(parts[6])
    info['blk_shape'] = tuple(int(x) for x in parts[7][1:-1].split(','))
    info['tot_num_cells_per_blk'] = float(parts[8])
    info['num_fixed_good_chans_per_blk'] = float(parts[9])
    info['avg_dropped_packets_frac'] = float(parts[10])
    return info
    

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

    rfiinfo = {
        'frac_bls_good': [],
        'frac_bls_bad': [],
        'frac_fixed_good_chans': [],
        'frac_fixed_bad_chans': [],
        'frac_good_cells_post_flagging': [],
        'frac_good_cells_pre_flagging': [],
        'flagging_frac': [],
        'dropped_packets_frac': []
    }

    found_a_rfi_stats_file = False
    for beamid in range(36):
        rfi_stats_path = scandir.beam_rfi_stats_path(beamid)
        try:
            last_line = get_last_uncommented_line(rfi_stats_path)
            if not last_line:
                #The rfi file is empty or the file doesn't exist
                continue
            final_values = parse_flagging_statistics_line(last_line)
            nbl, nf, nt = final_values['blk_shape']

            rfiinfo['frac_bls_good'].append(final_values['avg_num_good_bls_pre_flagging'] / nbl)
            rfiinfo['frac_bls_bad'].append(1 - rfiinfo['frac_bls_good'][-1])

            rfiinfo['frac_fixed_good_chans'].append(final_values['num_fixed_good_chans_per_blk'] / nf)
            rfiinfo['frac_fixed_bad_chans'].append(1 - rfiinfo['frac_fixed_good_chans'][-1])

            rfiinfo['frac_good_cells_post_flagging'].append(final_values['avg_num_good_cells_post_flagging'] / final_values['tot_num_cells_per_blk'])
            rfiinfo['frac_good_cells_pre_flagging'].append(final_values['avg_num_good_cells_pre_flagging'] / final_values['tot_num_cells_per_blk'])
            rfiinfo['flagging_frac'].append((final_values['avg_num_good_cells_pre_flagging'] - final_values['avg_num_good_cells_post_flagging']) / final_values['tot_num_cells_per_blk'])
            rfiinfo['dropped_packets_frac'].append(final_values['avg_dropped_packets_frac'])

            found_a_rfi_stats_file = True

        except Exception as E:
            my_error = f"!Error: Could not read flagging information from path - {rfi_stats_path}!\n{E}"
            log.exception(my_error)

            rfiinfo['frac_bls_good'].append(null_value)
            rfiinfo['frac_bls_bad'].append(null_value)

            rfiinfo['frac_fixed_good_chans'].append(null_value)
            rfiinfo['frac_fixed_bad_chans'].append(null_value)

            rfiinfo['frac_good_cells_post_flagging'].append(null_value)
            rfiinfo['frac_good_cells_pre_flagging'].append(null_value)
            rfiinfo['flagging_frac'].append(null_value)
            rfiinfo['dropped_packets_frac'].append(null_value)

            #raise ReadInfoException(my_error, E)
            #log.critical(traceback.format_exc())
            #IPython.embed()

    if not found_a_rfi_stats_file:
        rfiinfo = {}
    return rfiinfo


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
    found_a_pcb = False
    BW_values = []
    tobs_values = []
    fcen_values = []
    coords_ra_hms_values = []
    coords_dec_dms_values = []
    coords_ra_deg_values = []
    coords_dec_deg_values = []
    for beamid in range(36):
        try:
            filpath = scandir.beam_pcb_path(beamid)
            f = SF(filpath)
            tobs = f.nsamples * f.tsamp / 60         #minutes
            bw = np.abs(f.foff) * f.nchans
            fcen = f.fch1 + bw / 2
            ra = f.src_raj_deg
            dec = f.src_dej_deg
            #coord_string = f"{ra_to_hms(ra)}, {dec_to_dms(dec)} ({ra:.4f}, {dec:.4f})"
            coords_ra_hms = ra_to_hms(ra)
            coords_ra_deg = ra
            coords_dec_dms = dec_to_dms(dec)
            coords_dec_deg = dec

            found_a_pcb = True

        except Exception as E:
            log_msg = f"!Error: Could not read filterbank information from path - {filpath}!\n{E}"
            log.exception(log_msg)
            
            tobs = null_value
            bw = null_value
            fcen = null_value
            coords_ra_hms = null_value
            coords_ra_deg = null_value
            coords_dec_dms = null_value
            coords_dec_deg = null_value

            #raise ReadInfoException(log_msg, E)
            #beaminfo = {}
        finally:
            tobs_values.append(tobs)
            BW_values.append(bw)
            fcen_values.append(fcen)
            coords_ra_hms_values.append(coords_ra_hms)
            coords_ra_deg_values.append(coords_ra_deg)
            coords_dec_dms_values.append(coords_dec_dms)
            coords_dec_deg_values.append(coords_dec_deg)
        #    pcbinfo[f'beam_{beamid:0>2}'] = beaminfo
    

    pcbinfo = {
        'tobs' : tobs_values,
        'BW': BW_values,
        'fcen': fcen_values,
        'coords_ra_hms': coords_ra_hms_values,
        'coords_ra_deg': coords_ra_deg_values,
        'coords_dec_dms': coords_dec_dms_values,
        'coords_dec_deg': coords_dec_deg_values,
    }

    if not found_a_pcb:
        pcbinfo = {}
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

    planinfo = {
        'beamid': [],
        'target': [],
        'solar_elong_deg': [],
        'lunar_elong_deg': [],
        'wcs': {
            'coords_ra_deg': [],
            'coords_ra_hms': [],
            'coords_dec_deg': [],
            'coords_dec_dms': [],
            'gl_deg': [],
            'gb_deg': [],
            'az_deg': [],
            'alt_deg': [],
            'npix': [],
            'cellsize1_deg': [],
            'cellsize2_deg': [],
            'fov1_deg': [],
            'fov2_deg': [],
            'hourangle_hr': [],
            'lst_hr': []
        }
    }
    found_a_plan = False
    for beamid in range(36):
        planfile = scandir.beam_plan0_path(beamid)
        try:
            plan0 = np.load(planfile, allow_pickle=True)
            
            if not found_a_plan:        #This means that we save the values of the first plan we found, may or may not be beam 0
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
                planinfo['ants'] = list(set(range(1,30)) - set(plan0.values.flag_ants))
                planinfo['nbl'] = uf['NBL']
                planinfo['start_mjd'] = uf['STARTMJD']
                planinfo['epoch'] = uf['EPOCH']
                planinfo['nowtai'] = uf['NOWTAI']

                planinfo['tstart'] = plan0.tstart
            
            planinfo['beamid'].append(plan0.beamid)
            planinfo['target'].append(plan0.target_name)
            planinfo['solar_elong_deg'].append(plan0.phase_center.separation(get_sun(plan0.tstart)).deg)
            planinfo['lunar_elong_deg'].append(plan0.phase_center.separation(get_body("moon", plan0.tstart)).deg)

            planinfo['wcs']['coords_ra_deg'].append(plan0.ra.deg)
            planinfo['wcs']['coords_ra_hms'].append(ra_to_hms(plan0.ra.deg))
            planinfo['wcs']['coords_dec_deg'].append(plan0.dec.deg)
            planinfo['wcs']['coords_dec_dms'].append(dec_to_dms(plan0.dec.deg))
            planinfo['wcs']['gl_deg'].append(plan0.phase_center.galactic.l.deg)
            planinfo['wcs']['gb_deg'].append(plan0.phase_center.galactic.b.deg)
            planinfo['wcs']['az_deg'].append(plan0.craco_wcs.altaz.az.deg)
            planinfo['wcs']['alt_deg'].append(plan0.craco_wcs.altaz.alt.deg)
            planinfo['wcs']['npix'].append(plan0.craco_wcs.npix)
            planinfo['wcs']['cellsize1_deg'].append(plan0.craco_wcs.cellsize[0].deg)
            planinfo['wcs']['cellsize2_deg'].append(plan0.craco_wcs.cellsize[1].deg)
            planinfo['wcs']['fov1_deg'].append((plan0.craco_wcs.cellsize[0] * plan0.craco_wcs.npix).deg)
            planinfo['wcs']['fov2_deg'].append((plan0.craco_wcs.cellsize[1] * plan0.craco_wcs.npix).deg)
            planinfo['wcs']['hourangle_hr'].append(plan0.craco_wcs.hour_angle.hour)
            planinfo['wcs']['lst_hr'].append(plan0.craco_wcs.lst.hour)

            found_a_plan = True

        except Exception as E:
            my_error = f"!Error: Could not read plan information from path - {planfile}!\n{E}"
            log.exception(my_error)
            #raise ReadInfoException(my_error, E)
            planinfo['beamid'].append(null_value)
            planinfo['target'].append(null_value)
            planinfo['solar_elong_deg'].append(null_value)
            planinfo['lunar_elong_deg'].append(null_value)

            planinfo['wcs']['coords_ra_deg'].append(null_value)
            planinfo['wcs']['coords_ra_hms'].append(null_value)
            planinfo['wcs']['coords_dec_deg'].append(null_value)
            planinfo['wcs']['coords_dec_dms'].append(null_value)
            planinfo['wcs']['gl_deg'].append(null_value)
            planinfo['wcs']['gb_deg'].append(null_value)
            planinfo['wcs']['az_deg'].append(null_value)
            planinfo['wcs']['alt_deg'].append(null_value)
            planinfo['wcs']['npix'].append(null_value)
            planinfo['wcs']['cellsize1_deg'].append(null_value)
            planinfo['wcs']['cellsize2_deg'].append(null_value)
            planinfo['wcs']['fov1_deg'].append(null_value)
            planinfo['wcs']['fov2_deg'].append(null_value)
            planinfo['wcs']['hourangle_hr'].append(null_value)
            planinfo['wcs']['lst_hr'].append(null_value)

        #finally:
        #    planinfo[f'beam_{int(beamid):0>2}'] = beaminfo
        
    if not found_a_plan:
        planinfo = {}

    return planinfo


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

def extract_values_from_url(url):
    # Split the URL to get the query string part
    query_string = url.split('?')[1]

    # Split the query string into individual key-value pairs
    parameters = query_string.split('&')

    # Initialize variables to store the values of beam and totalsample
    beam = None
    totalsample = None

    # Iterate through the parameters and find the required values
    for param in parameters:
        key, value = param.split('=')
        if key == 'beam':
            beam = value
        elif key == 'totalsample':
            totalsample = value

    return beam, totalsample


def convert_urls_to_readable_links(url_list):
    '''
    Takes a list of URLs and converts them into markdown text with hyperlinked text that is more readable
    Returns a list    
    '''
    converted_urls = []
    for url in url_list:
        beam, totalsample = extract_values_from_url(url)
        #converted_urls.append(f"[Beam {beam} Totalsample {totalsample}]({url})")
        converted_urls.append(f"Cand in <{url}|Beam {beam} Totalsample {totalsample}>")

    return converted_urls

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
            log.critical(traceback.format_exc())
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
    num_cands_per_beam = []

    if len(candfiles) > 0:
        for candfile in candfiles:
            if snr is None:
                num_cands += candfile.ncands
                num_cands_per_beam.append(candfile.ncands)
            else:
                try:
                    ncands = sum( candfile.cands['snr'] >= snr )
                except KeyError:
                    ncands = sum( candfile.cands['SNR'] >= snr )
                num_cands += ncands
                num_cands_per_beam.append(ncands)
            
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
    num_clusters_per_beam = []

    if len(candfiles) > 0:
        for candfile in candfiles:
            if snr is None:
                num_clusters += candfile.nclusters
                num_clusters_per_beam.append(candfile.nclusters)
            else:
                try:
                    cands = candfile.cands[ candfile.cands['snr'] >= snr ]
                except KeyError:
                    cands = candfile.cands[ candfile.cands['SNR'] >= snr ]
                nclusters = cands['cluster_id'].nunique()
                num_clusters += nclusters
                num_clusters_per_beam.append(nclusters)
            
    return num_clusters, num_clusters_per_beam

#VG - change this function to return list with values for each beam, instead of dict
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
    
    if len(candfiles) > 0:
        for candfile in candfiles:
            if snr is None:
                cands = candfile.cands
            else:
                try:
                    cands = candfile.cands[ candfile.cands['snr'] >= snr ]
                except KeyError:
                    cands = candfile.cands[ candfile.cands['SNR'] >= snr ]

            value_counts = cands['LABEL'].value_counts()
            #num_classified_cands_per_beam[f'beam_{candfile.beamid:0>2}'] = value_counts.to_dict()
            for key, count in value_counts.items():
                num_classified_cands[key] += count
        
    return dict(num_classified_cands)#, num_classified_cands_per_beam

def format_pandas_row_as_string(row, beamid, snr_key):
    return f"B{int(beamid):02g}_MJD{row['mjd']}_DM{row['dm_pccm3']:.2f}_SNR{row[snr_key]:.1f}_BOX{row['boxc_width']}_RA{row['ra_deg']:.4f}_DEC{row['dec_deg']:.3f}"

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



class ObsInfo:

    def __init__(self, sbid:str, scanid:str, tstart:str, runname:str = 'results', runcandpipe=True, block_slack_post=False):
        '''
        sbid: str, SBIDs - Can accept SB0xxxxx, 0xxxxx, xxxx formats
        scanid: str, scanid - needs to be in 00 format
        tstart: str, tstart - nees to be in 20240807121212 format
        runname: str, runname - results/inj_r1 etc

        This class cannot summarise injections yet!

        '''
        self.scandir = ScanDir(sbid, f"{scanid}/{tstart}")
        self.sbid = format_sbid(self.scandir.scheddir.sbid, padding=True, prefix=True)
        self.scan = self.scandir.scan
        self.scanid = scanid 
        self.tstart = tstart
        self.runname = runname
        self.tstart = tstart
        self.block_slack_post = block_slack_post
        self._dict = {}
        self.run(runcandpipe = runcandpipe)

    def run(self, runcandpipe=True):
        try:
            self._dict["docid"] = f"{self.sbid}_scan_{self.scanid}_tstart_{self.tstart}_run_{self.runname}"
            self._dict["tstamp"] = parse_tstart_as_ISO_time(self.tstart)
            log.debug("Reading pcb info")
            self.pcb_stats = read_pcb_stats(self.scandir)
            log.debug("Reading plan info")
            self.plan_info = read_plan_info(self.scandir)
            log.debug("Reading flagging stats")
            self.rfi_info = read_rfi_stats_info(self.scandir)

            if runcandpipe:
                log.debug("Starting candpipe execution")
                self.run_candpipe()
            
            log.debug("Reading candidate files")
            self.cands_manager = ScanCandsManager(self.sbid, self.scanid, self.tstart, runname=self.runname, ignore_empty=True)

            self._dict['raw_pcb_info_dict'] = self.pcb_stats
            self._dict['raw_plan_info_dict'] = self.plan_info
            self._dict['raw_rfi_info_dict'] = self.rfi_info

            self.filter_info()
        except ReadInfoException as re:
            msg = f"Could not generate useful info due to error:\n{re.my_message}\n"
            msg+= f"Exception info:{re.exc}\n"
            msg+= f"{traceback.format_exc()}\n"
        except Exception as e:
            msg = f"Could not generate useful info due to error:\n{e}\n{traceback.format_exc()}"
        else:
            try:
                msg = self.gen_slack_msg()
            except Exception as e:
                msg = f"Could not create message from filtered info due to :\n{e}\n{traceback.format_exc()}"
        finally:
            msg = f"End of scan: {self.sbid}/{self.scanid}/{self.tstart}, runname={self.runname}\n" + msg
            self.post_on_slack(msg)
            
            outname = self.dump_json()
            try:
                log.info('Posting %s to elasticsearch', outname)
                self.post_to_elastic(outname)
            except Exception as e:
                log.exception(f"Exception posting %s to elasticsearch due to error", outname)

    def dump_json(self):
        outname = os.path.join(self.scandir.scan_head_dir, "scan_summary.json")
        
        with open(outname, 'w') as fp:
            json.dump(self._dict, fp, sort_keys=True, indent=4, cls=TrivialEncoder)

        return outname


    def post_to_elastic(self, json_path):
        '''
        Post the given file name to elasticsearc URL
        Uses the elasticsearch library to post the file to the elasticsearch URL
        Uses teh CRACO_ELASTIC_URL environment variable as the URL
        And ~/.config/craco_elastic_ca.crt as the certificate
        The ID is the SBID/SCANID/TSTART
        '''
        from elasticsearch import Elasticsearch
        url = os.environ['CRAFT_ELASTIC_URL']        

        es = Elasticsearch(
            url,
            ca_certs=os.path.expanduser('~/.config/craco_elastic_ca.crt'),
            verify_certs=True,
        )
        docid = self._dict["docid"] 
        es.index(
            id=docid,
            document=json.load(open(json_path)),
            index="cracoscans"            
        )


    def run_candpipe(self):
        '''
        YM: implement calling of the candpipe from this function       
        from craco.cadpipe import Candpipe, get_parser() etc
        '''
        from craco.candpipe import candpipe

        for scandir in self.scandir.scan_data_dirs:
            candout_dir = os.path.join(scandir, 'results/clustering_output')
            rundir = os.path.join(scandir, 'results')
            cand_fnames = glob.glob(os.path.join(rundir, 'candidates.*.txt'))
            log.debug('candidate output dir %s', candout_dir)
            os.makedirs(candout_dir, exist_ok=True)

            candpipe_args = candpipe.get_parser().parse_args(['-s', '--save-rfi', '-o', candout_dir])
            config = candpipe.load_default_config()
            log.debug('candpipe_args %s', candpipe_args)

            for cand_fname in cand_fnames:
                try:
                    log.info('run candpipe for %s', cand_fname)
                    pipe = candpipe.Pipeline(cand_fname, candpipe_args, config, src_dir=rundir, anti_alias=True)
                    pipe.run()
                except:
                    log.error(traceback.format_exc())
                    log.error(f"failed to run candpipe on {cand_fname}... aborted...")


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
    
    def _form_unknown_cand_str(self, cands, beamid):
        try:
            snrs = cands['snr']
            snr_key = 'snr'
        except KeyError as ke:
            snr_key = "SNR"
            #snrs = cands['SNR']

        return cands.apply(format_pandas_row_as_string, axis=1, args=[beamid, snr_key]).tolist()
        #strs = f"B{int(beamid):02g}_MJD{cands['mjd']}_DM{cands['dm_pccm3']:.2f}_SNR{snrs:.1f}_BOX{cands['boxc_width']}_RA{cands['ra_deg']:.4f}_DEC{cands['dec_deg']:.3f}"

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
        
        if len(candfiles) > 0:
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
                    classified_cands_per_beam[f'beam_{candfile.beamid:0>2}'] = classified_rows['MATCH_name'].unique().tolist()
                    classified_cands += classified_rows['MATCH_name'].unique().tolist()
                else:
                    # return a list of urls for unknown candidates 
                    #classified_cands_per_beam[f'beam_{candfile.beamid:0>2}'] = self._form_url(classified_rows, candfile.beamid)
                    #classified_cands += self._form_url(classified_rows, candfile.beamid)

                    classified_cands_per_beam[f'beam_{candfile.beamid:0>2}'] = self._form_unknown_cand_str(classified_rows, candfile.beamid)
                    classified_cands += self._form_unknown_cand_str(classified_rows, candfile.beamid)
                
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
        #candidates_info['num_classified_cands_per_beam'] was not being used anywhere, so I removed it completely
        #num_classified_cands, num_classified_cands_per_beam = get_num_classified_candidates(self.cands_manager.clustered_uniq_candfiles)
        num_classified_cands = get_num_classified_candidates(self.cands_manager.clustered_uniq_candfiles)
        candidates_info['num_classified_cands'] = num_classified_cands
        #candidates_info['num_classified_cands_per_beam'] = num_classified_cands_per_beam

        # total number of bright candidates for each classification 
        #num_classified_cands_bright, num_classified_cands_bright_per_beam = get_num_classified_candidates(self.cands_manager.clustered_uniq_candfiles, snr=snr)
        num_classified_cands_bright = get_num_classified_candidates(self.cands_manager.clustered_uniq_candfiles, snr=snr)
        candidates_info['num_classified_cands_bright'] = num_classified_cands_bright
        #candidates_info['num_classified_cands_bright_per_beam'] = num_classified_cands_bright_per_beam

        '''
        # total of PSR (names) detected in obs 
        pulsar_cands, pulsar_cands_per_beam = self._get_classified_candidates(self.cands_manager.clustered_uniq_candfiles, label='PSR')
        candidates_info['pulsar_cands'] = pulsar_cands 
        candidates_info['pulsar_cands_per_beam'] = pulsar_cands_per_beam
        '''

        # total of bright PSR (names) detected in obs 
        pulsar_cands_bright, pulsar_cands_bright_per_beam = self._get_classified_candidates(self.cands_manager.clustered_uniq_candfiles, label='PSR', snr=snr)
        candidates_info['pulsar_cands_bright'] = pulsar_cands_bright 
        candidates_info['pulsar_cands_bright_per_beam'] = pulsar_cands_bright_per_beam

        '''
        # total of RACS (names) detected in obs 
        racs_cands, racs_cands_per_beam = self._get_classified_candidates(self.cands_manager.clustered_uniq_candfiles, label='RACS')
        candidates_info['racs_cands'] = racs_cands 
        candidates_info['racs_cands_per_beam'] = racs_cands_per_beam
        '''

        # total of bright RACS (names) detected in obs 
        racs_cands_bright, racs_cands_bright_per_beam = self._get_classified_candidates(self.cands_manager.clustered_uniq_candfiles, label='RACS', snr=snr)
        candidates_info['racs_cands_bright'] = racs_cands_bright 
        candidates_info['racs_cands_bright_per_beam'] = racs_cands_bright_per_beam
        
        '''
        # total of CUSTOM sources (names)
        custom_cands, custom_cands_per_beam = self._get_classified_candidates(self.cands_manager.clustered_uniq_candfiles, label='CUSTOM')
        candidates_info['custom_cands'] = custom_cands 
        candidates_info['custom_cands_per_beam'] = custom_cands_per_beam
        '''

        # total of bright CUSTOM sources (names)
        custom_cands_bright, custom_cands_bright_per_beam = self._get_classified_candidates(self.cands_manager.clustered_uniq_candfiles, label='CUSTOM', snr=snr)
        candidates_info['custom_cands_bright'] = custom_cands_bright
        candidates_info['custom_cands_bright_per_beam'] = custom_cands_bright_per_beam

        '''
        # total of UNKNOWN sources (urls)
        unknown_cands, unknown_cands_per_beam = self._get_classified_candidates(self.cands_manager.clustered_uniq_candfiles, label='UNKNOWN')
        candidates_info['unknown_cands'] = unknown_cands 
        candidates_info['unknown_cands_per_beam'] = unknown_cands_per_beam
        '''

        # total of bright UNKNOWN sources (urls)
        unknown_cands_bright, unknown_cands_bright_per_beam = self._get_classified_candidates(self.cands_manager.clustered_uniq_candfiles, label='UNKNOWN', snr=snr)
        candidates_info['unknown_cands_bright'] = unknown_cands_bright
        candidates_info['unknown_cands_bright_per_beam'] = unknown_cands_bright_per_beam
            
        #self._dict['candidates_info'] = candidates_info
        return candidates_info


    def plot_candidates(self):
        '''
        Make a plot of all raw candidates coloured by their classification
        X-axis - RA
        Y-axis - Dec
        '''
        pass


    def get_data_quality_stats(self):
        '''
        To be implemented by VG
        Diagnostics -
            RFI statistics
            Dropped packet statistics

        '''       
        
        data_quality_diagnostics = {}
        rfiinfo = self.rfi_info
        if rfiinfo == {}:
            return data_quality_diagnostics
        
        #dp_stats = find_beam0_min_max_values(rfiinfo, 'dropped_packets_frac')
        data_quality_diagnostics['dropped_packets_fraction_mean'] = np.mean(rfiinfo['dropped_packets_frac'])
        data_quality_diagnostics['dropped_packets_fraction_min'] = np.min(rfiinfo['dropped_packets_frac'])
        data_quality_diagnostics['dropped_packets_fraction_max'] = np.max(rfiinfo['dropped_packets_frac'])

        #blg_stats = find_beam0_min_max_values(rfiinfo, 'frac_bls_good')
        data_quality_diagnostics['good_baselines_fraction_mean'] = np.mean(rfiinfo['frac_bls_good'])
        data_quality_diagnostics['good_baselines_fraction_min'] =  np.min(rfiinfo['frac_bls_good'])
        data_quality_diagnostics['good_baselines_fraction_max'] =  np.max(rfiinfo['frac_bls_good'])

        #blb_stats = find_beam0_min_max_values(rfiinfo, 'frac_bls_bad')
        data_quality_diagnostics['bad_baselines_fraction_mean'] = np.mean(rfiinfo['frac_bls_bad'])
        data_quality_diagnostics['bad_baselines_fraction_min'] = np.min(rfiinfo['frac_bls_bad'])
        data_quality_diagnostics['bad_baselines_fraction_max'] = np.max(rfiinfo['frac_bls_bad'])

        #fc_stats = find_beam0_min_max_values(rfiinfo, 'frac_fixed_good_chans')
        data_quality_diagnostics['good_channels_fraction_mean'] = np.mean(rfiinfo['frac_fixed_good_chans'])
        data_quality_diagnostics['good_channels_fraction_min'] = np.min(rfiinfo['frac_fixed_good_chans'])
        data_quality_diagnostics['good_channels_fraction_max'] = np.max(rfiinfo['frac_fixed_good_chans'])

        #fcb_stats = find_beam0_min_max_values(rfiinfo, 'frac_fixed_bad_chans')
        data_quality_diagnostics['bad_channels_fraction_mean'] = np.mean(rfiinfo['frac_fixed_bad_chans'])
        data_quality_diagnostics['bad_channels_fraction_min'] = np.min(rfiinfo['frac_fixed_bad_chans'])
        data_quality_diagnostics['bad_channels_fraction_max'] = np.max(rfiinfo['frac_fixed_bad_chans'])

        #fcp_stats = find_beam0_min_max_values(rfiinfo, 'frac_good_cells_pre_flagging')
        data_quality_diagnostics['good_cells_fraction_pre_rfi_flagging_mean'] = np.mean(rfiinfo['frac_good_cells_pre_flagging'])
        data_quality_diagnostics['good_cells_fraction_pre_rfi_flagging_min'] = np.min(rfiinfo['frac_good_cells_pre_flagging'])
        data_quality_diagnostics['good_cells_fraction_pre_rfi_flagging_max'] = np.max(rfiinfo['frac_good_cells_pre_flagging'])

        #fcpo_stats = find_beam0_min_max_values(rfiinfo, 'frac_good_cells_post_flagging')
        data_quality_diagnostics['good_cells_fraction_post_rfi_flagging_mean'] = np.mean(rfiinfo['frac_good_cells_post_flagging'])
        data_quality_diagnostics['good_cells_fraction_post_rfi_flagging_min'] = np.min(rfiinfo['frac_good_cells_post_flagging'])
        data_quality_diagnostics['good_cells_fraction_post_rfi_flagging_max'] = np.max(rfiinfo['frac_good_cells_post_flagging'])

        #ff_stats = find_beam0_min_max_values(rfiinfo, 'flagging_frac')
        data_quality_diagnostics['rfi_dynamic_flagging_fraction_mean'] = np.mean(rfiinfo['flagging_frac'])
        data_quality_diagnostics['rfi_dynamic_flagging_fraction_min'] = np.min(rfiinfo['flagging_frac'])
        data_quality_diagnostics['rfi_dynamic_flagging_fraction_max'] = np.max(rfiinfo['flagging_frac'])

        return data_quality_diagnostics


    def get_search_pipeline_params(self):
        '''
        To be implemented by Keith And VG 
        
        '''
        if self.plan_info == {} or self.pcb_stats == {}:
            return {}
        
        planinfo = self.plan_info
        search_params = {}
        search_params['num_dm_trials'] = planinfo['values']['ndm']
        search_params['dm_samps_min'] = 0
        search_params['hw_threshold'] = planinfo['values']['threshold']
        search_params['trigger_threshold'] = planinfo['values']['trigger_threshold']
        search_params['dm_samps_max'] = planinfo['values']['ndm'] - 1
        search_params['dm_trial_steps'] = 'linear'
        search_params['dm_trial_spacing'] = 1
        search_params['num_boxcar_width_trials'] = planinfo['values']['nbox']
        search_params['boxcar_width_samps_min'] = 1
        search_params['boxcar_width_samps_max'] = planinfo['values']['nbox']
        search_params['boxcar_trial_steps'] = 'linear'
        search_params['boxcar_trial_spacing'] = 1
        search_params['num_antennas'] = planinfo['nant']
        search_params['ants_used'] = planinfo['ants']
        search_params['num_baselines'] = planinfo['nbl']
        search_params['num_beams_planned'] = len(planinfo['values']['search_beams'])
        #tobs = find_beam0_min_max_values(self.pcb_stats, 'tobs')
        #search_params['num_beams_actual'] = sum(self.pcb_stats[beamid]['tobs'] for beamid in self.pcb_stats if beamid.startswith('beam_')) / tobs[2]

        #npix_b0_min_max = find_beam0_min_max_values(planinfo, 'npix')
        search_params['num_spatial_pixels_mean'] = np.mean(planinfo['wcs']['npix'])
        search_params['num_spatial_pixels_min'] = np.min(planinfo['wcs']['npix'])
        search_params['num_spatial_pixels_max'] = np.max(planinfo['wcs']['npix'])

        #fov1_b0_min_max = find_beam0_min_max_values(planinfo, 'fov1_deg')
        search_params['fov1_deg_mean'] = np.mean(planinfo['wcs']['fov1_deg'])
        search_params['fov1_deg_min'] = np.min(planinfo['wcs']['fov1_deg'])
        search_params['fov1_deg_max'] = np.max(planinfo['wcs']['fov1_deg'])

        #fov2_b0_min_max = find_beam0_min_max_values(planinfo, 'fov2_deg')
        search_params['fov2_deg_mean'] = np.mean(planinfo['wcs']['fov2_deg'])
        search_params['fov2_deg_min'] = np.min(planinfo['wcs']['fov2_deg'])
        search_params['fov2_deg_max'] = np.max(planinfo['wcs']['fov2_deg'])

        #cellsize1_b0_min_max = find_beam0_min_max_values(planinfo, 'cellsize1_deg')
        search_params['cellsize1_deg_mean'] = np.mean(planinfo['wcs']['cellsize1_deg'])
        search_params['cellsize1_deg_min'] = np.min(planinfo['wcs']['cellsize1_deg'])
        search_params['cellsize1_deg_max'] = np.max(planinfo['wcs']['cellsize1_deg'])

        #cellsize2_b0_min_max = find_beam0_min_max_values(planinfo, 'cellsize2_deg')
        search_params['cellsize2_deg_mean'] = np.mean(planinfo['wcs']['cellsize2_deg'])
        search_params['cellsize2_deg_min'] = np.min(planinfo['wcs']['cellsize2_deg'])
        search_params['cellsize2_deg_max'] = np.max(planinfo['wcs']['cellsize2_deg'])
    
        search_params['calibration_file'] = planinfo['values']['calibration']
        search_params['calibration_age_days'] = self.get_time_delay(planinfo['values']['calibration'])

        return search_params
    
    def get_time_delay(self, calpath):
        '''
        Extract the tstart of the observation used to generate a cal soln, and compare it with the tstart of self
        Return the time difference in days
        '''
        from datetime import datetime as DT
        timeformat="%Y%m%d%H%M%S"
        cal_time = DT.strptime(os.path.realpath(os.path.join(calpath, "00", "b00.uvfits")).strip().split("/")[-2], timeformat)
        obs_time = DT.strptime(self.tstart, timeformat)
        diff = (obs_time - cal_time).total_seconds() / 86400     #days
        return diff


    def get_scan_info(self):
        '''
        SBID related info -
            SBID
            Scan ID
            Tstart        
        '''
        scan_info = {}
        scan_info['sbid'] = self.sbid
        scan_info['scanid'] = self.scanid
        scan_info['tstart'] = self.tstart
        if self.plan_info == {}:
            scan_info['target'] = null_value
        else:
            scan_info['target'] = self.plan_info['target'][0]

        return scan_info

    def get_observation_params(self):
        '''
        Observation params -
                Beam footprint
                Central freq
                Bandwidth
                Time resolution
                Number of channels
                RA, DEC of beam0
                Alt, Az of beam0
                Gl, Gb of beam0
                Coordinates of sun
                
        '''
        if self.pcb_stats == {} or self.plan_info == {}:
            return {}
        obs_params = {}
        obs_params['beam_footprint'] = "TO BE IMPLEMENTED"
        obs_params['central_freq_MHz'] = self.pcb_stats['fcen'][0]
        obs_params['bandwidth_MHz'] = self.pcb_stats['BW'][0]
        obs_params['num_channels'] = self.plan_info['freq_info']['nchan']
        obs_params['sampling_time_ms'] = self.plan_info['tsamp_s'] * 1e3
        obs_params['guest_science_proposal'] = "TO BE IMPLEMENTED"
        
        #tobs = find_beam0_min_max_values(self.pcb_stats, 'tobs')
        obs_params['tobs_mean'] = np.mean(self.pcb_stats['tobs'])
        obs_params['tobs_min'] = np.min(self.pcb_stats['tobs'])
        obs_params['tobs_max'] = np.max(self.pcb_stats['tobs'])
        obs_params['tobs_sum'] = np.sum(self.pcb_stats['tobs'])

        #sol = find_beam0_min_max_values(self.plan_info, 'solar_elong_deg')
        obs_params['solar_elong_deg_mean'] = np.mean(self.plan_info['solar_elong_deg'])
        obs_params['solar_elong_deg_min'] = np.min(self.plan_info['solar_elong_deg'])
        obs_params['solar_elong_deg_max'] = np.max(self.plan_info['solar_elong_deg'])

        #lun = find_beam0_min_max_values(self.plan_info, 'lunar_elong_deg')
        obs_params['lunar_elong_deg_mean'] = np.mean(self.plan_info['lunar_elong_deg'])
        obs_params['lunar_elong_deg_min'] = np.min(self.plan_info['lunar_elong_deg'])
        obs_params['lunar_elong_deg_max'] = np.max(self.plan_info['lunar_elong_deg'])

        obs_params['coords_ra_deg_beam00'] = self.plan_info['wcs']['coords_ra_deg'][0]      #not gauranteed to be beam00 - just the first beam in the list. Should be beam 0 if beam 0 was searched
        obs_params['coords_dec_deg_beam00'] = self.plan_info['wcs']['coords_dec_deg'][0]
        obs_params['coords_ra_hms_beam00'] = self.plan_info['wcs']['coords_ra_hms'][0]
        obs_params['coords_dec_dms_beam00'] = self.plan_info['wcs']['coords_dec_dms'][0]

        obs_params['coords_gl_deg_beam00'] = self.plan_info['wcs']['gl_deg'][0]
        obs_params['coords_gb_deg_beam00'] = self.plan_info['wcs']['gb_deg'][0]

        obs_params['coords_az_deg_beam00'] = self.plan_info['wcs']['az_deg'][0]
        obs_params['coords_alt_deg_beam00'] = self.plan_info['wcs']['alt_deg'][0]

        return obs_params
        
    def gen_slack_msg(self):
        '''
        Compose a nice slack message using the self._dict and the plot
                
        '''
        msg = f"End of scan: {self.sbid}/{self.scanid}/{self.tstart}, runname={self.runname}\n"
        msg += f"Scan head dir: {self.scandir.scan_head_dir}\n"
        
        msg += "----------------\n"
        msg += "Scan info -> \n"
        if self.filtered_scan_info != {}:
            msg += f"- Target [Beam 0]: {self.filtered_scan_info['target']}\n"
        
        msg += "----------------\n"
        msg += "Obs info ->\n"
        if self.filtered_obs_info != {}:
            msg += f"- Duration [max (min,mean,sum)]: {self.filtered_obs_info['tobs_max']:.2f} ({self.filtered_obs_info['tobs_min']:.2f}, {self.filtered_obs_info['tobs_mean']:.2f}, {self.filtered_obs_info['tobs_sum']:.2f})\n"
            msg += f"- Central freq: {self.filtered_obs_info['central_freq_MHz']:.1f} MHz\n"
            msg += f"- Bandwidth: {self.filtered_obs_info['bandwidth_MHz']:.2f} MHz\n"
            msg += f"- Sampling time: {self.filtered_obs_info['sampling_time_ms']:.3f} ms\n"
            
        msg += "----------------\n"
        if self.filtered_search_info != {}:
            msg += "Search info ->\n"
            msg += f"- Nant: {self.filtered_search_info['num_antennas']}\n"
            msg += f"- Nbl: {self.filtered_search_info['num_baselines']}\n"
            msg += f"- Nchan: {self.filtered_obs_info['num_channels']}\n"
            msg += f"- Ndm (min-max): {self.filtered_search_info['num_dm_trials']} ({self.filtered_search_info['dm_samps_min']} - {self.filtered_search_info['dm_samps_max']}) samples\n"
            msg += f"- Nboxcar (min-max): {self.filtered_search_info['num_boxcar_width_trials']} ({self.filtered_search_info['boxcar_width_samps_min']} - {self.filtered_search_info['boxcar_width_samps_max']}) samples\n"
            msg += f"- Threshold (trigger): {self.filtered_search_info['trigger_threshold']}\n"
            msg += f"- Threshold (HW): {self.filtered_search_info['hw_threshold']}\n"

        msg += "----------------\n"
        msg += "Candidate info -> \n"
        if self.filtered_cands_info != {}:
            msg += f"- Num raw (incl. subthreshold): {self.filtered_cands_info['num_raw_cands_bright']} ({self.filtered_cands_info['num_raw_cands']})\n"
            msg += f"- Num raw clustered (incl. subthreshold): {self.filtered_cands_info['num_clustered_cands_bright']} ({self.filtered_cands_info['num_clustered_cands']})\n"

            msg += f"- Num RFI (incl. subthreshold): {self.filtered_cands_info['num_clustered_rfi_cands_bright']}\n"# ({self.filtered_cands_info['num_clustered_rfi_cands']})\n"
            msg += f"- Num localised (incl. subthreshold): {self.filtered_cands_info['num_clustered_uniq_cands_bright']}\n"# ({self.filtered_cands_info['num_clustered_uniq_cands']})\n"
            msg += f"- Num source types detected (incl. subthreshold): {self.filtered_cands_info['num_classified_cands_bright']}\n"#   ({self.filtered_cands_info['num_classified_cands']})\n"
            msg += f"- List of pulsars detected: {self.filtered_cands_info['pulsar_cands_bright']}\n"
            msg += f"- List of RACS sources detected: {self.filtered_cands_info['racs_cands_bright']}\n"
            msg += f"- List of custom catalog sources detected: {self.filtered_cands_info['custom_cands_bright']}\n"
            #msg += f"- List of unknwon sources detected (URLs): {convert_urls_to_readable_links(self.filtered_cands_info['unknown_cands_bright'])}\n"
        
        msg += "----------------\n"
        msg += "Data quality info ->\n"
        if self.filtered_dq_info != {}:
            msg += f"- Dropped packets + DM0 flag fraction avg [Beam 0 (min-max)]: {self.filtered_dq_info['dropped_packets_fraction_mean']:.2f} ({self.filtered_dq_info['dropped_packets_fraction_min']:.2f} - {self.filtered_dq_info['dropped_packets_fraction_max']:.2f})\n"
            msg += f"- Static freq flag fraction: {self.filtered_dq_info['bad_channels_fraction_mean']:.2f}\n"
            msg += f"- Flagged baselines fraction [Beam 0 (min-max)]: {self.filtered_dq_info['bad_baselines_fraction_mean']:.2f} ({self.filtered_dq_info['bad_baselines_fraction_min']:.2f} - {self.filtered_dq_info['bad_baselines_fraction_max']:.2f})\n"
            msg += f"- Dynamic rfi flagging fraction [Beam 0 (min-max)]: {self.filtered_dq_info['rfi_dynamic_flagging_fraction_mean']:.2f} ({self.filtered_dq_info['rfi_dynamic_flagging_fraction_min']:.2f} - {self.filtered_dq_info['rfi_dynamic_flagging_fraction_max']:.2f})\n"

        return msg
    
    def post_on_slack(self, msg):
        if self.block_slack_post:
            return
        log.debug(f"Posting message - \n{msg}")
        slack_poster = SlackPostManager(test=False, channel="C06FCTQ6078")
        slack_poster.post_message(msg)

    def filter_info(self):
        log.debug("Filtering relevant information")
        self.filtered_scan_info = self.get_scan_info()
        self.filtered_obs_info = self.get_observation_params()
        self.filtered_cands_info = self.get_candidates_info()
        self.filtered_dq_info = self.get_data_quality_stats()
        self.filtered_search_info = self.get_search_pipeline_params()

        self._dict['filtered_scan_info_dict'] = self.filtered_scan_info
        self._dict['filtered_obs_info_dict'] = self.filtered_obs_info
        self._dict['filtered_cands_info_dict'] = self.filtered_cands_info
        self._dict['filtered_dq_info_dict'] = self.filtered_dq_info
        self._dict['filtered_search_info_dict'] = self.filtered_search_info

def _main():
    args = get_parser()
    obsinfo = ObsInfo(sbid = args.sbid,
                      scanid = args.scanid,
                      tstart = args.tstart,
                      runcandpipe = args.run_candpipe,
                      block_slack_post = args.block_slack_post)
    #obsinfo.run()

def get_parser():
    a = argparse.ArgumentParser()
    a.add_argument("-sbid", type=str, help="SBID", required=True)
    a.add_argument("-scanid", type=str, help="scanid", required=True)
    a.add_argument("-tstart", type=str, help="tstart", required=True)
    a.add_argument("-no_candpipe", dest='run_candpipe', action='store_false', help="Don't run candpipe (def:False)", default=True)
    a.add_argument('-no_slack', dest='block_slack_post', action='store_true', help="Don't post a message on slack (def: False)", default = False)

    args = a.parse_args()
    return args

if __name__ == '__main__':
    _main()
