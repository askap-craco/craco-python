#!/usr/bin/env python
"""
Summarise sttistics fomr andidates and send to slcak

Copyright (C) CSIRO 2022
"""
import os
import sys
import logging
import pandas as pd
from craft.sigproc import SigprocFile as SF
from craco.datadirs import ScanDir
from craco.craco_run.auto_sched import SlackPostManager
from craco.metadatafile import MetadataFile as MF
import glob
import numpy as np
import subprocess
import logging 

log = logging.getLogger(__name__)


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
        coord_string = f"{ra_to_hms(ra)}, {dec_to_dms(dec)} ({ra:.4f}, {dec:.4f})"

        msg = f"- Duration: {dur:.1f} min\n- BW: {bw:.1f} MHz\n- Fcen: {fcen:.1f} MHz\n- Beam0 coords: {coord_string}\n"
    except:
        msg = f"!Error: Could not read filterbank information from path - {filpath}!"
    finally:
        return msg

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
    log.info(f"queuing up summarise cands")
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
        cmd = f"""summarise_cands -sbid {sbid} -scanid {scanid} -tstart {tstart}"""

        subprocess.run(
            [f"tsp {cmd}"], shell=True, capture_output=True,
            text=True, env=ecopy,
        )
        log.info(f"Queued summarise cands job - with command - {cmd}")


def get_metadata_info(scan):
    metapath = os.path.join(scan.scan_head_dir, "metafile.json")
    if not os.path.exists(metapath):
        msg = f"Metafile not found at path - {metapath}"
    else:
        try:
           mf = MF(metapath)
           source_names = str(list(mf.sources(0).keys()))
           msg = f"Beam 0 source names - {source_names}\n"
        except Exception as E:
            emsg = f"Could not load the metadata info from {metapath} due to this error - {E}"
            log.critical(emsg)
            msg = emsg
    return msg



def _main():
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
    parser = ArgumentParser()
    parser.add_argument('-snr', type=float, default=9, help='SNR selection threshold for candidates checking')
    parser.add_argument('-sbid', type=str, required=True, help="SBID")
    parser.add_argument('-scanid', type=str, required=True, help="Scan ID")
    parser.add_argument('-tstart', type=str, required=True, help="Tstart")
    values = parser.parse_args()

    num_uniq_cands = 0
    num_unknown_cands = 0
    raw_num = 0

    msg = f"""End of scan::\n{values.sbid}/{values.scanid}/{values.tstart}\n"""
    try:
        scan = ScanDir(sbid = values.sbid, scan = f"{values.scanid}/{values.tstart}")
    except ValueError as VE:
        msg += f"Cannot instantiate a ScanDir object with the given arguments!"
    else:
         msg += f"Scan head dir: {scan.scan_head_dir}\n"
         found_pcb = False
         metainfo = get_metadata_info(scan)
         msg += metainfo

         for inode, node_dir in enumerate(scan.scan_data_dirs):

             unclustered_candidate_files = glob.glob(node_dir + "/results/candidates.b*.txt")
             uniq_candidate_files = glob.glob(node_dir + "/results/clustering_output/candidates.b*.uniq.csv")

             filpath = os.path.join(node_dir, "pcbb00.fil")
             if os.path.exists(filpath):
                 found_pcb = True
                 search_dur_message = read_filterbank_stats(filpath)

             if len(unclustered_candidate_files) == 0:
                 msg += f"No raw candidate files found on node: {node_dir}!\n"
                 continue

             try:
                 for raw_file in unclustered_candidate_files:
                     nlines = -1
                     with open(raw_file, 'r') as f:
                         for _  in f:
                             nlines += 1
                     raw_num += nlines
             except Exception as E:
                 emsg = f"Could not read the raw candidate file - {raw_file} due the following error - \n{E}\n"
                 log.critical(emsg)
                 continue
             
             if len(uniq_candidate_files) == 0:
                 msg += f"No clustered candidate files found on node: {node_dir}!\n"
                 continue

             try:
                 for uniq_file in uniq_candidate_files:
                     df = pd.read_csv(uniq_file, index_col=0)
                     num_uniq_cands += len(df)

                     snr = values.snr
                     df = df[ df['snr'] >= snr ]

                     #df['Unknown'] = df['PSR_name'].isna() & df['RACS_name'].isna() & df['NEW_name'].isna() & df['ALIAS_name'].isna()
                     df ['Unknown'] = df['LABEL'] == "UNKNOWN"
         
                     num_unknown_cands += len(df[df['Unknown']])
             except Exception as E:
                 emsg = f"Could not read the uniq candidate file - {uniq_file} due to the following error - \n{E}"
                 log.critical(emsg)
                 continue

         if not found_pcb:
             if num_unknown_cands == 0 and raw_num == 0:
                 msg += "We don't have a pcb file for this scan --> didn't search!\n"
             else:
                 msg += "We don't have a pcb file, but somehow candidates exist for this scan - something went wrong, please take a look VG, Andy, Keith!!!\n"
         else:
             msg += f"""{search_dur_message}\nTotal raw cands: {raw_num}\nTotal unknown cands: {num_unknown_cands}\n"""
    finally:     
        log.info("Posting message - \n" + msg)
        slack_poster = SlackPostManager(test=False, channel="C05Q11P9GRH")
        slack_poster.post_message(msg)

if __name__ == '__main__':
    _main()
