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
#from craco.craco_run.auto_sched import SlackPostManager
import glob
import numpy as np

log = logging.getLogger(__name__)

__author__ = "Keith Bannister <keith.bannister@csiro.au>"

def format_msg(r):
    msg = 'UNKNOWN=' + str(r['Unknown']) + ' ' \
        'PSR=' + str(r['PSR_name']) + ' ' \
        'RACS=' + str(r['RACS_name']) + ' ' \
        'CRACO=' + str(r['NEW_name']) + ' ' \
        'ALIAS=' + str(r['ALIAS_name']) + ' ' \
        + r['link']
    
    if r["Unknown"] > 0: return f"*{msg}* \n"
    return f"{msg} \n"

def beam_of(f):
    beamno = int(os.path.basename(f).split('.')[1][1:])
    return beamno


def read_file(filename, snr=8, cet_remove=False):

    f = pd.read_csv(filename, index_col=0)
    # f = f[ (f['dm'] < 150) ]
    f = f[ f['SNR'] >= snr ]

    if cet_remove:
        # remove all central ghost things
        f = f[ ~( (f['lpix'] >= 126) & (f['mpix'] >= 126) & (f['lpix'] <= 130) & (f['lpix'] <= 130) ) ]

    return f


def read_filterbank_stats(filpath):
    try:
        f = SF(filpath)
        dur = f.nsamples * f.tsamp / 60         #minutes
        bw = np.abs(f.foff) * f.nchans
        fcen = f.fch1 + bw / 2
        ra = f.src_raj_deg
        dec = f.src_dej_deg

        msg = f"Duration: {dur:.1f} min\nBW: {bw:.1f} MHz\nFcen: {fcen:.1f} MHz\nBeam0 coords: {ra},{dec}\n"
    except:
        msg = f"!Error: Could not read filterbank information from path - {filpath}!"
    finally:
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


    scan = ScanDir(sbid = values.sbid, scan = f"{values.scanid}/{values.tstart}")


    for inode, node_dir in enumerate(scan.scan_data_dirs):
        # assert beam_of(f) == ibeam

        unclustered_candidate_files = glob.glob(node_dir + "/results/candidates.b*.txt")
        uniq_candidate_files = glob.glob(node_dir + "/results/clustering_output/candidates.b*.uniq.csv")

        filpath = os.path.join(node_dir, "pcbb00.fil")
        if os.path.exists(filpath):
            search_dur_message = read_filterbank_stats(filpath)

        for uniq_file in uniq_candidate_files:
            df = pd.read_csv(uniq_file, index_col=0)
            num_uniq_cands += len(df)

            # snr >= 9
            snr = values.snr
            df = df[ df['snr'] >= snr ]

            df['Unknown'] = df['PSR_name'].isna() & df['RACS_name'].isna() & df['NEW_name'].isna() & df['ALIAS_name'].isna()
    
            num_unknown_cands += len(df[df['Unknown']])

        for raw_file in unclustered_candidate_files:
            nlines = -1
            with open(raw_file, 'r') as f:
                for _  in f:
                    nlines += 1
            raw_num += nlines

    msg = f"""Finished processing scan::\n{values.sbid}/{values.scanid}/{values.tstart}\n{search_dur_message}\nTotal raw cands: {raw_num}\nTotal unknown cands:{num_unknown_cands}\n"""
    print(msg)
    #slack_poster = SlackPostManager(test=False, channel="C05Q11P9GRH")
    #slack_poster.post_message(msg)


if __name__ == '__main__':
    _main()
