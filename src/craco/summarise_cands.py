#!/usr/bin/env python
"""
Summarise sttistics fomr andidates and send to slcak

Copyright (C) CSIRO 2022
"""
import pylab
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import logging
import pandas as pd
import json
from IPython import embed
from slack_sdk import WebClient

log = logging.getLogger(__name__)

__author__ = "Keith Bannister <keith.bannister@csiro.au>"

def format_msg(r):
    msg = 'UNKNOWN=' + str(r['Unknown']) + ' ' \
        'PSR=' + str(r['PSR_name']) + ' ' \
        'RACS=' + str(r['RACS_name']) + ' ' \
        'CRACO=' + str(r['NEW_name']) + ' ' \
        'ALIAS=' + str(r['ALIAS_name']) + ' ' \
        + r['link']
    
    if r["Unknown"] >= 0: return f"*{msg}* \n"
    return f"{msg} \n"

def beam_of(f):
    beamno = int(os.path.basename(f).split('.')[1][1:])
    return beamno

def _main():
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
    parser = ArgumentParser(description='Script description', formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('-v', '--verbose', action='store_true', help='Be verbose')
    parser.add_argument('-o', '--output', type=str, default=None, help="output concat file for all beams")
    parser.add_argument(dest='files', nargs='+')
    parser.set_defaults(verbose=False)
    values = parser.parse_args()
    if values.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)


    all_df = []
    links = []
    num_cands = 0
    raw_num = 0
    rfi_num = 0

    files = sorted(values.files, key=beam_of)

    for ibeam, f in enumerate(files):
        # assert beam_of(f) == ibeam
        beamno = beam_of(f)

        df = pd.read_csv(f, index_col=0)
        num_cands += len(df)

        # snr >= 9
        df = df[ df['SNR'] >= 9 ]

        df['Unknown'] = df['PSR_name'].isna() & df['RACS_name'].isna() & df['NEW_name'].isna() & df['ALIAS_name'].isna()
        df['beamno'] = beamno
        all_df.append(df)

        if len(df) != 0:
            links.append(f'<http://localhost:8024/beam?fname={f}| Beam{beamno:02d}>')

        rawcat = os.path.join(os.path.dirname(os.path.dirname(f)), os.path.basename(f).split('.uniq.csv')[0])
        print(beamno, f, rawcat)

        rficat = os.path.join(os.path.dirname(f), os.path.basename(f).replace('uniq', 'rfi'))
        print(rficat)

        raw_num += sum(1 for _ in open(rawcat))
        rfi_num += sum(1 for _ in open(rficat))
        print(rfi_num)


    df = pd.concat(all_df)

    rawstats = f'raw_cand={raw_num} clustered={num_cands} rfi={rfi_num} cand={len(df)}'
    print(rawstats)


    if values.output is not None:
        # unknown = df[ df['Unknown'] ]
        # unknown.to_csv(values.output, index=False)
        df.to_csv(values.output, index=False)

    summary = df.groupby('beamno').count()
    summary2 = df.groupby('beamno').sum()

    # links = [f'*<http://localhost:8024/beam?fname={fname}| Beam{ibeam:02d}>*\n ' for ibeam, fname in enumerate(files)]
    summary['link'] = links
    summary['Unknown'] = summary2['Unknown']

    print(summary)

    # columns = ['NEW_name','NEW_sep','link']
    # tab = summary[columns].to_markdown()
    # fields = [{'type':'mrkdwn','text':t} for t in msgs]


    scanname = '/'.join(f.split('/')[4:8])


    msgs = [format_msg(r) for _, r in summary.iterrows()]

    # msgs = [scanname + ' ' + '\n'] + msgs

    msgs = [scanname + ' ' + '\n'] + [rawstats + ' ' + '\n' ] + msgs

    blocks1 = [{"type": "divider"}]

    blocks1 += [
        {
            'type':'section',
            'text':{
                'type':'mrkdwn',
                'text':t
            }
        } for t in msgs
    ]

    blocks1 += [{"type": "divider"}]

    token = os.environ["SLACK_CRACO_TOKEN"]
    client = WebClient(token=token)
    channel = 'C05Q11P9GRH'
    client.chat_postMessage(channel=channel, blocks=blocks1)

if __name__ == '__main__':
    _main()
