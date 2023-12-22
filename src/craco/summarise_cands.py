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

def _main():
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
    parser = ArgumentParser(description='Script description', formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('-v', '--verbose', action='store_true', help='Be verbose')
    parser.add_argument(dest='files', nargs='+')
    parser.set_defaults(verbose=False)
    values = parser.parse_args()
    if values.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)



    all_df = []

    def beam_of(f):
        beamno = int(os.path.basename(f).split('.')[1][1:])
        return beamno

    files = sorted(values.files, key=beam_of)

    for ibeam, f in enumerate(files):
        assert beam_of(f) == ibeam
        df = pd.read_csv(f)
        print(ibeam, f)
        df['beamno'] = ibeam
        all_df.append(df)

    df = pd.concat(all_df)

    summary = df.groupby('beamno').count()
    links = [f'*<http://localhost:8024/beam?fname={fname}| Beam{ibeam:02d}>*\n U=0 N=0 RACS=0' for ibeam, fname in enumerate(files)]
    summary['link'] = links

    columns = ['NEW_name','NEW_sep','link']

    tab = summary[columns].to_markdown()

    msgs = [str(r['NEW_name']) + ' ' + r['link'] for _, r in summary.iterrows()]
    blocks1 = [{'type':'section','text':{'type':'mrkdwn','text':t}} for t in msgs]
    fields = [{'type':'mrkdwn','text':t} for t in msgs]
    scanname = '/'.join(f.split('/')[4:8])

    blocks = [
        {
            'type':'section',
            'text':f'Result of processing scan {scanname}'
        },
        {
            'type':'section',
            'fields':fields
        }
    ]


    token = os.environ["SLACK_CRACO_TOKEN"]
    client = WebClient(token=token)
    channel = 'C05Q11P9GRH'
    client.chat_postMessage(channel=channel, blocks=blocks1)
    
    

if __name__ == '__main__':
    _main()
