#!/usr/bin/env python

from craco.casda_archiver import ArchiveManager
from craco.mattermost_messager import MattermostPostManager

import argparse

def get_parser():
    a = argparse.ArgumentParser()
    a.add_argument("-sbid", type=str, help="SBID", required=True)
    a.add_argument("-scanid", type=str, help="scanid", required=True)
    a.add_argument("-tstart", type=str, help="tstart", required=True)

    args = a.parse_args()
    return args

def main():
    args = get_parser()

    scan = f"{args.scanid}/{args.tstart}"

    mm = MattermostPostManager()

    try:
        am = ArchiveManager()
        dbstatus = am.insert_record(sbid=args.sbid, scan = scan)
        mm.post_message(f"SBID {args.sbid} Scan {scan} has been added to the archive database with status {dbstatus}")
    except Exception as e:
        mm.post_message(f"SBID {args.sbid} Scan {scan} failed to be added to the archive database with the following error:\n{e}")    

if __name__ == "__main__":
    main()

