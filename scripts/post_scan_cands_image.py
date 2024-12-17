#!/usr/bin/env python
from craco.craco_run.slackpost import RealTimeScanAlarm

CAND_POST_TS_SOCKET = "/data/craco/craco/tmpdir/queues/cands"
TMPDIR = "/data/craco/craco/tmpdir"

def run_cand_post(outdir, test=False):
    if test: channel = "C06C6D3V03S"
    else: channel = "C085V0B3D7A"

    scanalarm = RealTimeScanAlarm(outdir, channel=channel, postlimit=10)
    scanalarm.run_main()

def main():
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
    parser = ArgumentParser(description="running candidates posting for a given outdir", formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("-outdir", type=str, help="path to the ourdir")

    args = parser.parse_args()

    run_cand_post(args.outdir)

if __name__ == "__main__":
    main()

