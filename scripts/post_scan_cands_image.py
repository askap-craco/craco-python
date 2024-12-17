#!/usr/bin/env python
from craco.craco_run.slackpost import RealTimeScanAlarm

CAND_POST_TS_SOCKET = "/data/craco/craco/tmpdir/queues/cands"
TMPDIR = "/data/craco/craco/tmpdir"

def format_outdir(outdir):
    parts = outdir.strip().split("/")
    if len(parts) > 0:
        for ip, part in enumerate(parts):
            if part.startswith("SB0"):
                sbid = part
                scanid = parts[ip + 2]
                tstart = parts[ip + 3]
                
                if len(sbid) == 8 and len(scanid) == 2 and len(tstart) == 14:
                    return f"{sbid}/scans/{scanid}/{tstart}"

    raise RuntimeError(f"Could not parse sbid, scanid and tstart from {outdir}")

def run_cand_post(outdir, test=False):
    if test: channel = "C06C6D3V03S"
    else: channel = "C085V0B3D7A"

    ### we need the ourdir to be SB0XXXX/scans/00/20240101100000
    outdir = format_outdir(outdir)

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

