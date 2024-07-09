#!/usr/bin/env python

from craco.craco_run.slackpost import TabAlarm

def main():
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
    parser = ArgumentParser(description="running pybdsf on fits image", formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("-workdir", type=str, help="path to the fits file")
    parser.add_argument("-channel", type=str, help="channel id for the posting", default="C06C6D3V03S")

    args = parser.parse_args()

    tabalarm = TabAlarm(workdir=args.workdir, channel=args.channel)
    tabalarm.post_status()

if __name__ == "__main__":
    main()