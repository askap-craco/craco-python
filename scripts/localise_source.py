#!/usr/bin/env python
from craco.craco_run import localiser

### call localiser.py to localise source...
def main():
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
    parser = ArgumentParser(description="get burst localisation from fits file", formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("--fieldfpath", type=str, help="path to the field fits file", required=True)
    parser.add_argument("--burstfpath", type=str, help="path to the burst fits file", required=True)
    parser.add_argument("--fieldcpath", type=str, help="path to the field cat file", default="")
    parser.add_argument("--burstcpath", type=str, help="path to the burst cat file", default="")
    parser.add_argument("--ra", type=float, help="source ra in degree", required=True)
    parser.add_argument("--dec", type=float, help="source dec in degree", required=True)
    parser.add_argument("--summary", type=str, help="source summary that will appear in the end", default="")
    parser.add_argument("--workdir", type=str, help="working directory", default="./")

    args = parser.parse_args()

    if args.fieldcpath == "": fieldcpath = args.fieldfpath.replace(".fits", ".gauss.cat.fits")
    else: fieldcpath = args.fieldcpath

    if args.burstcpath == "": burstcpath = args.burstfpath.replace(".fits", ".gauss.cat.fits")
    else: burstcpath = args.burstcpath

    srcposcorr = localiser.CorrSrcPos(
        fieldfitspath=args.fieldfpath,
        burstfitspath=args.burstfpath,
        fieldcatpath = fieldcpath,
        burstcatpath = burstcpath,
        srcra = args.ra, srcdec = args.dec,
        workdir = args.workdir,
        summary = args.summary,

    )

    srcposcorr.correct_src_coord()
    srcposcorr.plot_burst_field()

if __name__ == "__main__":
    main()


