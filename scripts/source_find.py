#!/usr/bin/env python
import bdsf

def run_bdsf(fitsfname):
    img = bdsf.process_image(fitsfname)
    gausscat_fname = fitsfname.replace(".fits", ".gauss.cat.fits")
    img.write_catalog(
        outfile = gausscat_fname, clobber = True,
        format = "fits", catalog_type = "gaul",
    )

def main():
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
    parser = ArgumentParser(description="running pybdsf on fits image", formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("-fits", type=str, help="path to the fits file")

    args = parser.parse_args()

    run_bdsf(args.fits)

if __name__ == "__main__":
    main()
