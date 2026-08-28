#!/usr/bin/env python3

from casatasks import importuvfits

def run(infits, outms):
    importuvfits(
        fitsfile=infits,
        vis=outms
    )

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Convert UVFITS to Measurement Set")
    parser.add_argument("--infits", help="Input UVFITS file")
    args = parser.parse_args()

    infits = args.infits
    outms = infits.rstrip(".uvfits") + ".ms"

    run(infits, outms)