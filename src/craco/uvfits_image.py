from craco import uvfits_meta
import argparse
import numpy as np


def get_parser():
    a = argparse.ArgumentParser()
    a.add_argument("uvpath", type=str, help="Path to the uvfits file to extract from")
    a.add_argument("-metadata", type=str, help="Path to the metadata file")
    a.add_argument("-outname", type=str, help="Name of the output file", default=None)

    args = a.parse_args()
    return args

def main():
    args = get_parser()
    #-------------------- Write your code here Andy ------------------



    #-----------------------------------------------------------------




if __name__ == '__main__':
    main()


