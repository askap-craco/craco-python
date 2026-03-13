#!/usr/bin/env python
# scripts to convert craco calibration solution to casa calibration table
from casacore import tables
import numpy as np
import pickle

from craco import plotbp

import os
import logging
logging.basicConfig(
    level = logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)
# logger.setLevel(logging.INFO)

def load_pickle_data(fname):
    with open(fname, "rb") as fp:
        data = pickle.load(fp)
    return data

def make_tabdesc(coldesc):
    """
    make table description based on the json file
    """
    coldesc = [tables.makecoldesc(col, coldesc[col]) for col in coldesc]
    return tables.maketabdesc(coldesc)

def _work_refant(bp):
    bp = bp[0]
    nant, nchan, npol = bp.shape
    for iant in range(nant):
        antdata = bp[iant, :, (0, 3)]
        antdata_degree = np.angle(antdata, deg=True)
        if (abs(antdata_degree) > 0.1).sum() == 0:
            return iant
    raise ValueError("no reference antenna found...")
#     for iant in range(nant):
#         antdata = bp[iant, :, (0, 3)]
#         if np.isnan(antdata).sum() == 0:
#             return iant
#     raise ValueError("no reference antenna found...")

def make_caltab(calpath, bp, refant=None, datapath=None):

    _, nant, nchan, npol = bp.shape
    
    coldesc = load_pickle_data(f"{datapath}/CAL_coldesc.pkl")
    tb = make_tabdesc(coldesc)
    tabpath = "{}".format(calpath)
    tab = tables.table(tabpath, tb, nrow=nant) # 36 antennas
    
    ###
    if refant is None:
        refant = _work_refant(bp) # bp should be a 4d array
    
    # columns
    tab.putcol("TIME", np.zeros(nant, dtype=int))
    tab.putcol("FIELD_ID", np.zeros(nant, dtype=int)) # only one source
    tab.putcol("SPECTRAL_WINDOW_ID", np.zeros(nant, dtype=int)) # again, only one spectral window
    tab.putcol("ANTENNA1", np.arange(nant, dtype=int))
    tab.putcol("ANTENNA2", np.ones(nant, dtype=int) * refant)
    tab.putcol("INTERVAL", np.zeros(nant))
    tab.putcol("OBSERVATION_ID", np.zeros(nant, dtype=int))
    tab.putcol("FLAG", np.zeros((nant, nchan, 2), dtype=bool)) # assume to flag here...
    tab.putcol("SNR", np.ones((nant, nchan, 2), ) * 10.)
    tab.putcol("PARAMERR", np.ones((nant, nchan, 2)))
    
    ### work out the parameter
    tab.putcol("CPARAM", bp[..., [0, 3]][0])
    tab.close()

def make_spwtab(calpath, freqs, datapath=None):
    ### work out frequencies...
    nchan = freqs.shape[0]
    chan_width = np.mean(freqs[1:] - freqs[:-1])
    total_bandwidth = chan_width * nchan
    
    coldesc = load_pickle_data(f"{datapath}/SPW_coldesc.pkl")
    coldata = load_pickle_data(f"{datapath}/SPW_coldata.pkl")
    ### make table
    tb = make_tabdesc(coldesc)
    
    tabpath = "{}/SPECTRAL_WINDOW".format(calpath)
    tab = tables.table(tabpath, tb, nrow=1) # for 36 antenna...
    for col in coldata:
        tab.putcol(col, coldata[col])
    
    ### put rows deleted...
    tab.putcol("CHAN_FREQ", np.array([freqs]))
    tab.putcol("REF_FREQUENCY", np.array([freqs[0]]))
    tab.putcol("CHAN_WIDTH", np.ones((1, nchan)) * chan_width)
    tab.putcol("EFFECTIVE_BW", np.ones((1, nchan)) * chan_width)
    tab.putcol("RESOLUTION", np.array([freqs]))
    tab.putcol("NUM_CHAN", np.array([nchan]))
    tab.putcol("TOTAL_BANDWIDTH", np.array([total_bandwidth]))
    
    tab.close()

def make_fietab(calpath, datapath=None):
    ### load data
    coldesc = load_pickle_data(f"{datapath}/FIELD_coldesc.pkl")
    coldata = load_pickle_data(f"{datapath}/FIELD_coldata.pkl")
    ### make table
    tb = make_tabdesc(coldesc)
    
    tabpath = "{}/FIELD".format(calpath)
    tab = tables.table(tabpath, tb, nrow=1) # for 36 antenna...
    for col in coldata:
        tab.putcol(col, coldata[col])
    tab.close()

def make_histab(calpath, datapath=None):
    ### load data
    coldesc = load_pickle_data(f"{datapath}/HISTORY_coldesc.pkl")
#     coldata = load_pickle_data("./data/HISTORY_coldata.pkl")
    ### make table
    tb = make_tabdesc(coldesc)
    
    tabpath = "{}/HISTORY".format(calpath)
    tab = tables.table(tabpath, tb, nrow=0) # for 36 antenna...
    tab.close()

def make_obstab(calpath, datapath=None):
    ### load data
    coldesc = load_pickle_data(f"{datapath}/OBS_coldesc.pkl")
    coldata = load_pickle_data(f"{datapath}/OBS_coldata.pkl")
    ### make table
    tb = make_tabdesc(coldesc)
    
    tabpath = "{}/OBSERVATION".format(calpath)
    tab = tables.table(tabpath, tb, nrow=1) # for 36 antenna...
    
    ### put data in it...
    for col in coldata:
        tab.putcol(col, coldata[col])
    tab.close()

def make_anttab(calpath, datapath=None):
    ### load data
    coldesc = load_pickle_data(f"{datapath}/ANTENNA_coldesc.pkl")
    coldata = load_pickle_data(f"{datapath}/ANTENNA_coldata.pkl")
    ### make table
    tb = make_tabdesc(coldesc)
    
    tabpath = "{}/ANTENNA".format(calpath)
    tab = tables.table(tabpath, tb, nrow=36) # for 36 antenna...
    
    ### put data in it...
    for col in coldata:
        tab.putcol(col, coldata[col])
    tab.close()

def make_full_casabp(calpath, bp, freqs, datapath=None, refant=None):
    if datapath is None:
        datapath = os.path.join(os.path.dirname(__file__), "data/casa_bp_ms_data")
    logger.info(f"start making casa calibration table...")
    logger.info(f"datapath set to {datapath}...")

    make_caltab(calpath, bp, refant=refant, datapath=datapath)
    make_spwtab(calpath, freqs, datapath=datapath)
    make_fietab(calpath, datapath=datapath)
    make_histab(calpath, datapath=datapath)
    make_obstab(calpath, datapath=datapath)
    make_anttab(calpath, datapath=datapath)

    ### add keywords
    t = tables.table(calpath, readonly=False)
    t.putkeyword("ParType", "Complex")
    # t.putkeyword("MSName", "cal.B0")
    t.putkeyword("VisCal", "B Jones")
    t.putkeyword("PolBasis", "unknown")
    t.putkeyword("OBSERVATION", f"Table: {calpath}/OBSERVATION")
    t.putkeyword("ANTENNA", f"Table: {calpath}/ANTENNA")
    t.putkeyword("FIELD", f"Table: {calpath}/FIELD")
    t.putkeyword("SPECTRAL_WINDOW", f"Table: {calpath}/SPECTRAL_WINDOW")
    t.putkeyword("HISTORY", f"Table: {calpath}/HISTORY")
    t.flush()

    ### add info...
    t.putinfo({
        "type": "Calibration",
        "subType": "B Jones",
        "readme": "",
    })

    t.close()

    logger.info(f"calibration table saved to {calpath} successfully...")

def run_convert(calpath, tabpath, datapath=None, freqpath=None, overwrite=False):

    if os.path.exists(tabpath) and not overwrite:
        logger.info(f"calibration table {tabpath} already exists...")
        logger.info(f"ovewrite flag set to False... will do nothing...")
        return

    ### work out the frequency file position
    if freqpath is None:
        logger.info(f"no frequency file path provided - will work it our automatically")
        if "smooth" in calpath:
            freqpath = calpath.replace(".smooth.npy", ".freq.npy")
        elif "bin" in calpath:
            freqpath = calpath.replace(".bin", ".freq.npy")
        else:
            logger.error("please specify --freqpath as the calibration file does not have a standard name")
            raise ValueError(f"No freqpath provided, not a standard calpath...  - calpath: {calpath}")

    logger.info(f"loading frequency file - {freqpath}...")
    freqs = np.load(freqpath)

    ### load bp files...
    if calpath.endswith(".bin"):
        logger.info(f"calpath is in the format of binary file... start loading...")
        bpcls = plotbp.Bandpass.load(calpath)
        bp = bpcls.bandpass.copy()
    else: # assume all other files are nnumpy files...
        logger.info(f"calpath likely numpy files... start loading...")
        bp = np.load(calpath)

    if os.path.exists(tabpath):
        logger.info(f"{tabpath} already exists...")
        if overwrite: 
            logger.info(f"overwrite flag set to True... deleting the file...")
            os.system(f"rm -r {tabpath}")
        else: # it should be handled above... but just in case...
            logger.info(f"ovewrite flag set to False... will do nothing...")
            return

    make_full_casabp(tabpath, bp, freqs, datapath=datapath)


def main():
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
    parser = ArgumentParser(description='Produce casa calibration table from CRACO numpy solution', formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("-c", "--calpath", type=str, help="Path to the CRACO solution", required=True)
    parser.add_argument("-p", "--tabpath", type=str, help="Path to the casa calibration table", required=True)
    parser.add_argument("-d", "--datapath", type=str, help="Path to the data files", default=None)
    parser.add_argument('-f', "--freqpath", type=str, help="Path to the frequency numpy file", default=None)
    parser.add_argument("--overwrite", action='store_true', help="overwrite the casa calibration table",default = False)

    values = parser.parse_args()

    run_convert(values.calpath, values.tabpath, values.datapath, values.freqpath, values.overwrite)

if __name__ == "__main__":
    main()