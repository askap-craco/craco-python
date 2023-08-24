#!/usr/bin/env python
# function to extract a snippet of uvfits data

from craft import uvfits
import numpy as np

from astropy.io import fits

def uvfits_snippet(uvpath, tstart, tend, outpath=None):
    """
    note: tend not included...
    """
    uvsource = uvfits.open(uvpath)
    arr = uvsource.vis[tstart*uvsource.nbl:tend*uvsource]
    
    da_table = uvsource.hdulist[0]
    aux_table = uvsource.hdulist[1:]
    
    
    par_name, par_data = make_par(arr)
    bu_data = fits.GroupData(
        arr["DATA"], bitpix=-32, bzero=0.0, bscale=1.0,
        parnames=par_name, pardata=par_data,
    )
    
    bu_table = fits.GroupsHDU(bu_data, header=da_table.header)
    nhdu = fits.HDUList([bu_table, *aux_table])
    
    if outpath is None:
        outpath = uvpath.replace(".uvfits", f".t{tstart}_{tend}.uvfits".format())
    nhdu.writeto(outpath)

