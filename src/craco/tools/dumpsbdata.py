#!/usr/bin/env python
"""
Template for making scripts to run from the command line

Copyright (C) CSIRO 2022
"""
import pylab
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import logging
from craco.cmdline import strrange

# pylint: disable-msg=E0611
# need to import iceutils before interfaces
from askap.iceutils import get_service_object
import askap.interfaces as iceint
from askap.interfaces.schedblock import ObsState
from askap.parset import ParameterSet
import askap.parset as parset
from astropy.coordinates import SkyCoord


log = logging.getLogger(__name__)



__author__ = "Keith Bannister <keith.bannister@csiro.au>"

def _main():
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
    parser = ArgumentParser(description='Script description', formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('-v', '--verbose', action='store_true', help='Be verbose')
    parser.add_argument('--sbids', type=strrange, help='String range of sbids to dump')
    #parser.set_defaults(verbose=False)
    values = parser.parse_args()
    if values.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)
    
    import Ice
    # TODO: Wrap this nicely.
    comm = Ice.initialize(sys.argv)

    sb_service = get_service_object(comm,
                    "SchedulingBlockService@DataServiceAdapter",
                    iceint.schedblock.ISchedulingBlockServicePrx)
    
    import csv
    fieldnames = None

    with open ('sbids.csv', 'w') as fout:
        #writer = csv.writer(f)
        writer = None
            
        os.makedirs('parsets', exist_ok=True)

        for sbid in values.sbids:
            log.info('Loading SBID %d', sbid)
            try:
                f = {'sbid':sbid,
                 'alias':sb_service.getAlias(sbid),
                 'state':sb_service.getState(sbid),
                 'owner': sb_service.getOwner(sbid),
                 'sbtemplate': sb_service.getSBTemplate(sbid),
                 'schedtime': sb_service.getScheduledTime(sbid)}
            except iceint.schedblock.NoSuchSchedulingBlockException:
                log.info('SBID %d not found', sbid)
                continue

            ofile = f'parsets/SB{sbid:06}_obs_variables.parset'
            pfile = f'parsets/SB{sbid:06}_obs_parameters.parset'
            # Some parsets contiain error messages that are multiline. These can't be parsed off disk.
            # as it throws an exception
            # We just try and parse them, and if it fails, we fetch them from the service.
            try:
                obs_variables = ParameterSet(ofile)
            except:
                obs_variables = ParameterSet(sb_service.getObsVariables(sbid, ''))

            try:
                obs_params = ParameterSet(pfile)
            except:
                obs_params = ParameterSet(sb_service.getObsParameters(sbid))
                
            obs_variables.to_file(ofile)
            obs_params.to_file(pfile)
            p = obs_variables.to_flat_dict()
            p.update(obs_params.to_flat_dict())
            #p = parset.merge(obs_variables, obs_params)

       
            for k,v in p.items(): 
                if '%d' in k:
                    k = k % 1

                try :
                    if k.endswith('field_direction'):
                        bits = v.replace('[','').replace(']','').split(',')
                        ra, dec, csys = bits
                        field_direction = SkyCoord(ra, dec, unit=('hourangle', 'degree'))
                        f['field_direction_ra'] = field_direction.ra.deg
                        f['field_direction_dec'] = field_direction.dec.deg
                    elif k.endswith('pol_axis'):
                        f['pol_axis_name'], f['pol_axis_angle'] = v.replace('[','').replace(']','').split(',')
                    elif k.startswith('schedblock'):
                        pass
                    else:
                        if isinstance(v, list) or v.strip().startswith('['):
                            pass
                        else:
                            #print(k,v)
                            if fieldnames is None or k in fieldnames:
                                f[k] = v
                except:
                    log.info('value for %s=%s unparseable', k, v)
            
            if writer is None:
                fieldnames = f.keys()
                writer = csv.DictWriter(fout, fieldnames=fieldnames)
                
                writer.writeheader()

            writer.writerow(f)
                 

    

    

if __name__ == '__main__':
    _main()
