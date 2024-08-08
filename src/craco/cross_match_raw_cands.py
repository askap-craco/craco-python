import numpy as np
import pandas as pd
from candidate_manager import SBCandsManager
import argparse
import logging
import os, sys
import IPython

def get_uniq_classification(clustered_cand):
    '''
    Finds the appropriate classification label for a clustered candidate

    Input - A row of pandas dataframe (or numpy ordered dict) containing all the parameters of the candidate from the clustered candfile
    Output - string [UNKNOWN/PSR/RACS/CUSTOM] and if it was an alias or not (bool)
    '''
    alias = 0
    cuc = clustered_cand
    if (cuc['PSR_name'].isna() & cuc['RACS_name'].isna() & cuc['NEW_name'].isna() & cuc['ALIAS_name'].isna()).values:
        label='UKNOWN'
    elif (~cuc['PSR_name'].isna()).values:
        label='PSR'
    elif (~cuc['RACS_name'].isna()).values:
        label='RACS'
    elif (~cuc['NEW_name'].isna().values):
        label='CUSTOM'
    elif (~cuc['ALIAS_name'].isna()).values:
        alias = 1
        if cuc['ALIAS_name'].values[0].startswith("J"):
            label='PSR'
        elif cuc['ALIAS_name'].values[0].startswith("RACS_"):
            label='RACS'
        else:
            label='CUSTOM'

    return label, alias

def get_RFI_classification(clustered_rfi_cand):
    if (126 <= clustered_rfi_cand['lpix'].values <= 130) and (126 <= clustered_rfi_cand['mpix'].values <= 130):
        label='CGHOST'
    elif (clustered_rfi_cand['num_samps'].values < 3) and (clustered_rfi_cand['num_spatial'].values == 1):
        label='NOISE'
    else:
        label='RFI'
    return label

def main():

    log.info(f"Working on SBID - {args.sbid}, runname - {args.runname}")
    x = SBCandsManager(args.sbid, runname = args.runname)
    log.debug(f"Found {x.n_clusteredrawcandfiles} rawcat candfiles to add labels to")

    for candfile in x.clustered_raw_candfiles:
      try:
        log.debug(f"Working on candfile - {candfile}")
        beamid = candfile.beamid
        clustered_rfi_file = x.filter_candfiles(x.clustered_rfi_candfiles, beamid=beamid, tstart=candfile.tstart, scanid=candfile.scanid)
        clustered_uniq_file = x.filter_candfiles(x.clustered_uniq_candfiles, beamid=beamid,tstart=candfile.tstart, scanid=candfile.scanid)
        clustered_inj_file = x.filter_candfiles(x.clustered_inj_candfiles, beamid=beamid,tstart=candfile.tstart, scanid=candfile.scanid)
        inj = False
        if len(clustered_inj_file) > 0:
            inj = True

        for ii, icand in candfile.cands.iterrows():
            alias = 0
            
            cluster_id_in_rfi = (clustered_rfi_file[0].cands['cluster_id'].values == int(icand['cluster_id']))
            ncands_in_rfi = len(clustered_rfi_file[0].cands[cluster_id_in_rfi])
            spatial_id_in_rfi = (clustered_rfi_file[0].cands['spatial_id'].values == int(icand['spatial_id'])) & cluster_id_in_rfi
            nspatial_match_in_rfi = len(clustered_rfi_file[0].cands[spatial_id_in_rfi])
            
            cluster_id_in_uniq = (clustered_uniq_file[0].cands['cluster_id'].values == int(icand['cluster_id']))
            ncands_in_uniq = len(clustered_uniq_file[0].cands[cluster_id_in_uniq])
            spatial_id_in_uniq = (clustered_uniq_file[0].cands['spatial_id'].values == int(icand['spatial_id'])) & cluster_id_in_uniq
            nspatial_match_in_uniq = len(clustered_uniq_file[0].cands[spatial_id_in_uniq])

            if inj:
                cluster_id_in_inj = (clustered_inj_file[0].cands['cluster_id'].values == int(icand['cluster_id']))
                ncands_in_inj = len(clustered_inj_file[0].cands[cluster_id_in_inj])
                spatial_id_in_inj = (clustered_inj_file[0].cands['spatial_id'].values == int(icand['spatial_id'])) & cluster_id_in_inj
                nspatial_match_in_inj = len(clustered_inj_file[0].cands[spatial_id_in_inj])
            else:
                ncands_in_inj = 0
                nspatial_match_in_inj = 0


            if inj and (nspatial_match_in_inj == 1):
                #Found cluster id and spatial id in the injection file
                label="INJECTION"
            elif nspatial_match_in_rfi == 1:
                #Found cluster id and spatial id in the rfi file
                label = get_RFI_classification(clustered_rfi_file[0].cands[spatial_id_in_rfi])
            elif nspatial_match_in_uniq == 1:
                #Found cluster id and spatial id in the uniq file
                label, alias = get_uniq_classification(clustered_uniq_file[0].cands[spatial_id_in_uniq])

            if nspatial_match_in_rfi + nspatial_match_in_inj + nspatial_match_in_uniq == 0:
                #If both - cluster id and spatial id didn't match, then we have to assume that the spatial id got dropped somehow
                #Now, since the decision on whether the source is astrophysical or not dependes on m7,m6 and m3 metrics all of which 
                #are calculated using the full time-dm cluster, all spatial IDs will have the same values of m7,m6 and m3, which means
                #that the logic to classify whether a source is RFI or not will apply equally to all of them.

                #Hence - if any spatial ID ends up being labelled as RFI, I'd say all spatial clusters would have been RFI
                
                if ncands_in_rfi > 0:
                    #And since NOISE and CGHOST candidates should not have sidelobes, the only type of candidates that could have 
                    #had sidelobes is RFI
                    
                    label = "RFI"

                elif (ncands_in_uniq > 0) and (ncands_in_inj == 0):
                    #Now, if any spatial cluster ended up in the uniq file, then we pick the brightest of them and assume that the sidelobe
                    #must have belonged to the brightest candidate in the uniq file
                    if ncands_in_uniq == 1:
                        label, aliased = get_uniq_classification(clustered_uniq_file[0].cands[cluster_id_in_uniq])
                    elif ncands_in_uniq == 2:
                        try:
                            brightest_sibling_loc = [np.argmax(clustered_uniq_file[0].cands.iloc[cluster_id_in_uniq]['SNR'])]
                        except KeyError as KE:
                            brightest_sibling_loc = [np.argmax(clustered_uniq_file[0].cands.iloc[cluster_id_in_uniq]['snr'])]

                        brightest_sibling = clustered_uniq_file[0].cands.iloc[cluster_id_in_uniq].iloc[brightest_sibling_loc]
                        
                        label, alias = get_uniq_classification(brightest_sibling)
                    else:
                        log.critical("Something unexpected happened")
                        raise RuntimeError("Something unexpeceted happened")

                elif (ncands_in_inj > 0) and (ncands_in_uniq == 0):
                    label="INJECTION"

                elif ncands_in_inj == 1 and ncands_in_uniq == 1:
                    #We need to work out which one is brighter and assign the sidelobe to its classification

                    inj_snr = clustered_inj_file[0].cands.iloc[cluster_id_in_inj]['snr']
                    try:
                        uniq_snr = clustered_uniq_file[0].cands.iloc[cluster_id_in_uniq]['snr']
                    except KeyError as KE:
                        uniq_snr = clustered_uniq_file[0].cands.iloc[cluster_id_in_uniq]['SNR']

                    if inj_snr.iloc[0] > uniq_snr.iloc[0]:
                        label = "INJECTION"
                    else:
                        label, aliased = get_uniq_classification(clustered_uniq_file[0].cands[cluster_id_in_uniq])
            candfile.cands.loc[ii, 'LABEL'] = label
            candfile.cands.loc[ii, 'ALIASED'] = alias

        outname = candfile.fname.strip(".csv") + ".labelled.csv"
        log.info(f"Saving the output to {outname}")
        candfile.cands.to_csv(outname, index=False)

      except Exception as E:
        IPython.embed()

    log.info(f"Finished processing - {args.sbid}")
    log.info("----------------------------------------------------")


if __name__ == '__main__':

    
    log = logging.getLogger(__name__)
    logging.basicConfig(filename="/CRACO/SOFTWARE/craco/craftop/logs/cross_match_raw_cands.log",
                        format='[%(asctime)s] %(levelname)s: %(message)s', 
                        level=logging.DEBUG)
    stdout_handler = logging.StreamHandler(sys.stdout)
    log.addHandler(stdout_handler)

    a = argparse.ArgumentParser()
    a.add_argument("sbid", type=str, help="SBID to process")
    a.add_argument("-runname", type=str, help='Runname (def: results)', default='results')
    args = a.parse_args()
    main()
