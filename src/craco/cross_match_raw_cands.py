import numpy as np
import pandas as pd
from candidate_manager import SBCandsManager
import argparse
import logging
import os

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

    log.info(f"Working on SBID - {args.sbid}")
    x = SBCandsManager(args.sbid)
    log.debug(f"Found {x.n_clusteredrawcandfiles} rawcat candfiles to add labels to")
    for candfile in x.clustered_raw_candfiles:
        log.debug(f"Working on candfile - {candfile}")
        beamid = candfile.beamid
        clustered_rfi_file = x.filter_candfiles(x.clustered_rfi_candfiles, beamid=beamid, tstart=candfile.tstart, scanid=candfile.scanid)
        clustered_uniq_file = x.filter_candfiles(x.clustered_uniq_candfiles, beamid=beamid,tstart=candfile.tstart, scanid=candfile.scanid)
        for ii, icand in candfile.cands.iterrows():
            alias = 0
            cluster_id_in_rfi = (clustered_rfi_file[0].cands['cluster_id'].values == int(icand['cluster_id']))
            cluster_id_in_uniq = (clustered_uniq_file[0].cands['cluster_id'].values == int(icand['cluster_id']))
    
            ncands_in_rfi = len(clustered_rfi_file[0].cands[cluster_id_in_rfi])
            ncands_in_uniq = len(clustered_uniq_file[0].cands[cluster_id_in_uniq])
    
            if ncands_in_uniq > 0:
    
                if ncands_in_uniq == 1:
                    label, alias = get_uniq_classification(clustered_uniq_file[0].cands[cluster_id_in_uniq])
    
                elif ncands_in_uniq == 2:
                    
                    spatial_id_in_uniq = (clustered_uniq_file[0].cands['spatial_id'].values == int(icand['spatial_id'])) & cluster_id_in_uniq
                    nspatial_match_in_uniq = len(clustered_uniq_file[0].cands[spatial_id_in_uniq])
    
                    if nspatial_match_in_uniq == 1:
                        #We have got a match of cluster id and spatial id in the uniq file - simple case, get the classification
                        label, alias = get_uniq_classification(clustered_uniq_file[0].cands[spatial_id_in_uniq])
                    
                    elif nspatial_match_in_uniq == 0:
                        # We got a match of the cluster id, but not the spatial id - so now we assume that the spatial id for this icand was dropped
                        #So we have to pick the brightest candidate that made it, and associate icand with that candidate
                        try:
                            brightest_sibling_loc = [np.argmax(clustered_uniq_file[0].cands.iloc[cluster_id_in_uniq]['SNR'])]
                        except KeyError as KE:
                            brightest_sibling_loc = [np.argmax(clustered_uniq_file[0].cands.iloc[cluster_id_in_uniq]['snr'])]
                        brightest_sibling = clustered_uniq_file[0].cands.iloc[cluster_id_in_uniq].iloc[brightest_sibling_loc]
                        
                        label, alias = get_uniq_classification(brightest_sibling)
    
                else:
                    log.critical("Something unexpected happened in the unique classification")
                    raise RuntimeError("Something unexpercted happened in uniq classification")
    
            
            elif ncands_in_rfi > 0:
                #If we have found the cluster id of this candidate in the RFI file
                #Then we try to determine which type of RFI was this candidate
                alias = 0
                #print("I was here " + str(ncands_in_rfi))
                #print(clustered_rfi_file[0].cands[cluster_id_in_rfi], type(clustered_rfi_file[0].cands[cluster_id_in_rfi]))
                if ncands_in_rfi == 1:
                    label = get_RFI_classification(clustered_rfi_file[0].cands[cluster_id_in_rfi])
                elif ncands_in_rfi == 2:
                    spatial_id_in_rfi = (clustered_rfi_file[0].cands['spatial_id'].values == int(icand['spatial_id'])) & cluster_id_in_rfi
                    nspatial_match_in_rfi = len(clustered_rfi_file[0].cands[spatial_id_in_rfi])
                    
                    if nspatial_match_in_rfi == 1:
                        #print(type(clustered_rfi_file[0].cands[spatial_id_in_rfi]), clustered_rfi_file[0].cands[spatial_id_in_rfi])
                        label = get_RFI_classification(clustered_rfi_file[0].cands[spatial_id_in_rfi])
                    elif nspatial_match_in_rfi == 0:
                        #Neither CGHOST nor NOISE candidates should have sidelobes. So the fact that this candidate is missing must mean
                        #that it is a sidelobe of something that is not CGHOST OR NOISE, i.e. RFI
                        label = 'RFI'
                    else:
                        log.critical("Something unexpected happened in rfi classification")
                        raise RuntimeError("Something unexpected happened in rfi classification")
            else:
                log.critical("Something very seriously went wrong in the candpipe")
                raise RuntimeError("Something very seriously went wrong in the candpipe")
    
            candfile.cands.loc[ii, 'LABEL'] = label
            candfile.cands.loc[ii, 'ALIASED'] = alias

        outname = candfile.fname.strip(".csv") + ".labelled.csv"
        log.info(f"Saving the output to {outname}")
        candfile.cands.to_csv(outname, index=False)

    log.info(f"Finished processing - {args.sbid}")
    log.info("----------------------------------------------------")


if __name__ == '__main__':

    
    log = logging.getLogger(__name__)
    logging.basicConfig(filename="/CRACO/SOFTWARE/craco/craftop/logs/cross_match_raw_cands.log",
                        format='[%(asctime)s] %(levelname)s: %(message)s', 
                        level=logging.DEBUG)

    a = argparse.ArgumentParser()
    a.add_argument("sbid", type=str, help="SBID to process")
    args = a.parse_args()
    main()
