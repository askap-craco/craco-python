import numpy as np
import pandas as pd
from candidate_manager import SBCandsManager
import glob



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


def main():
    x = SBCandsManager("SB063225")

    for candfile in x.clustered_raw_candfiles:
        print(candfile)
        beamid = candfile.beamid
        clustered_rfi_file = x.filter_candfiles(x.clustered_rfi_candfiles, beamid=beamid, tstart=candfile.tstart, scanid=candfile.scanid)
        clustered_uniq_file = x.filter_candfiles(x.clustered_uniq_candfiles, beamid=beamid,tstart=candfile.tstart, scanid=candfile.scanid)
        for ii, icand in candfile.cands.iterrows():
            alias = 0
            cluster_id_in_rfi = (clustered_rfi_file[0].cands['cluster_id'].values == int(icand['cluster_id']))
            spatial_id_in_rfi = (clustered_rfi_file[0].cands['spatial_id'].values == int(icand['spatial_id']))
            
            is_it_in_rfi = np.where( cluster_id_in_rfi & spatial_id_in_rfi  )
            
            cluster_id_in_uniq = (clustered_uniq_file[0].cands['cluster_id'].values == int(icand['cluster_id']))
            spatial_id_in_uniq = (clustered_uniq_file[0].cands['spatial_id'].values == int(icand['spatial_id']))
            
            is_it_in_uniq = np.where(  cluster_id_in_uniq  & spatial_id_in_uniq )
            if len(is_it_in_rfi[0]) > 0:
                #if int(icand['cluster_id']) in clustered_rfi_file[0].cands['cluster_id'].values and int(icand['spatial_id']) in clustered_rfi_file[0].cands['spatial_id'].values:
                #loc = np.where( (clustered_rfi_file[0].cands['cluster_id'].values == int(icand['cluster_id'])) &\
                                #(clustered_rfi_file[0].cands['spatial_id'].values == int(icand['spatial_id'])) )[0]
                loc = is_it_in_rfi[0]
                clustered_rfi_cand = clustered_rfi_file[0].cands.iloc[loc]
                #print('crfi', clustered_rfi_cand,"--->", clustered_rfi_cand['num_samps'],"====>", type(clustered_rfi_cand['num_samps']))
                
                if (126 <= icand['lpix'] <= 130) and (126 <= icand['mpix'] <= 130):
                    label='CGHOST'
                elif clustered_rfi_cand['num_samps'].values[0] < 3:
                    label='NOISE'
                else:
                    label='RFI'
    
            elif len(is_it_in_uniq[0]) > 0:
                #elif int(icand['cluster_id']) in clustered_uniq_file[0].cands['cluster_id'].values and int(icand['spatial_id']) in clustered_uniq_file[0].cands['spatial_id'].values:
                #loc = np.where(  (clustered_uniq_file[0].cands['cluster_id'].values == int(icand['cluster_id']))  &\
                #                 (clustered_uniq_file[0].cands['spatial_id'].values == int(icand['spatial_id']))  )[0]
                loc = is_it_in_uniq[0]
                clustered_uniq_cand = clustered_uniq_file[0].cands.iloc[loc]
                cuc = clustered_uniq_cand
    
                label, alias =get_uniq_classification(clustered_uniq_cand)
                #print('cuc', cuc)
    
            else:
                is_only_cluster_in_rfi =  np.where(cluster_id_in_rfi)
                is_only_cluster_in_uniq = np.where(cluster_id_in_uniq)
    
                if len(is_only_cluster_in_rfi[0]) > 2:
                    #This should never happen - there is a bug in Yuanming's code - blame her
                    raise Exception("Yuanming messed up")
                elif len(is_only_cluster_in_rfi[0]) > 0:
                    #All siblings of this candidate ended up in the RFI bucket, so we can safely regard this raw candidate as RFI too
    
                    #There is no way that NOISE AND CENTRAL GHOST can have siblings(separate spatial clusters with same cluster id) that get dropped
                    #So we assume that they are all RFI
                    label='RFI'
                elif len(is_only_cluster_in_uniq[0])>2:
                    #This should never happen - there is a bug in Yuanming's code - blame her
                    raise Exception("Yuanming messed up")
                elif len(is_only_cluster_in_uniq[0]) > 0:
                    locs = is_only_cluster_in_uniq[0]
                    if len(locs) > 1:
                        try:
                            brightest_sibling_loc = [np.argmax(clustered_uniq_file[0].cands.iloc[locs]['SNR'])]
                        except KeyError as KE:
                            brightest_sibling_loc = [np.argmax(clustered_uniq_file[0].cands.iloc[locs]['snr'])]
                        brightest_sibling = clustered_uniq_file[0].cands.iloc[locs].iloc[brightest_sibling_loc]
                    else:
                        brightest_sibling = clustered_uniq_file[0].cands.iloc[locs]
                        
                    label, alias = get_uniq_classification(brightest_sibling)
                    
                else:
                    raise RuntimeError(f"Could not find cluster id {int(icand['cluster_id'])} and {int(icand['spatial_id'])} in the RFI or uniq files")
    
            candfile.cands.loc[ii, 'LABEL'] = label
            candfile.cands.loc[ii, 'ALIASED'] = alias
    print("DONE")        



main()
