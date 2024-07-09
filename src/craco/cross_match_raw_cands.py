import numpy as np
import pandas as pd
from candidate_manager import SBCandsManager
import glob

import warnings
warnings.filterwarnings('ignore')

def main():
    x = SBCandsManager("SB061942")

    for candfile in x.clustered_raw_candfiles:
        print(candfile)
        beamid = candfile.beamid
        clustered_rfi_file = x.filter_candfiles(x.clustered_rfi_candfiles, beamid=beamid, tstart=candfile.tstart, scanid=candfile.scanid)
        clustered_uniq_file = x.filter_candfiles(x.clustered_uniq_candfiles, beamid=beamid,tstart=candfile.tstart, scanid=candfile.scanid)
        for ii, icand in candfile.cands.iterrows():
            alias = False
    
            is_it_in_rfi = np.where( (clustered_rfi_file[0].cands['cluster_id'].values == int(icand['cluster_id'])) &\
                                (clustered_rfi_file[0].cands['spatial_id'].values == int(icand['spatial_id'])) )
            is_it_in_uniq = np.where(  (clustered_uniq_file[0].cands['cluster_id'].values == int(icand['cluster_id']))  &\
                                 (clustered_uniq_file[0].cands['spatial_id'].values == int(icand['spatial_id']))  )
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
    
                #print('cuc', cuc)
                
                if (cuc['PSR_name'].isna() & cuc['RACS_name'].isna() & cuc['NEW_name'].isna() & cuc['ALIAS_name'].isna()).values:
                    label='UKNOWN'
                elif (~cuc['PSR_name'].isna()).values:
                    label='PSR'
                elif (~cuc['RACS_name'].isna()).values:
                    label='RACS'
                elif (~cuc['NEW_name'].isna().values):
                    label='CUSTOM'
                elif (~cuc['ALIAS_name'].isna()).values:
                    alias = True
                    if cuc['ALIAS_name'].values[0].startswith("J"):
                        label='PSR'
                    elif cuc['ALIAS_name'].values[0].startswith("RACS_"):
                        label='RACS'
                    else:
                        label='CUSTOM'
            else:
                raise RuntimeError(f"Could not find cluster id {int(icand['cluster_id'])} and {int(icand['spatial_id'])} in the RFI or uniq files")
    
            candfile.cands.iloc[ii]['LABEL'] = label
            candfile.cands.iloc[ii]['ALIASED'] = alias
    print("DONE")
     
main()
