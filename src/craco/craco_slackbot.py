#!/usr/bin/env python
import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN

from slack_sdk import WebClient

import glob
import re
import os

class CracoCand:
    def __init__(
        self, candpath, threshold=8,
        mentionlst=["<@U049R2ZMKAN>", "<@U012FPE7D2B>"]
    ):
        self.get_cand_info(candpath)
        
        self.df = pd.read_csv(candpath, index_col=0)
        self.df = self.df[self.df["SNR"] >= threshold].copy()
        ### convert everything to string for RACS_name and PSR_name
        self.df["PSR_name"] = self.df["PSR_name"].astype(str)
        self.df["RACS_name"] = self.df["RACS_name"].astype(str)
        self.df["NEW_name"] = self.df["NEW_name"].astype(str)
        
        self.racs_source = self.get_known_racs_source()
        self.psr_source = self.get_known_psr_source()
        self.unknown_source = self.get_unknown_source()
        self.newcat_source = self.get_newcat_source()
        
        self.mentionlst = mentionlst
        
    def get_cand_info(self, fpath):
        # pat = "craco/SB(\d{6})/scans/(\d{2})/(\d+)/(.*)/clustering_output/candidates.b(\d{2}).txt"
        pat = "craco/SB(\d{6})/scans/(\d{2})/(\d+)/(.*)/clustering_output/candidates.txtb(\d{2})"
        match = re.findall(pat, fpath)
        assert len(match) == 1, "wrong file path produced..."
        self.sbid, self.scan, self.tstart, self.runname, self.beam = match[0]
        self.sbid = int(self.sbid)
        
    def _group_by_known_source(self, df, bycol, pickcol="SNR"):
        ### here we use for loop to do that, you could try using another way
        unique_src = df[bycol].unique()
        
        src_cand_lst = []
        for src in unique_src:
            srcdf = df[df[bycol] == src]
            src_cand_lst.append(
                srcdf.iloc[[srcdf[pickcol].argmax()]]
            )
        return pd.concat(src_cand_lst)
        
    
    def get_known_psr_source(self, ):
        psr_cands = self.df[self.df["PSR_name"] != "nan"]
        if len(psr_cands) == 0: return None
        
        return self._group_by_known_source(psr_cands, "PSR_name")
    
        
    def get_known_racs_source(self, ):
        racs_cands = self.df[self.df["RACS_name"] != "nan"]
        if len(racs_cands) == 0: return None
        
        return self._group_by_known_source(racs_cands, "RACS_name")

    def get_newcat_source(self, ):
        newcat_cands = self.df[self.df["NEW_name"] != "nan"]
        if len(newcat_cands) == 0: return None

        return self._group_by_known_source(newcat_cands, "NEW_name")
        
    def get_unknown_source(self, groupradius=30, min_samples=1):
        unknown_source = self.df[
            (self.df["PSR_name"] == "nan") &
            (self.df["RACS_name"] == "nan")
        ].copy()
#         ### for testing
#         unknown_source = self.df.copy()
        if len(unknown_source) == 0: return None
        
        ### run a basic cluster just based on the ra and dec...
        dbscan = DBSCAN(eps=groupradius/3600., min_samples=min_samples)
        dbscan = dbscan.fit(unknown_source[["ra_deg", "dec_deg"]].to_numpy())
        unknown_source["source_id"] = dbscan.labels_
        unknown_source["sourcename"] = unknown_source["source_id"].apply(lambda x:f"unknown{x}")
        
        ###
        return self._group_by_known_source(unknown_source, "source_id")
    
    def _construct_cand_page(self, port=8023, unique=True):
        # http://localhost:8023/beam?sbid=53268&beam=23&unique=True&scanpath=00/20230926222942/results
        scanpath = f"{self.scan}/{self.tstart}/{self.runname}"
        return f"localhost:{port}/beam?sbid={self.sbid}&beam={self.beam}&unique={unique}&scanpath={scanpath}"
    
    def _construct_source_page(self, row, port=8023):
        # sbid=53268&beam=23&scan=00&tstart=20230926222942&results=results
        # &dm=0.0&boxcwidth=6&lpix=118&mpix=234&totalsample=4964&ra=120.62876&dec=-17.86136
        filequery = f"sbid={self.sbid}&beam={self.beam}&scan={self.scan}&tstart={self.tstart}&runname={self.runname}"
        candquery = f'dm={row["dm_pccm3"]}&boxcwidth={row["boxc_width"]}&lpix={row["lpix"]}&mpix={row["mpix"]}&totalsample={row["total_sample"]}&ra={row["ra_deg"]}&dec={row["dec_deg"]}'
        return f"localhost:{port}/candidate?{filequery}&{candquery}"
    
    ### format slack information...
    def _format_slack_df_notice(self, df, header="", maxsrc=10, srcnamecol=""):
        if len(df) == 0: return None # if there is no source... return nothing
        
        notice = f"*{header}* - "
        notice += f"{len(df)} sources found \n"
        if len(df) > maxsrc:
            notice += f"more than {maxsrc} sources found in the dataframe... please refer to the candidate page - "
            notice += f"<http://{self._construct_cand_page()}|*unique candidate page*>"
            
        else:
            for i, row in df.iterrows():
                notice += f'\t• *{row[srcnamecol]}* - <http://{self._construct_source_page(row)}|candidate page> \n'
        return notice
    
    def format_slack_notice(self, maxsrc=10, racs=True, psr=True, unknown=True, newcat=True):
        if len(self.df) == 0: return None
        
        notice = f"*{self.sbid} BEAM{self.beam} candidate notification* - _{self.scan}/{self.tstart}/{self.runname}_ \n"
        notice += f"{len(self.df)} unique candidates found in this run \n"
        if racs and self.racs_source is not None:
            notice += self._format_slack_df_notice(
                self.racs_source, header="RACS Bright Sources", maxsrc=maxsrc, srcnamecol="RACS_name"
            )
        if psr and self.psr_source is not None:
            notice += self._format_slack_df_notice(
                self.psr_source, header="Known Pulsars", maxsrc=maxsrc, srcnamecol="PSR_name"
            )
        if newcat and self.newcat_source is not None:
            notice += self._format_slack_df_notice(
                self.newcat_source, header="**TEST** User Defined Sources", maxsrc=maxsrc, srcnamecol="NEW_name"
            )
        
        if unknown and self.unknown_source is not None:
            notice += self._format_slack_df_notice(
                self.unknown_source, header="Unknown Sources", maxsrc=maxsrc, srcnamecol="sourcename"
            )
            
            ### add mention here
            if len(self.mentionlst) > 0: 
                notice += ", ".join(self.mentionlst) + " please check..."
        
        return notice

def get_slack_client():
    token = os.environ["SLACK_CRACO_TOKEN"]
    return WebClient(token=token)

def send_slack_msg(msg, client, channel="C05Q11P9GRH"):
    if msg is None: return
    return client.chat_postMessage(
        channel=channel,
        text=msg,
    )

def check_piperun(sbid, runname, client):
    candfiles = glob.glob(
        f"/data/seren-??/big/craco/SB0{sbid}/scans/??/*/{runname}/clustering_output/candidates.b??.txt.uniq.csv"
        # f"/data/seren-??/big/craco/SB0{sbid}/scans/??/*/{runname}/clustering_output/candidates.txtb??.uniq.csv"
    )

    for candfile in candfiles:
        candobj = CracoCand(candfile)
        notice = candobj.format_slack_notice()
        if notice is not None:
            send_slack_msg(notice, client, )

def _main():
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
    parser = ArgumentParser(
        description='post slack notification for craco offline pipeline run...', 
        formatter_class=ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("-s", type=str, help="schedule block id to process")
    parser.add_argument("-r", type=str, help="pipeline run name", default="results")
    values = parser.parse_args()

    client = get_slack_client()
    check_piperun(values.s, values.r, client)

if __name__ == "__main__":
    _main()