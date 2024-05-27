

from craco import datadirs as DD
from craco.metadatafile import MetadataFile

class ObsInfo:

    def __init__(self, sbid):
        self.sbid = DD.format_sbid(sbid) 
        self.schedir = DD.SchedDir(self.sbid)
        
        