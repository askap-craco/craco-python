#!/usr/bin/env python
from craco import uvfits_meta

import subprocess
import glob
import os

import logging
log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def subprocess_execute(cmds, envs=None,):
    """
    run the command with `subprocess` and print out messages on stdout
    """
    if isinstance(cmds, str): cmds = [cmds]
    if envs is None: envs = os.environ.copy()

    log.info(f"running following command {cmds}")
    with subprocess.Popen(
        cmds, stdout=subprocess.PIPE, bufsize=1, universal_newlines=True,
        env=envs, text=True, shell=True,
    ) as p:
        for outputline in p.stdout:
            print(outputline, end="")

    return p.returncode

### function to run each step
class UvfitsImager:
    def __init__(
        self, uvpath, calpath=None, tstart=0, tend=-1,
        casabin="/CRACO/SOFTWARE/craco/wan342/Software/casa-release-5.4.1-31.el7/bin/casa",
        cleanpath="`which casa_quick_clean.py`",
        snippath="`which uvfits_snippet.py`",
        sfpath="`which source_find.py`",
        cleandir=False,
    ):
        self.uvpath = uvpath
        self.calpath = calpath

        ### set snippet length
        if tstart < 0: tstart = 0
        self.tstart = tstart
        maxblock = self.uvfits_nblocks
        if tend > maxblock or tend == -1: tend = maxblock
        self.tend = tend

        self.casabin = casabin
        self.snippath = snippath
        self.cleanpath = cleanpath
        self.sfpath = sfpath
        self.cleandir = cleandir

        self._prepare_imaging()

    def _prepare_imaging(self):
        # work out folder
        pathsplit = self.uvpath.split("/")
        self.workdir = "/".join(pathsplit[:-1]) + "/snippet"
        if not os.path.exists(self.workdir): os.makedirs(self.workdir)
        self.uvfname = pathsplit[-1]
        # snippet name
        outfname = self.uvfname.replace(".fits", "")
        outfname += f".t{self.tstart}_{self.tend}"
        if self.calpath is not None: outfname += ".cal"
        self.snipfpath = f"{self.workdir}/{outfname}.uvfits"
        self.imagepath = f"{self.workdir}/{outfname}.image.fits"

    @property
    def uvfits_nblocks(self):
        uvsource = uvfits_meta.open(self.uvpath)
        maxblock = uvsource.nblocks_raw
        uvsource.close()
        return maxblock

    def fix_uvfits(self):
        log.info(f"running fixuvfits on {self.uvpath}...")
        subprocess_execute(f"`which fixuvfits` {self.uvpath}")

    def snippet_uvfits(self):
        log.info(f"making uvfits snippet... from tstart - {self.tstart} to tend - {self.tend}")
        cmd = f"python {self.snippath} {self.uvpath}"
        if self.calpath is not None: cmd += f" -calib {self.calpath}"
        cmd += f" -tstart {self.tstart} -tend {self.tend}"
        cmd += f" -outname {self.snipfpath}"
        subprocess_execute(cmd)

    def tclean_uvfits(self):
        log.info(f"running casa tclean on {self.snipfpath}")
        cmd = f"{self.casabin} --nologger --log2term -c"
        cmd += f" {self.cleanpath} {self.snipfpath} {self.workdir} f f" # no flag, no w-proj
        subprocess_execute(cmd)

    def sourcefind_fits(self):
        log.info("get path to the cleaned image...")
        fitsfiles = glob.glob(f"{self.workdir}/clean/*/*.image.fits")
        if len(fitsfiles) == 0: self.rawimagepath = None
        else: self.rawimagepath = fitsfiles[0]

        if self.rawimagepath is None: return
        log.info(f"copying cleaned image from - {self.rawimagepath} to {self.workdir}")
        subprocess_execute(f"cp {self.rawimagepath} {self.imagepath}")

        log.info(f"running pybdsf on {self.imagepath}...")
        subprocess_execute(f"{self.sfpath} -fits {self.imagepath}")

    def clean_workdir(self):
        log.info("remove unuseful folders and files...")
        cmd = f"rm -r {self.workdir}/*.ms"
        subprocess_execute(cmd)
        cmd = f"rm -r {self.workdir}/clean"
        subprocess_execute(cmd)

    def run(self):
        self.fix_uvfits()
        self.snippet_uvfits()
        self.tclean_uvfits()
        self.sourcefind_fits()
        if self.cleandir: self.clean_workdir()

def main():
    # the following are some default setting the script location etc.
    casabin = "/CRACO/SOFTWARE/craco/wan342/Software/casa-release-5.4.1-31.el7/bin/casa"
    snippet_script_path = "`which uvfits_snippet.py`"
    tclean_script_path = "`which casa_quick_clean.py`"
    sf_script_path = "`which source_find.py`"
    # snippet_script_path = "/CRACO/DATA_00/craco/wan342/craco-python/scripts/uvfits_snippet.py"
    # tclean_script_path = "/CRACO/DATA_00/craco/wan342/craco-python/scripts/casa_quick_clean.py"
    # sf_script_path = "/CRACO/DATA_00/craco/wan342/craco-python/scripts/source_find.py"

    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
    parser = ArgumentParser(description="casa uvfits imager", formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("-uvpath", type=str, help="path to the uvfits file for casa imaging", required=True)
    parser.add_argument("-calpath", type=str, help="path to the calibration to be used", default=None)
    parser.add_argument("-tstart", type=int, help="start sample for making uvfits", default=0)
    parser.add_argument("-tend", type=int, help="end sample for making uvfits", default=-1)
    parser.add_argument("-casabin", type=str, help="path to the casa binary", default=casabin)
    parser.add_argument("-snippath", type=str, help="path to the snippet script path", default=snippet_script_path)
    parser.add_argument("-cleanpath", type=str, help="path to the tclean script path", default=tclean_script_path)
    parser.add_argument("-sfpath", type=str, help="path to the source finding scripts", default=sf_script_path)
    parser.add_argument("-cleandir", action="store_true", help="whether to clean the work directory or not")

    args = parser.parse_args()

    imager = UvfitsImager(
        args.uvpath, args.calpath, args.tstart, args.tend, 
        args.casabin, args.cleanpath, args.snippath, args.sfpath,
        args.cleandir,
    )
    imager.run()

if __name__  == "__main__":
    main()

