### tclean routine to check data quality...
import sys
import os

argv = sys.argv
if argv[0] == "casa": argv = argv[2:]

ms = argv[1]
workdir = argv[2]
flag = argv[3]
wide = argv[4]

niter=100
robust=-0.5
imsize=5120
threshold="2mJy"

cell="2.5arcsec"
uvrange=">100m"

### import uvfits to ms
if ms.endswith(".uvfits"):
    fitsfile=ms
    ms=fitsfile.replace(".uvfits", ".ms")
    importuvfits(
        fitsfile=fitsfile,
        vis=ms
    )


### do some flagging before imaging
if flag == "t":
    clearstat()
    flagdata(vis=ms, mode="clip", clipzeros=True)
    flagdata(vis=ms, mode="tfcrop", clipzeros=True)
    flagdata(vis=ms, mode="rflag", clipzeros=True)
    clearstat()

if ms.endswith("/"): ms = ms[:-1]
msfname = ms.split("/")[-1]

cleandir = "{}/clean/{}_niter{:.1f}_robust{}_imsize{}".format(
    workdir, msfname, niter, robust, imsize
)

if wide == "t": cleandir = cleandir + "_wide"

if not os.path.exists(cleandir):
    os.makedirs(cleandir)
else:
    os.system("rm -r {}/*".format(cleandir))

imagename="{}/{}_niter{}_robust{}_imsize{}".format(cleandir, msfname, niter, robust, imsize)

if wide == "t":
    imagename="{}/{}_niter{}_robust{}".format(cleandir, msfname, niter, robust) + "_wide"
    tclean(
        vis=ms,
        imsize=imsize,
        imagename=imagename,
        uvrange=uvrange,
        cell=cell,
        weighting="briggs",
        robust=robust,
        pblimit=-1,
        niter=niter,
        gridder="widefield",
        wprojplanes=-1,
        # savemodel="modelcolumn",
        threshold=threshold,
    )

else:
    tclean(
        vis=ms,
        imsize=imsize,
        imagename=imagename,
        uvrange=uvrange,
        cell=cell,
        weighting="briggs",
        robust=robust,
        pblimit=-1,
        niter=niter,
        # savemodel="modelcolumn",
        threshold=threshold,
    )

exportfits(
    imagename="{}.image".format(imagename),
    fitsimage="{}.image.fits".format(imagename),
)

os.system("rm casa-*.log *.last")