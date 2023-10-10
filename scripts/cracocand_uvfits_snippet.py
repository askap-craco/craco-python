from craft import uvfits, craco_plan
from craco import craco_candidate

from craft.craco import bl2array, bl2ant
from craco import preprocess

import argparse

from astropy.io import fits
import numpy as np

import warnings
warnings.filterwarnings('ignore')

def make_parameter_cols(arr):
    parnames = []
    pardata = []
    parbzeros = []
    first_date = arr['DATE'][0]
    for parname in arr.dtype.names:
        if parname != "DATA":
            parnames.append(parname)
            
            if parname == 'DATE':
                pardata.append(arr[parname] - first_date)
                parbzeros.append(first_date)
            else:
                pardata.append(arr[parname])
                parbzeros.append(0)

    return parnames, pardata, parbzeros

def makeGroupData(visrows):
    parnames, pardata, parbzeros= make_parameter_cols(visrows)
    GroupData = fits.GroupData(visrows['DATA'], parnames = parnames, pardata = pardata, bzero = 0.0, bscale = 1.0, parbzeros=parbzeros)
    return GroupData

class UvfitsSnippet:
    
    def __init__(
        self, uvfitspath, dmpccm, totalsamp,
        boxcwidth, padding=0, norm=False, calpath=None,
    ):
        self.uvsource = uvfits.open(uvfitspath)
        self.dmpccm = dmpccm
        self.totalsamp = int(totalsamp)
        self.boxcwidth = int(boxcwidth)
        self.padding = int(padding)
        
        if self.padding == 0: self.norm = False
        else: self.norm = norm
            
        self._extract_vis()
            
        self.calpath = calpath
        ### make plan
        self.plan = craco_plan.PipelinePlan(self.uvsource, "--ndm 2")
        
    def _extract_vis(self, ):
        dmdelay = craco_candidate.calculate_dm_tdelay(
            self.uvsource.channel_frequencies[0], 
            self.uvsource.channel_frequencies[-1], 
            self.dmpccm,
        )
        dmdelaysamp = round(dmdelay / self.uvsource.tsamp.value)
        self.burstrange = (self.totalsamp - dmdelaysamp - self.boxcwidth, self.totalsamp) # both included
        _visrange = (self.burstrange[0] - self.padding, self.burstrange[1] + self.padding)
        
        ### start to extract data and also update the visrange
        burst_data_dict, self.burst_uvw, self.vis_range = self.uvsource.time_block_with_uvw_range(_visrange)
        self.data = bl2array(burst_data_dict)
#         self.blids = sorted(burst_data_dict.keys())
        
        nbl = self.uvsource.nbl
        self.dummy_data = self.uvsource.vis[
            (self.totalsamp - self.boxcwidth - 1)*nbl:self.totalsamp*nbl
        ]
        self.dummy_data["DATA"][..., -1] = 1. # put all dummy weights to 1
        
    def calibrate_data(self, data, calpath=None):
        if calpath is None: return data
        calibrator = preprocess.Calibrate(
            plan=self.plan, block_dtype=np.ma.core.MaskedArray,
            miriad_gains_file=calpath, 
            baseline_order=self.plan.baseline_order
        )

        return calibrator.apply_calibration(data)
    
    def normalise_data(self, data, target_rms=1):
        if self.norm == False: 
            print("no normalisation allowed...")
            return preprocess.fill_masked_values(data, fill_value=0)
        norm_data = preprocess.normalise(
            data, target_input_rms=target_rms,
        )

        return preprocess.fill_masked_values(norm_data, fill_value=0) # in this case it is normal numpy array
    
    def dedisperse_data(self, data, dm):
        dedisperser = preprocess.Dedisp(
            freqs=self.uvsource.channel_frequencies, tsamp=self.uvsource.tsamp.value,
            dm_pccc=dm
        )

        dedisp_data = dedisperser.dedisperse(0, data)
        if isinstance(dedisp_data, np.ma.core.MaskedArray):
            return preprocess.fill_masked_values(dedisp_data, fill_value=0)
        return dedisp_data
    
    def extract_vis(self, data):
        end_idx = self.totalsamp - self.vis_range[0]
        start_idx = self.totalsamp - self.vis_range[0] - self.boxcwidth
        return data[..., start_idx:end_idx+1]
    
    def put_data_dummy(self, data):
        assert data.ndim == 4, "only 4-dimesional arrays are allowed..."
        data_nbl, data_nchan, data_npol, data_nt = data.shape
        
        assert data_npol == 1, "we do not support multiple polarisation products currently..."
        
        datavis = self.dummy_data.copy()
        ### work out how many nt, nbl, nchan
        nrow = datavis.size
        nbl = self.uvsource.nbl
        nt = nrow // nbl
        nchan = datavis[0]["DATA"].shape[3]
        assert nbl == data_nbl, f"not the same number of baselines - dummy got {nbl}, but data got {data_nbl}"
        assert nt == data_nt, f"not the same number of samples - dummy got {nt}, but data got {data_nt}"
        assert nchan == data_nchan, f"not the same number of samples - dummy got {nchan}, but data got {data_nchan}"
        
        maskarray = False
        if isinstance(data, np.ma.core.MaskedArray):
            maskarray = True
            mask = data.mask
            data = data.data
        
        ### start to write the data...
        for it in range(nt):
            datavis["DATA"][
                it*nbl:(it+1)*nbl, 0, 0, 0, :, 0, 0
            ] = data[..., 0, it].real # real
            datavis["DATA"][
                it*nbl:(it+1)*nbl, 0, 0, 0, :, 0, 1
            ] = data[..., 0, it].imag # imaginary
            
            if maskarray:
                datavis["DATA"][
                    it*nbl:(it+1)*nbl, 0, 0, 0, :, 0, 2
                ] = (~mask[..., 0, it]).astype(float) # weight
            
        return datavis
    
    def dump_data_uvfits(self, datavis, outname):
        datagroup = makeGroupData(datavis)
        datahdu = fits.GroupsHDU(datagroup, header=self.uvsource.hdulist[0].header)
        auxhdus = self.uvsource.hdulist[1:]
        
        newhdu = fits.HDUList([datahdu, *auxhdus])
        
        newhdu.writeto(outname, overwrite=True)
        
    def dump_burst_data(self, outname=None):
        ### work out name first
        if outname is None:
            uvfname = self.uvsource.filename.split("/")[-1]
            namelst = []
            if self.calpath is not None: namelst.append("cal")
            if self.norm: namelst.append("norm")
            ### information for dedispersion
            namelst.append(f"dm{self.dmpccm:.2f}")
            namelst.append(f"samp{self.totalsamp}")
            namelst.append(f"boxcw{self.boxcwidth}")
            namelst.append(f"pad{self.padding}")
            namelst.append(f"nbl{self.uvsource.nbl}")
            newname = ".".join(namelst)
            
            outname = uvfname.replace(".uvfits", f".{newname}.uvfits")
            
        data = self.calibrate_data(self.data, self.calpath)
        data = self.normalise_data(data, target_rms = 1)
        data = self.dedisperse_data(data, self.dmpccm)
        
        data = self.extract_vis(data)
        visdata = self.put_data_dummy(data)
        
        self.dump_data_uvfits(visdata, outname)

if __name__ == "__main__":
    a = argparse.ArgumentParser()
    a.add_argument("-uv", type=str, help="Path to the uvfits file to read from")
    a.add_argument("-cal", type=str, help="Path to the calibration file, None if no calibration to be applied", default=None)
    a.add_argument("-dm", type=float, help="Despersion measure (physical units) to apply, default is 0", default=0.)
    a.add_argument("-ts", type=int, help="total sample of the candidate detection directly from the pipeline")
    a.add_argument("-w", type=int, help="boxcar width of the candidate, 0 by default", default=0.)
    a.add_argument("-pad", type=int, help="how many samples to pad (to get normalisation)", default=30)
    a.add_argument("-norm", action="store_true", help="normalise the data", default=False)

    args = a.parse_args()

    uvsnippet = UvfitsSnippet(
        uvfitspath=args.uv, dmpccm=args.dm, totalsamp=args.ts,
        boxcwidth=args.w, padding=args.pad, norm=args.norm, 
        calpath=args.cal,
    )

    uvsnippet.dump_burst_data()