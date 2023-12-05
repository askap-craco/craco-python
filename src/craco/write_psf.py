from craft import uvfits, craco_kernels, craco_plan
from craco import postprocess, uvfits_meta
import numpy as np
import matplotlib.pyplot as plt
from craft.cmdline import strrange

def write_psf(outname, plan, iblk=None):
    #print(f"HELLOOOO, {outname}")
    gridder_obj = craco_kernels.Gridder("", plan, "")
    imager_obj = craco_kernels.Imager("", plan, "")
    block = np.ones((plan.nbl, plan.nf, 2), dtype=np.complex128)
    block.imag = 0

    grid = gridder_obj(block)
    #plt.figure()
    #plt.imshow(np.abs(np.fft.fftshift(grid)), aspect='auto', interpolation='None')
    
    useful_info = {
        'NCHAN': plan.nf,
        'TSAMP': plan.tsamp_s.value,
        'FCH1_Hz': plan.freqs[0],
        'CH_BW_Hz': plan.foff,
        'NANT': plan.nant,
        'NBL': plan.nbl,
        'STARTMJD': plan.tstart.mjd,
        'IBLK': iblk,
        'TARGET': plan.target_name,
        'UV': str(plan.values.uv),
        'BSCALE': 1.0,
        'BZERO': 0.0,
        'BUNIT': "UNCALIB"
        }


    img = imager_obj(np.fft.fftshift(grid[..., 0])).astype(np.complex64)
    imgout = (img.real + img.imag) / 2
    postprocess.create_header_for_image_data(outname,
                                            wcs = plan.wcs,
                                            im_shape = (plan.npix, plan.npix), 
                                            dtype=np.dtype('>f4'), 
                                            kwargs = useful_info, 
                                            image_data = imgout)

def run_test(args):
    f = uvfits_meta.open(args.uv, skip_blocks=args.seek_samps, metadata_file = args.metadata)
    values = craco_plan.get_parser().parse_args([])
    values.flag_ants = args.flag_ants
    plan = craco_plan.PipelinePlan(f, values)
    write_psf(outname = args.o, plan = plan)

if __name__ == '__main__':
    import argparse
    a = argparse.ArgumentParser()
    a.add_argument("-o", type=str, help="Outname", required = True)
    a.add_argument("-uv", type=str, help="Path to a uvfits file to make plan from", required=True)
    a.add_argument("-metadata", type=str, help="Path to the metadata file", default=None)
    a.add_argument("-seek_samps", type=int, help="Seek samps into the uvfits file before creating a plan (def:0)", default=0)
    a.add_argument("-flag_ants", type=strrange, help="Flag these antennas (def=None)", default=None)

    args = a.parse_args()
    run_test(args)
