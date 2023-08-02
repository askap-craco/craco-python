import numpy as np
from craft.craco_kernels import Prepare, Gridder, Imager ,CracoPipeline, FdmtGridder
from craft import craco_plan
from craft import uvfits
from craft.cmdline import strrange
from craco import preprocess, postprocess
from craft.craco import bl2array
import matplotlib.pyplot as plt
import IPython
from PIL import Image


def make_PIL_images_from_array(arr, cmap=None):

    def normalise_image(image):
        image -= np.min(image)
        image_max = max([np.max(image), 1])     #To avoid dividing by zeros when image is all zeros
        image = image / image_max
        return image

    if cmap is None:
        cmap = 'viridis'
    cm = plt.get_cmap(cmap)
    if arr.ndim == 3:
        #Assume that the last axis is time, so make an image for every frame
        image_stack = []
        for ii in range(arr.shape[-1]):
            this_image = normalise_image(arr[..., ii])
            colored_image = cm(this_image)
            #Extreact the RGB values and convert them into integers 0-255 (they are in float 0-1 before this step)
            colored_image = (colored_image[..., :3] * 255).astype(np.uint8)
            image_stack.append(Image.fromarray(colored_image))
            return image_stack
        
    elif arr.ndim == 2:
        this_image = normalise_image(arr)
        colored_image = cm(this_image)
        colored_image = (colored_image[..., :3] * 255).astype(np.uint8)
        return Image.fromarray(colored_image)
    else:
        raise ValueError(f"The array should have 2 or 3 dimensions, given - {arr.ndim}")
    


def plot_block(block, title = None):
    if type(block) == dict:
        myblock = bl2array(block)
    else:
        myblock = block

    if myblock.ndim == 4:
        #Pol axis exists
        myblock = myblock.copy().mean(axis=-2)

    f, ax = plt.subplots(2, 2)
    ax[0, 0].imshow(myblock.real.sum(axis=0).squeeze(), aspect='auto')
    ax[0, 0].set_title("Sum of all baselines.real")
    ax[0, 1].imshow(myblock.imag.sum(axis=0).squeeze(), aspect='auto')
    ax[0, 1].set_title("Sum of all baselines.imag")
    ax[1, 0].imshow(np.abs(myblock).sum(axis=0).squeeze(), aspect='auto')
    ax[1, 0].set_title("Sum of all abs(baselines)")
    ax[1, 1].imshow(myblock[3].real, aspect='auto')
    ax[1, 1].set_title("Baseline 3 real part")
    if title:
        f.suptitle(title)
    plt.show()

def get_parser():
    parser = craco_plan.get_parser()
    parser.add_argument("--injection_params_file", type=str, help="Path to an injection params file", default=None)
    parser.add_argument("--seek_s", type=float, help="Seek (in seconds) into the file (def:0)", default=None)
    parser.add_argument("--seek_samps", type=float, help="Seek (in samps) into the file (def:0)", default=None)
    parser.add_argument("--tx", type=int, help="Average in time by a factor of tx", default=1)
    parser.add_argument("--norm", action='store_true', help="Normalise the data (baseline subtraction and rms setting to 1)",default = False)
    parser.add_argument("--proper", action='store_true', help="Do proper uvw coordinate while gridding", default=False)
    parser.add_argument("--rfi", action='store_true', help="Perform RFI mitigation on the data", default = False)
    parser.add_argument('--flag-chans', help='Flag these channel numbers (strrange)', type=strrange)
    g = parser.add_mutually_exclusive_group()
    g.add_argument("--dedm_pccc", type=float, help="De-DM in pc/cc")
    g.add_argument("--dedm_samps", type=int, help="De-DM in sample units", default=0)
    parser.add_argument("--plot_blocks", action='store_true', help="Plot the blocks after each pre-processing steps", default=False)
    parser.add_argument("--ogif", type=str, help="Name (path) of the output gif. Don't specify if you don't want to save a gif", default=None)
    parser.add_argument("--ofits", type=str, help="Name of the output fits file. Don't specify if you don't want to save a fits", default=None)
    parser.add_argument("--stats_image", action='store_true', help="Generate a mean and rms image out of the data too",default=False)
    args = parser.parse_args()

    assert args.tx > 0, "Tx has to be > 0"
    if args.tx != 1:
        assert args.tx % 2 == 0, "I can only handle averaging by a multiple of 2 at the moment"

    return args


def main():
    args = get_parser()
    #del args
    #values = craco_plan.get_parser().parse_args()#["-u {0}".format(args.uvfits)])
    values = args
    #values.uv = myfits
    values.nt = args.nt
    values.ndm = 2
    ss = 0
    tmp = uvfits.open(values.uv)
    if args.seek_s:
        tsamp = tmp.tsamp
        ss = int(args.seek_s / tsamp)
    elif args.seek_samps:
        ss = int(args.seek_samps)

    assert ss < tmp.vis.size, f"Can't seek to {ss} samps which is beyond the nsamps in file {tmp.size}"
    del tmp
    uvsource = uvfits.open(values.uv, skip_blocks=ss)

    py_plan = craco_plan.PipelinePlan(uvsource, values)

    if args.injection_params_file:
        from Visibility_injector import inject_in_fake_data as VI
        block_type=np.ndarray
    else:
        block_type = np.ma.core.MaskedArray
    #block_type = np.ndarray
    #gridder_obj = FdmtGridder(uvsource, py_plan, values)
    direct_gridder = Gridder(uvsource, py_plan, values)
    imager_obj = Imager(uvsource, py_plan, values)
    if args.dedm_pccc:
        brute_force_dedipserser = preprocess.Dedisp(freqs = py_plan.freqs, tsamp = py_plan.tsamp_s.value, baseline_order = py_plan.baseline_order, dm_pccc=args.dedm_pccc)
        dm_samps = brute_force_dedipserser.dm
        dm_pccc = brute_force_dedipserser.dm_pccc
    elif args.dedm_samps:
        brute_force_dedipserser = preprocess.Dedisp(freqs = py_plan.freqs, tsamp = py_plan.tsamp_s.value, baseline_order = py_plan.baseline_order, dm_samps=args.dedm_samps)
        dm_samps = brute_force_dedipserser.dm
        dm_pccc = brute_force_dedipserser.dm_pccc
    else:
        dm_samps = 0
        dm_pccc = 0


    if args.calibration:
        calibrator = preprocess.Calibrate(plan = py_plan, block_dtype=block_type, miriad_gains_file=args.calibration, baseline_order=py_plan.baseline_order)
    
    if args.rfi or args.flag_chans:
        rfi_cleaner = preprocess.RFI_cleaner(block_dtype=block_type, baseline_order=py_plan.baseline_order)

    if args.ofits or args.stats_image:
        useful_info = {
            'DM_samps': dm_samps,
            'DM_pccc': dm_pccc,
            'NCHAN': py_plan.nf,
            'TSAMP': py_plan.tsamp_s.value * args.tx,
            'Tx': args.tx,
            'FCH1_Hz': py_plan.freqs[0],
            'CH_BW_Hz': py_plan.foff,
            'NANT': py_plan.nant,
            'NBL': py_plan.nbl,
            'STARTMJD': py_plan.tstart.mjd,
            'TARGET': py_plan.target_name,
            'RFI_cl':args.rfi,
            'NORM': args.norm,
            'BTYPE': block_type.__name__,
            'NT': args.nt,
            'CALFILE': str(args.calibration),
            'INJFILE': str(args.injection_params_file),
            'UV': str(py_plan.values.uv),
            'PROPER': str(args.proper),

            'BSCALE': 1.0,
            'BZERO': 0.0,
            'BUNIT': "UNCALIB"
            }

        img_handler = postprocess.ImageHandler(outname=args.ofits, wcs = py_plan.wcs, im_shape=(1, py_plan.npix, py_plan.npix), dtype=np.dtype('>f4'), useful_info = useful_info)
        next_image = np.zeros((py_plan.npix, py_plan.npix))
        img_t_index = 0

    if args.injection_params_file:
        FV = VI.FakeVisibility(plan=py_plan, injection_params_file=args.injection_params_file, outblock_type=dict)
        uvdata_source = FV.get_fake_data_block()
    else:
        if args.proper:
            uvdata_source = uvsource.time_blocks_with_uvws(py_plan.nt)
        else:
            uvdata_source = uvsource.time_blocks(py_plan.nt)

    if args.ogif:
        images = []
    if args.stats_image:
        Ai = np.zeros((py_plan.npix, py_plan.npix))
        Qi = np.zeros((py_plan.npix, py_plan.npix))
        N = 1
    try:
        for iblock, block in enumerate(uvdata_source):
            if args.proper:
                block, uvws = block
                print(f"Splitting the block into uvws and data block of types - {type(uvws), type(block)}")
                
            
            print(f"Working on block {iblock}, dtype={type(block)}")
            #IPython.embed()
            if type(block) == dict and block_type != dict:
                block = bl2array(block)

            assert type(block) == block_type, f"On no... I need blocks to be of type {block_type}, got {type(block)}"

            
            if args.plot_blocks:
                plot_block(block, title="The raw input block")
            if args.calibration:
                block = calibrator.apply_calibration(block)
                if args.plot_blocks:
                    plot_block(block, title="The calibrated block")
            if args.flag_chans:
                block = rfi_cleaner.flag_chans(block, args.flag_chans, 0)
            if args.rfi:
                block, _, _, _ = rfi_cleaner.run_IQRM_cleaning(block, False, False, False, False, True, True)
                if args.plot_blocks:
                    plot_block(block, title="The cleaned block")

            if args.norm:
                block = preprocess.normalise(block, target_input_rms=values.target_input_rms)
                if args.plot_blocks:
                    plot_block(block, title="The normalised block")

            block = preprocess.average_pols(block, keepdims=False)
            if args.plot_blocks:
                plot_block(block, title="The pol-averaged block")

            if dm_samps > 0:
                block = brute_force_dedipserser.dedisperse(iblock, block)
                if args.plot_blocks:
                    plot_block(block, title="The dedispersed block")

            if args.proper:
                gridded_block = direct_gridder.grid_with_uvws(block, uvws)
            else:
                gridded_block = direct_gridder(block)
            for t in range(py_plan.nt // 2):
                imgout = imager_obj(np.fft.fftshift(gridded_block[..., t])).astype(np.complex64)
                if args.stats_image:
                    Qi = Qi + (N -1)/N * (imgout.real - Ai)**2
                    Ai = Ai + (imgout.real - Ai)/N
                    N+=1

                    Qi = Qi + (N -1)/N * (imgout.imag - Ai)**2
                    Ai = Ai + (imgout.imag - Ai)/N
                    N+=1

                    mean_image = Ai
                    rms_image = np.sqrt(Qi/N)

                if args.ofits is not None:
                    if args.tx == 1:
                        img_handler.put_new_frames([imgout.real])
                        img_handler.put_new_frames([imgout.imag])
                    else:
                        img_t_index = iblock * py_plan.nt + 2*t
                        
                        if img_t_index > 0 and img_t_index % args.tx == 0:
                            img_handler.put_new_frames([next_image / args.tx])
                            next_image = np.zeros((py_plan.npix, py_plan.npix))
                        
                        next_image += imgout.real
                        next_image += imgout.imag


                if args.ogif is not None:
                    images.append(make_PIL_images_from_array(imgout.real))
                    images.append(make_PIL_images_from_array(imgout.imag))
            
            if args.ogif is not None:
                print(f"Saving GIF as {args.ogif} now")
                images[0].save(args.ogif, save_all = True, append_images = images[1:], duration=2, loop = 0)
            '''
            print(f"Running c.prepare(block) on iblock: {iblock}")
            #IPython.embed()
            block = c.prepare(block)
            print("Running c.fdmt(block)")
            fdmt_out = c.fdmt(block)
            print("Now running the gridder and imager")
            
            for idm in range(1):
                for t in range(c.plan.nt //2):
                    print(f"Starting the gridder for t = {t}")
                    gridout = gridder_obj(idm, t, fdmt_out)
                    print(f"Starting the imager for t = {t}")
                    imgout = imager_obj(np.fft.fftshift(gridout)).astype(np.complex64)
                    print("Done")
                    
                    #plt.figure()
                    #plt.imshow(imgout.real, aspect='auto', interpolation="None")
                    #plt.title(f"isamp {iblock * py_plan.nt + 2*t}")
                    #plt.figure()
                    #plt.imshow(imgout.imag, aspect='auto', interpolation="None")
                    #plt.title(f"isamp {iblock * py_plan.nt + 2*t + 1}")
                    #plt.show()
                    
                    if args.ofits is not None:
                        img_handler.put_new_frames([imgout.real])
                        img_handler.put_new_frames([imgout.imag])

                    if args.ogif is not None:
                        images.append(make_PIL_images_from_array(imgout.real))
                        images.append(make_PIL_images_from_array(imgout.imag))

            if args.ogif is not None:
                print(f"Saving GIF as {args.ogif} now")
                images[0].save(args.ogif, save_all = True, append_images = images[1:], duration=2, loop = 0)
            '''
        
    except KeyboardInterrupt as KE:
        print("Caught keyboard interrupt -- closing the files")
        if args.ofits and args.tx > 0 and img_t_index > 0 and (2*(img_t_index)) % args.tx > 0:
            print("Appedning the partially averaged frame in the end")
            img_handler.put_new_frames([next_image / (img_t_index % args.tx)])
        pass
    finally:
        if args.ofits:
            img_handler.close()
        if args.stats_image:
            
            if args.injection_params_file:
                mean_image_fname = args.injection_params_file + ".mean_image.fits"
                rms_image_fname = args.injection_params_file + ".rms_image.fits"
            else:
                mean_image_fname = args.ofits + ".mean_image.fits"
                rms_image_fname = args.ofits + ".rms_image.fits"
            print("Saving the stats images in: {0} and {1}".format(mean_image_fname, rms_image_fname))
            useful_info['NSUMMED'] = N
            postprocess.create_header_for_image_data(mean_image_fname, wcs = py_plan.wcs, im_shape = (py_plan.npix, py_plan.npix), dtype=np.dtype('>f4'), kwargs = useful_info, image_data = mean_image)
            postprocess.create_header_for_image_data(rms_image_fname, wcs = py_plan.wcs, im_shape = (py_plan.npix, py_plan.npix), dtype=np.dtype('>f4'), kwargs = useful_info, image_data = rms_image)
            

if __name__ == '__main__':
    main()

    








