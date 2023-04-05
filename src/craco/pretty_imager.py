import argparse
import numpy as np
from craft.craco_kernels import Prepare, Gridder, Imager ,CracoPipeline, FdmtGridder
from craft import craco_plan
from craft import uvfits
from craco import preprocess, postprocess
from craft.craco import bl2array
import matplotlib.pyplot as plt
from Visibility_injector import inject_in_fake_data as VI
import IPython
from PIL import Image


def make_PIL_images_from_array(arr, cmap=None):

    def normalise_image(image):
        image -= np.min(image)
        image = image / np.max(image)
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
    ax[0, 0].imshow(np.abs(myblock.sum(axis=0)).squeeze(), aspect='auto')
    ax[0, 0].set_title("CAS")
    ax[0, 1].imshow(np.abs(myblock.real.sum(axis=0)).squeeze(), aspect='auto')
    ax[0, 1].set_title("Real part of CAS")
    ax[1, 0].imshow(myblock[3].real, aspect='auto')
    ax[1, 0].set_title("Baseline 3 Real part")
    ax[1, 1].imshow(myblock[3].imag, aspect='auto')
    ax[1, 1].set_title("Baseline 3 Imag part")
    if title:
        f.suptitle(title)
    plt.show()

def main():
    #del args
    #values = craco_plan.get_parser().parse_args()#["-u {0}".format(args.uvfits)])
    values = args
    #values.uv = myfits
    values.nt = args.nt
    values.ndm = 2
    uvsource = uvfits.open(values.uv)
    py_plan = craco_plan.PipelinePlan(uvsource, values)

    block_type = dict
    c = CracoPipeline(values)
    #gridder_obj = FdmtGridder(uvsource, py_plan, values)
    direct_gridder = Gridder(uvsource, py_plan, values)
    imager_obj = Imager(uvsource, py_plan, values)
    brute_force_dedipserser = preprocess.Dedisp(freqs = py_plan.freqs, tsamp = py_plan.tsamp_s.value, baseline_order = py_plan.baseline_order, dm_samps=args.dedm)

    if args.calfile:
        calibrator = preprocess.Calibrate(block_dtype=block_type, miriad_gains_file=args.calfile, baseline_order=py_plan.baseline_order)
    
    rfi_cleaner = preprocess.RFI_cleaner(block_dtype=block_type, baseline_order=py_plan.baseline_order)
    if args.ofits:
        useful_info = {
            'DM_samps': brute_force_dedipserser.dm,
            'TARGET': 'FAKE',
            'BSCALE': 1.0,
            'BZERO': 0.0,
            'BUNIT': "UNCALIB"
            }

        img_handler = postprocess.ImageHandler(outname=args.ofits, wcs = py_plan.wcs, im_shape=(py_plan.npix, py_plan.npix), dtype=np.dtype('>f4'), useful_info = useful_info)

    if args.injection_params_file:
        FV = VI.FakeVisibility(plan=py_plan, injection_params_file=args.injection_params_file, outblock_type=dict)
        uvdata_source = FV.get_fake_data_block()
    else:
        uvdata_source = c.uvsource.time_blocks(py_plan.nt)

    images = []
    try:
        for iblock, block in enumerate(uvdata_source):
            print("Working on block", iblock)

            if type(block) == dict and block_type != dict:
                block = bl2array(block)

            assert type(block) == block_type, f"On no... I need blocks to be of type {block_type}, got {type(block)}"

            
            #--plot_block(block, title="The raw input block")
            if args.calfile:
                block = calibrator.apply_calibration(block)
            #--plot_block(block, title="The calibrated block")
            #block, _, _, _ = rfi_cleaner.run_IQRM_cleaning(np.abs(block), False, False, False, False, True, True)
            #plot_block(block, title="The cleaned block")

            block = preprocess.normalise(block, target_input_rms=values.target_input_rms)
            block = preprocess.average_pols(block, keepdims=False)

            #--plot_block(block, title="Fully pre-processed block")

            if args.dedm > 0:
                block = brute_force_dedipserser.dedisperse(iblock, block)
                plot_block(block, title="Plotting the dedispersed block")

            gridded_block = direct_gridder(block)
            for t in range(c.plan.nt // 2):
                imgout = imager_obj(np.fft.fftshift(gridded_block[..., t])).astype(np.complex64)
                if args.ofits is not None:
                    img_handler.put_new_frames([imgout.real])
                    img_handler.put_new_frames([imgout.imag])
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
        pass
    finally:
        if args.ofits is not None:
            img_handler.close()


if __name__ == '__main__':
    parser = craco_plan.get_parser()
    parser.add_argument("-cf", "--calfile", type=str, help="Path to the calibration file")
    parser.add_argument("--injection_params_file", type=str, help="Path to an injection params file")
    parser.add_argument("-nt", type=int, help="nt for the block size (def = 64)", default=64)
    parser.add_argument("-dedm", type=int, help="De-DM in sample units (def = 0)", default=0)
    parser.add_argument("-ogif", type=str, help="Name (path) of the output gif. Don't specify if you don't want to save a gif", default=None)
    parser.add_argument("-ofits", type=str, help="Name of the output fits file. Don't specify if you don't want to save a fits", default=None)
    args = parser.parse_args()
    main()

    








