import argparse
import numpy as np
from craft.craco_kernels import Prepare, Gridder, Imager ,CracoPipeline, FdmtGridder
from craft import craco_plan
from craft import uvfits
from craco import preprocess
from craft.craco import bl2array
import matplotlib.pyplot as plt

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
    ax[0, 1].set_title("Real part")
    ax[1, 0].imshow(myblock[3].real, aspect='auto')
    ax[1, 0].set_title("Basline 3 Real part")
    ax[1, 1].imshow(myblock[3].imag, aspect='auto')
    ax[1, 1].set_title("Basline 3 Imag part")
    if title:
        plt.title(title)
    plt.show()

def temp_main():
    #del args
    #values = craco_plan.get_parser().parse_args()#["-u {0}".format(args.uvfits)])
    values = args
    #values.uv = myfits
    values.nt = 16
    values.ndm = 2
    uvsource = uvfits.open(values.uv)
    py_plan = craco_plan.PipelinePlan(uvsource, values)

    c = CracoPipeline(values)
    gridder_obj = FdmtGridder(uvsource, py_plan, values)
    imager_obj = Imager(uvsource, py_plan, values)

    brute_force_dedipserser = preprocess.Dedisp(freqs = py_plan.freqs, tsamp = py_plan.tsamp_s.value, baseline_order = py_plan.baseline_order, dm_pccc=478)

    calibrator = preprocess.Calibrate(block_dtype=dict, miriad_gains_file=args.calfile, baseline_order=py_plan.baseline_order)
    rfi_cleaner = preprocess.RFI_cleaner(block_dtype=dict, baseline_order=py_plan.baseline_order)

    images = []
    for iblock, block in enumerate(c.uvsource.time_blocks(py_plan.nt)):
        print("-------------- >  Working on block", iblock, "block_type=", type(block))
        #block = bl2array(block)
        plot_block(block, title="The raw input block")
        block = calibrator.apply_calibration(block)
        #plot_block(block)
        #block, _, _, _ = rfi_cleaner.run_IQRM_cleaning(np.abs(block), False, False, False, False, True, True)
        
        plot_block(block, title="The calibrated block")
        block = preprocess.normalise(block, target_input_rms=values.target_input_rms)
        #for ii, item in block.items():
        #    print(f"The shape of {ii}th baseline is {item.shape}")
        block = preprocess.average_pols(block, keepdims=False)
        
        plot_block(block, title="Fully pre-processed block")

        block = brute_force_dedipserser.dedisperse(iblock, block)
        plot_block(block, title="Plotting the dedispersed block")

        print("Running c.prepare(block)")
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
                plt.figure()
                plt.imshow(imgout.real, aspect='auto', interpolation="None")
                plt.title(f"isamp {2*t}")
                plt.figure()
                plt.imshow(imgout.imag, aspect='auto', interpolation="None")
                plt.title(f"isamp {2*t + 1}")
                plt.show()




def main():
    values = craco_plan.get_parser().parse_args(["--uv{0}".format(args.uvfits)])
    uvsource = uvfits.open(values.uv)
    py_plan = craco_plan(uvsource, values)
    
    calibrator = preprocess.Calibrate(block_dtype=np.ma.core.MaskedArray, miriad_gains_file="Path to file", baseline_order=py_plan.baseline_order)
    rfi_cleaner = preprocess.RFI_cleaner(block_dtype=np.ma.core.MaskedArray, baseline_order=py_plan.baseline_order)
    prepare = Prepare(uvsource, py_plan, values)
    gridder = Gridder(uvsource, py_plan, values)
    imager = Imager(uvsource, py_plan, values)

    images = np.empty((py_plan.nt//2, py_plan.npix, py_plan.npix), dtype=np.float64)
    for iblock, block in enumerate(uvsource.time_blocks(py_plan.nt)):
        preprocessed_block = preprocess(block)
        prepared_block = prepare(preprocessed_block)
        gridded_block = gridder(prepared_block)
        img = imager(gridded_block)
        images[::2] = img.real
        images[1::2] = img.imag

        variability_image = images.std(axis=0)
        mean_integrated_image = images.mean(axis=0)

        np.save(f"images_block{iblock}", images)
        np.save(f"variability_image{iblock}", variability_image)
        np.save(f"mean_integrated_image{iblock}", mean_integrated_image)

if __name__ == '__main__':
    parser = craco_plan.get_parser()
    parser.add_argument("-cf", "--calfile", type=str, help="Path to the calibration file")
    args = parser.parse_args()
    temp_main()

    







