import argparse
import numpy as np
from craft.craco_kernels import Prepare, Gridder, Imager ,CracoPipeline, FdmtGridder
from craft import craco_plan
from craft import uvfits
from craco import preprocess
from craft.craco import bl2array
import matplotlib.pyplot as plt

def plot_block(block):
    f, ax = plt.subplots(2, 2)
    ax[0, 0].imshow(np.abs(block.sum(axis=0).mean(axis=-2)).squeeze(), aspect='auto')
    ax[0, 0].set_title("CAS")
    ax[0, 1].imshow(np.abs(block.real.sum(axis=0).mean(axis=-2)).squeeze(), aspect='auto')
    ax[0, 1].set_title("Real part")
    ax[1, 0].imshow(block[3].mean(axis=-2).real, aspect='auto')
    ax[1, 0].set_title("Basline 3 Real part")
    ax[1, 1].imshow(block[3].mean(axis=-2).imag, aspect='auto')
    ax[1, 1].set_title("Basline 3 Imag part")
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

    
    calibrator = preprocess.Calibrate(block_dtype=np.ma.core.MaskedArray, miriad_gains_file=args.calfile, baseline_order=py_plan.baseline_order)
    rfi_cleaner = preprocess.RFI_cleaner(block_dtype=np.ma.core.MaskedArray, baseline_order=py_plan.baseline_order)

    images = []
    for iblock, block in enumerate(c.uvsource.time_blocks(py_plan.nt)):
        print("Working on block", iblock)
        block = bl2array(block)
        plot_block(block)
        calibrated_block = calibrator.apply_calibration(block)
        plot_block(calibrated_block)
        cleaned_block, _, _, _ = rfi_cleaner.run_IQRM_cleaning(np.abs(calibrated_block), False, False, False, False, True, True)
        plot_block(cleaned_block)
        normalised_block = preprocess.normalise(cleaned_block, target_input_rms=values.target_input_rms)
        plot_block(normalised_block)
        prepared_block = c.prepare(normalised_block)
        fdmt_out = c.fdmt(prepared_block)
        for idm in range(1):
            for t in range(c.plan.nt //2):
                gridout = gridder_obj(idm, t, fdmt_out)
                imgout = imager_obj(np.fft.fftshift(gridout)).astype(np.complex64)
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

    








