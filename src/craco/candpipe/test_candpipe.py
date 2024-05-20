#!/usr/bin/env python
"""
Run tests for candpipe
Copyright (C) CSIRO 2022
"""
import pylab
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import sys
import logging
import pytest
import yaml
from craco.candpipe.candpipe import *
from craco.plot_cand import load_cands
from astropy.io import fits

log = logging.getLogger(__name__)

__author__ = "Keith Bannister <keith.bannister@csiro.au>"

@pytest.fixture
def config():
    config_file = './testdata/candpipe/config.yaml'
    with open(config_file, 'r') as yaml_file:
        c = yaml.safe_load(yaml_file)

    return c

def test_beam_from_candfile():
    f = '/data/craco/craco/wan348/benchmarking/SB61584.n500.candidates.b24.txt'
    assert beam_from_cand_file(f) == 24

    assert beam_from_cand_file('candidates.b24.txt') == 24

def test_candpipe_runs_noalias(config):
    parser = get_parser()
    cand_fname = 'testdata/candpipe/pulsar/SB61584.n500.candidates.b24.txt'

    args = parser.parse_args([cand_fname])
    pipe = Pipeline.from_candfile(cand_fname, args, config)
    cands = pipe.run()

    # Yuanming writes something that in the end does
    example_cands = pd.read_csv('testdata/candpipe/pulsar/SB61584.n500.candidates.b24.txt.uniq.csv')
    assert check_identical(cands, example_cands) == True, f'Missing candidates'
    assert check_identical(example_cands, cands) == True, f'Extra candidates'

def test_candpipe_runs_anti_alias(config):
    parser = get_parser()
    cand_fname = 'testdata/candpipe/alias/candidates.b25.txt'
    args = parser.parse_args([cand_fname])
    pipe = Pipeline(cand_fname, args, config, src_dir=None, anti_alias=True)
    assert len(pipe.steps) == 5 #check its actually runnign the anti aliasing
    cands = pipe.run()

    # Yuanming writes something that in the end does
    # example_cands = pd.read_csv('testdata/candpipe/alias/SB61585.no_alias_filtering.candidates.b25.txt.uniq.csv')
    example_cands = pd.read_csv('testdata/candpipe/alias/candidates.b25.txt.uniq.csv')
    assert check_identical(cands, example_cands, keyname='ALIAS_name') == True, f'Missing candidates'
    assert check_identical(example_cands, cands, keyname='ALIAS_name') == True, f'Extra candidates'

def test_candpipe_runs_anti_alias_nofile(config):
    parser = get_parser()
    cand_fname = 'testdata/candpipe/alias/candidates.b25.txt'
    args = parser.parse_args([cand_fname, '--verbose'])
    beam=25
    pipe = Pipeline(beam, args, config, src_dir=None, anti_alias=True)
    pipe2 = Pipeline(cand_fname, args, config, src_dir=None, anti_alias=True)

    hdr = fits.getheader('testdata/candpipe/alias/psf.beam25.iblk0.fits')
    pipe.set_current_psf(0,hdr)
    assert pipe.psf_header == pipe2.psf_header
    assert pipe.get_current_fov() == pipe2.get_current_fov()
    assert len(pipe.steps) == 5 #check its actually runnign the anti aliasing
    assert len(pipe.steps[2].catalogs) >= 0, 'Catalogs should have been loaded!'

    input_cands = load_cands(cand_fname, fmt='pandas')
    cands = pipe.process_block(input_cands)

    check_steps_identical(pipe, pipe2, input_cands)
    # Yuanming writes something that in the end does
    # example_cands = pd.read_csv('testdata/candpipe/alias/SB61585.no_alias_filtering.candidates.b25.txt.uniq.csv')
    example_cands = pd.read_csv('testdata/candpipe/alias/candidates.b25.txt.uniq.csv')
    assert check_identical(cands, example_cands, keyname='ALIAS_name') == True, f'Missing candidates'
    assert check_identical(example_cands, cands, keyname='ALIAS_name') == True, f'Extra candidates'

def check_steps_identical(pipe, pipe2, input_cands):
    din1 = input_cands
    din2 = input_cands

    # Check steps identical. For debugging.
    for s1,s2 in zip(pipe.steps, pipe2.steps):
        dout1 = s1(pipe, din1)
        dout2 = s2(pipe2, din2)
        assert len(dout1) == len(dout2)
        assert dout1.equals(dout2)
        din1 = dout1
        din2 = dout2




def check_identical(data1, data2,  keyname='PSR_name'):
    '''
    # # check number of candidates in both files
    # snr = 8
    
    # for ind, candfile in enumerate(candfiles):
    #     candfile1 = candfile.replace('test2', 'test1')
    #     candfile2 = candfile
    #     print()
    #     print('cand1', candfile1)
    #     print('cand2', candfile2)
    
    #     cand1 = read_file(candfile1, snr=snr)
    #     cand2 = read_file(candfile2, snr=snr)
    
    #     print(ind, len(cand1), len(cand2))
    #     print('check if any cand1 not in cand2')
    #     check_identical(cand1, cand2)
    #     print('check if any cand2 not in cand1')
    #     check_identical(cand2, cand1)
    '''

    for i in range(len(data1)):

        lpix = data1.iloc[i]['lpix']
        mpix = data1.iloc[i]['mpix']
        tsamp = data1.iloc[i]['total_sample']
        dm = data1.iloc[i]['dm']
        iblk = data1.iloc[i]['iblk']
        category = data1.iloc[i][keyname]

        if category is None or pd.isna(category):
            print(tsamp, category, 'None or nan')
            ind = sum( (data2['lpix'] == lpix) & \
                        (data2['mpix'] == mpix) & \
                        (data2['total_sample'] == tsamp) & \
                        (data2['dm'] == dm) )
        else:
            ind = sum( (data2['lpix'] == lpix) & \
                        (data2['mpix'] == mpix) & \
                        (data2['total_sample'] == tsamp) & \
                        (data2['dm'] == dm) & \
                        (data2[keyname] == category) )

        cluster_id = data1.iloc[i]['cluster_id']

        if ind == 0:
            print('cand1 is not in cand2, cluster_id', cluster_id, 'total_sample', tsamp)
            return False

    return True 

    # assert check_identical(cands, other_cands) == True, f'Candi9dates not identical'

def cand_blocker(cands):
    if isinstance(cands, str):
        cands = load_cands(cands)
    
    maxblk = max(cands['iblk'])
    for iblk in range(maxblk+1):
        yield cands[cands['iblk'] == iblk]

def test_convert_np_to_df(config):
    parser = get_parser()
    cand_fname = 'testdata/candpipe/super_scattered_frb/candidates.b04.txt'
    args = parser.parse_args([cand_fname])
    pipe = Pipeline(cand_fname, args, config, src_dir=None, anti_alias=True)
    cands = load_cands(cand_fname)
    df = pipe.convert_np_to_df(cands)
    assert len(df) == len(cands)

def test_load_default_config():
    assert load_default_config() != None


def test_candpipe_block_by_block(config):
    parser = get_parser()
    cand_fname = 'testdata/candpipe/super_scattered_frb/candidates.b04.txt'
    args = parser.parse_args([])
    beamno = 4
    pipe = Pipeline(beamno, None, config, src_dir='testdata/candpipe/super_scattered_frb/', anti_alias=True)
    cands = load_cands(cand_fname)

    #all_clustered_cands = [pipe.process_block(cblk) for cblk in cand_blocker(cands)]
    all_clustered_cands = [pipe.process_block(cblk) for cblk in cand_blocker(cands)]
    for c in all_clustered_cands:
        assert isinstance(c, pd.DataFrame)

    all_clustered_cands = pd.concat(all_clustered_cands)
    

    assert len(all_clustered_cands) >= 1, 'Expected at least 1 candidate '
    assert len(all_clustered_cands) < len(cands), 'Should have been less candidates after pipeline!'

def test_copy_best_cand():
    cand_fname = 'testdata/candpipe/super_scattered_frb/candidates.b04.txt'
    cands_np = load_cands(cand_fname)
    cands_idx = np.argsort(cands_np, order='snr')
    cands_np_sorted = cands_np[cands_idx[::-1]]
    N = len(cands_np)
    beamno = 4
    pipe = Pipeline(beamno, None, None, src_dir='testdata/candpipe/super_scattered_frb/', anti_alias=False)
    cands_df = pipe.convert_np_to_df(cands_np)
    cands_df_sorted = cands_df.sort_values(by='snr', ascending=False)
    NMAX = 8
    cands_np2 = np.zeros(NMAX, dtype=cands_np.dtype)

    # check if there's more candidates than NMAX it fills everythign
    assert len(cands_df) > NMAX
    copy_best_cand(cands_df, cands_np2)

    # because sorting isn't stable, you can't compare them exactly if the SNRs are equal
    # so we just assert that we got the SNRs right
    assert np.all(cands_np2['snr'] == cands_np_sorted[:NMAX]['snr'])

    # check if there's less candidates than NMAX it puts -1 in the remainder
    short_df = cands_df.iloc[:3]
    copy_best_cand(short_df, cands_np2)
    assert np.all(cands_np2['snr'][:3] == short_df.sort_values(by='snr', ascending=False)['snr'])
    assert np.all(cands_np2['snr'][3:] == -1)

    # check if there's none, everything is -1
    copy_best_cand(short_df.head(0), cands_np2)
    assert np.all(cands_np2['snr'][3:] == -1)

def test_candpipe_len0(config):
    parser = get_parser()
    cand_fname = 'testdata/candpipe/super_scattered_frb/candidates.b04.txt'
    args = parser.parse_args([])
    beamno = 4
    pipe = Pipeline(beamno, None, config, src_dir='testdata/candpipe/super_scattered_frb/', anti_alias=True)
    cands = load_cands(cand_fname)
    cblk = cands[:0] # make length 0 array
    dout = np.zeros(8, dtype=CandidateWriter.out_dtype)
    cands_df = pipe.process_block(cblk, dout)
    assert len(cands_df) == 0
    # Shoudl also have additional column




def test_candpipe_block_by_block_np(config):
    parser = get_parser()
    cand_fname = 'testdata/candpipe/super_scattered_frb/candidates.b04.txt'
    args = parser.parse_args([])
    beamno = 4
    pipe = Pipeline(beamno, None, config, src_dir='testdata/candpipe/super_scattered_frb/', anti_alias=True)
    cands = load_cands(cand_fname)

    dout = np.zeros(8, dtype=CandidateWriter.out_dtype)

    #all_clustered_cands = [pipe.process_block(cblk) for cblk in cand_blocker(cands)]
    #all_clustered_cands = [pipe.process_block(cblk, dout) for cblk in cand_blocker(cands)]
    for iblk, cblk in enumerate(cand_blocker(cands)):
        cands_df = pipe.process_block(cblk, dout)        
        cands_df = filter_df_for_unknown(cands_df)
        n = len(cands_df)
        nout = min(n, len(dout))
        best_df = cands_df.sort_values(by='snr', ascending=False)
        
        if n > 0:
            print('hello', n)
        assert np.all(best_df.iloc[:n]['snr'] == dout[:n]['snr'])

        print(iblk, n, dout['snr'])
        if not np.all(dout[n:]['snr'] == -1):
            print(iblk, n, dout[n:]['snr'])

        assert np.all(dout[n:]['snr'] == -1)

def test_candpipe_missing_clasifications(config):
    ''' See https://jira.csiro.au/browse/CRACO-244
    # contains piles of candidates that the candpipe seems to have not classified as vela
    # LIke this one
    # ban115@skadi-00:/CRACO/DATA_03/craco/SB062401/scans/00/20240515090811/test$ grep 257 ../candidates.b20.txt  | grep ^103
    #103.4	128	128	0	1	14	1	3325	257	3.5528	60445.381749111	71.365	128.83333	-45.17639	20	4487.5
    
    Also: there was a problem where if the initial blcok was empty there was no catalog loaded. That's been fixed. This dataset tests it.


    # From the log file
    # 

Number of WCS axes: 2
CTYPE : 'RA---SIN'  'DEC--SIN'  
CRVAL : 128.833333355614  -45.17638895171388  
CRPIX : 129.0  129.0  
PC1_1 PC1_2  : 1.0  0.0  
PC2_1 PC2_2  : 0.0  1.0  
CDELT : -0.004296875  0.004296875  
NAXIS : 0  0
    '''

    cand_fname = 'testdata/candpipe/velaclass/candidates.b20.txt'
    hdr = fits.header.Header()
    hdr['CRVAL1'] = 128.833333355614
    hdr['CRVAL2'] =  -45.17638895171388 
    hdr['CRPIX1'] = 129
    hdr['CRPIX2'] = 129
    hdr['CDELT1'] =  -0.004296875
    hdr['CDELT2'] = 0.004296875  
    hdr['NAXIS1'] = 256
    hdr['NAXIS2'] = 256
    
    # Guess
    hdr['FCH1_HZ'] = 850e6
    hdr['CH_BW_HZ'] = 1e6
    hdr['NCHAN'] = 288
    hdr['TSAMP'] = 13.7e-3


    parser = get_parser()
    args = parser.parse_args([])
    beamno = 4
    pipe = Pipeline(beamno, None, config, src_dir='.', anti_alias=True)
    iblk=0
    pipe.set_current_psf(iblk, hdr)

    if not os.path.exists(cand_fname):
        # ONly check this on tethys - that file is too big to checkin
        return
    cands = load_cands(cand_fname)

    dout = np.zeros(8, dtype=CandidateWriter.out_dtype)

    #all_clustered_cands = [pipe.process_block(cblk) for cblk in cand_blocker(cands)]
    #all_clustered_cands = [pipe.process_block(cblk, dout) for cblk in cand_blocker(cands)]
    for iblk, cblk in enumerate(cand_blocker(cands)):
        cands_df = pipe.process_block(cblk, dout)
        if len(cands_df) != 0 and iblk == 1:
            thecand = cands_df[cands_df['total_sample'] == 257]
            if len(thecand) == 1:                             
                thecand = thecand.iloc[0]
                assert thecand['PSR_name'] == 'J0835-4510'
        
        if iblk >= 1:
            break


def test_candpipe_dump_output(config):
    test_df = pd.read_csv("testdata/candpipe/pulsar/SB61584.n500.candidates.b24.txt.uniq.csv", index_col=0)
    beamno = 24
    args = get_parser().parse_args(['--outdir', 'clustering_output'])
    if not os.path.exists(args.outdir):
        os.mkdir(args.outdir)
    candpipe_obj = Pipeline(beamno, args, config, anti_alias=True)
    '''
    t = np.arange(5)
    candpipe_obj.uniq_cands_fout.write_cands(t)
    rt = np.load(candpipe_obj.uniq_cands_fout.outname)
    assert np.allclose(t, rt)
    '''
    #'''
    for icands in cand_blocker(test_df):
        npy_cands = candpipe_obj.convert_df_to_np(icands)
        candpipe_obj.uniq_cands_fout.write_cands(npy_cands)

    candpipe_obj.uniq_cands_fout.close()
    rx_cands = np.load(candpipe_obj.uniq_cands_fout.outname)
    in_cands = test_df
    in_rows = [in_cands.iloc[0], in_cands.iloc[-1]]
    out_rows = [rx_cands[0], rx_cands[-1]]

    for icand, ocand in zip(in_rows, out_rows):
        for ii in range(len(icand)):
            if test_df.dtypes.iloc[ii] == np.dtype('O'):
                assert str(icand.iloc[ii]) == str(ocand[ii])
            else:
                assert np.isclose(icand.iloc[ii], ocand[ii], equal_nan=True), f"{ii} \n {icand} \n {ocand}"

    #'''

def test_candpipe_intermediate_test_output(config):
    cand_fname = 'testdata/candpipe/super_scattered_frb/candidates.b04.txt'
    test_df = load_cands(cand_fname)
    beamno = 24
    args = get_parser().parse_args(['--outdir', 'clustering_output', '-s'])
    if not os.path.exists(args.outdir):
        os.mkdir(args.outdir)
    candpipe_obj = Pipeline(beamno, args, config, anti_alias=False)

    for ii, cand_block in enumerate(cand_blocker(test_df)):
        cand_out = candpipe_obj.process_block(cand_block)

    candpipe_obj.close()


def _main():
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
    parser = ArgumentParser(description='Script description', formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('-v', '--verbose', action='store_true', help='Be verbose')
    parser.add_argument(dest='files', nargs='+')
    parser.set_defaults(verbose=False)
    values = parser.parse_args()
    if values.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)
    

if __name__ == '__main__':
    _main()
