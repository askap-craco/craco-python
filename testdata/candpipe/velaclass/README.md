# About this
First 1000 candiadtes from vela at the phase center.

/CRACO/DATA_03/craco/SB062401/scans/00/20240515090811/candidates.b20.txt

To check that general mechanisms of loading teh right catalog work.

The WCS for this beam got from teh log file ws was as follwos

Number of WCS axes: 2
CTYPE : 'RA---SIN'  'DEC--SIN'  
CRVAL : 128.833333355614  -45.17638895171388  
CRPIX : 129.0  129.0  
PC1_1 PC1_2  : 1.0  0.0  
PC2_1 PC2_2  : 0.0  1.0  
CDELT : -0.004296875  0.004296875  
NAXIS : 0  0 image_params=ImageParams2d lparams=ImageParms1D fov=1.10(1.10)deg os=2.45(2.10) imgpix=15.5 arcsec synth beam=37.9 arcsec uvcell=52.1 lambda mparams=ImageParms1D fov=1.10(1.10)deg os=2.75(2.10) imgpix=15.5 arcsec synth beam=42.5 arcsec uvcell=52.1 lambda
/CRACO/SOFTWARE/ban115/craft/sr


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
