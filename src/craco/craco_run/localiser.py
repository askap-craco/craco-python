import numpy as np
import matplotlib.pyplot as plt

from astropy.coordinates import SkyCoord 
from astropy.wcs import WCS
from astropy.nddata import Cutout2D
from astropy.table import Table
from astropy import units
from astropy.io import fits
from astropy.visualization import ZScaleInterval, ImageNormalize

from astroquery.utils.tap.core import Tap, TapPlus
from astroquery.vizier import Vizier

import os

import logging
log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def compare_survey(
        target_coord, reference_coord, 
        target_ra_err=None, target_dec_err=None, 
        reference_ra_err=None, reference_dec_err=None, 
        survey='', radius=5*units.arcsec, ax=None, floor=0
    ):
    
    idx, sep, _ = target_coord.match_to_catalog_sky(reference_coord)

    ind1 = sep < radius
    
    target_match_coord = target_coord[ind1]
    reference_match_coord = reference_coord[idx][ind1]
    
    # check if they're unique matches
#     print(np.unique(idx[ind1]).shape[0])
#     print(sum(ind1))
    
    dra, ddec = target_match_coord.spherical_offsets_to(reference_match_coord)
    dra, ddec = dra.arcsec, ddec.arcsec

    if target_ra_err is not None and target_dec_err is not None and reference_ra_err is not None and reference_dec_err is not None:
        xerr = np.sqrt(target_ra_err[ind1]**2 + reference_ra_err[idx][ind1]**2 + floor**2)
        yerr = np.sqrt(target_dec_err[ind1]**2 + reference_dec_err[idx][ind1]**2 + floor**2)
        
        ax.errorbar(dra, ddec, xerr=xerr, yerr=yerr, fmt='o', alpha=0.4, c='black')

        # weights! 
        xw, yw = 1/xerr**2, 1/yerr**2
        dra_mean = np.average(dra, weights=xw)
        ddec_mean = np.average(ddec, weights=yw)

        dra_rms = np.sqrt((1 / np.sum(xw))**2 * np.sum( xw**2 * xerr**2 ))
        ddec_rms = np.sqrt((1 / np.sum(yw))**2 * np.sum( yw**2 * yerr**2 ))
        
#         print(dra.shape[0] - 1)
#         print(dra_rms, np.sum(1/xerr**2 * dra**2), np.sum(1/xerr**2), dra_mean)
        

    else:
        ax.scatter(dra, ddec, marker='o', alpha=0.4, c='black')
        dra_mean, ddec_mean = dra.mean(), ddec.mean()
        dra_rms, ddec_rms = dra.std(), ddec.std()
#         print(dra_mean, ddec_mean, dra_rms, ddec_rms)
        
    ax.axvline(dra_mean, color='red', ls='--', label=r'$\bar\Delta RA={:.2f}\pm {:.3f}$"'.format(dra_mean, dra_rms))
    ax.axhline(ddec_mean, color='red', ls='--', label=r'$\bar\Delta DEC={:.2f}\pm {:.3f}$"'.format(ddec_mean, ddec_rms))

    ax.set_xlabel(r'$\Delta$RA')
    ax.set_ylabel(r'$\Delta$DEC')
    ax.set_title(survey)

    ax.legend()

    return dra_mean, ddec_mean, dra_rms, ddec_rms

def compare_position(target_coord, reference_coord, ptype='sep', survey='', radius=5*units.arcsec, ax=None, vmin=None, vmax=None):
    
    idx, sep, _ = target_coord.match_to_catalog_sky(reference_coord)
    
    target_match_coord = target_coord[sep < radius]
    reference_match_coord = reference_coord[idx][sep < radius]
    
    dra, ddec = target_match_coord.spherical_offsets_to(reference_match_coord)
    dra, ddec = dra.arcsec, ddec.arcsec
    
    if ptype == 'ra':
        sm = ax.scatter(target_match_coord.ra, target_match_coord.dec, c=dra, cmap='seismic', alpha=0.8, vmin=vmin, vmax=vmax)
    elif ptype == 'dec':
        sm = ax.scatter(target_match_coord.ra, target_match_coord.dec, c=ddec, cmap='seismic', alpha=0.8, vmin=vmin, vmax=vmax)
    else:
        sm = ax.scatter(target_match_coord.ra, target_match_coord.dec, c=sep.arcsec[sep<radius], cmap='magma_r', alpha=0.8, vmin=vmin, vmax=vmax)
    
    ax.set_title(survey + ' - ' + ptype)
    
    colorbar = plt.colorbar(sm)

    ax.set_xlabel('RA (deg)')
    ax.set_ylabel('DEC (deg)')
    

# calculate weighted average value
def calcu_weighted_average(value_list, err_list):

    var = np.array(np.array(err_list)**2).reshape(-1, 1)
    weight = 1 / var
    value = np.array(value_list).reshape(-1, 1)
    
    mean_out = np.diagonal(np.dot(value.T, weight)) / np.sum(weight, axis=0)
    var_out = np.diagonal(np.dot((var).T, weight**2)) / (np.sum(weight, axis=0)**2)
    return mean_out, np.sqrt(var_out)

# calculate weighted average value, another approach
def measure_weighted_mean(value_list, err_list):

    var = np.array(err_list)**2
    weight = 1 / var
    value = np.array(value_list)
    
    mean_out = np.average(value_list, weights=1/err_list**2)
    var_out = (1 / np.sum(weight))**2 * np.sum( weight**2 * err_list**2 )
    return mean_out, np.sqrt(var_out)

def convert_maj_to_radec(eeMaj, eeMin, PA):
    '''
    infrared catalogue will normally report Major axis fitting error instead of RA error
    '''
    # convert PA from degree to radians 
    PA = PA/180*np.pi

    sigra = np.sqrt( eeMaj**2 * np.sin(PA)**2 + eeMin**2 * np.cos(PA)**2 )
    sigdec = np.sqrt( eeMaj**2 * np.cos(PA)**2 + eeMin**2 * np.sin(PA)**2 )

    return sigra, sigdec

def get_survey_offset(
        target_coord, reference_coord, target_ra_err=None, target_dec_err=None, 
        reference_ra_err=None, reference_dec_err=None, 
        radius=5*units.arcsec, floor=0
    ):
    
    idx, sep, _ = target_coord.match_to_catalog_sky(reference_coord)

    ind1 = sep < radius
    
    target_match_coord = target_coord[ind1]
    reference_match_coord = reference_coord[idx][ind1]
    
    # check if they're unique matches
    # print(np.unique(idx[ind1]).shape[0])
    # print(sum(ind1))
    
    dra, ddec = target_match_coord.spherical_offsets_to(reference_match_coord)
    dra, ddec = dra.arcsec, ddec.arcsec

    xerr = np.sqrt(target_ra_err[ind1]**2 + reference_ra_err[idx][ind1]**2 + floor**2)
    yerr = np.sqrt(target_dec_err[ind1]**2 + reference_dec_err[idx][ind1]**2 + floor**2)

    return dra, ddec, np.array(xerr), np.array(yerr)


def get_racs_catalogue(ra, dec, radius=1., maxtrial=3):
    ntrial = 1
    while ntrial <= maxtrial:
        try:
            tap = TapPlus(url="https://casda.csiro.au/casda_vo_tools/tap")
            job = tap.launch_job_async(f"SELECT * FROM AS110.racs_dr1_gaussians_galacticcut_v2021_08_v02 WHERE 1=CONTAINS(POINT('ICRS', ra, dec),CIRCLE('ICRS', {ra},{dec},{radius}))")
            racs_raw = job.get_results()
            break
        except:
            ntrial += 1
    if ntrial > maxtrial:
        racs_raw = None
    
    return racs_raw

def _get_vizier_wise(centercoord, radius):
    v = Vizier(columns=['AllWISE', 'RAJ2000', 'DEJ2000', 'eeMaj', 'eeMin', 'eePA'], row_limit=-1)
    wise = v.query_region(centercoord, radius=radius*units.degree, catalog='II/328')[0]
    coord = SkyCoord(wise['RAJ2000'], wise['DEJ2000'], unit=units.degree)
    ra_err, dec_err = convert_maj_to_radec(wise['eeMaj'], wise['eeMin'], wise['eePA'])
    return coord, ra_err, dec_err

def _get_vizier_2mass(centercoord, radius):
    v = Vizier(columns=['_2MASS', 'RAJ2000', 'DEJ2000', 'errMaj', 'errMin', 'errPA'], row_limit=-1)
    twomass = v.query_region(centercoord, radius=radius*units.degree, catalog='2MASS')[0]
    coord = SkyCoord(twomass['RAJ2000'], twomass['DEJ2000'], unit=units.degree)
    ra_err, dec_err = convert_maj_to_radec(twomass['errMaj'], twomass['errMin'], twomass['errPA'])
    return coord, ra_err, dec_err
    

def _get_vizier_vlass(centercoord, radius):
    v = Vizier(columns=['CompName', 'RAJ2000', 'DEJ2000', 'e_RAJ2000', 'e_DEJ2000'], row_limit=-1)
    vlass = v.query_region(centercoord, radius=radius*units.degree, catalog='VLASS')[0]
    coord = SkyCoord(vlass['RAJ2000'], vlass['DEJ2000'], unit=units.degree)
    ra_err = vlass["e_RAJ2000"]; dec_err = vlass["e_DEJ2000"]
    return coord, ra_err, dec_err

def vizier_catalogue(ra, dec, survey="WISE", radius=1., maxtrial=3):
    centercoord = SkyCoord(ra, dec, unit=units.degree)
    
    if survey == "WISE": query_func = _get_vizier_wise
    elif survey == "2MASS": query_func = _get_vizier_2mass
    elif survey == "VLASS": query_func = _get_vizier_vlass
        
    ntrial = 1
    while ntrial <= maxtrial:
        try:
            coord, ra_err, dec_err = query_func(centercoord, radius=radius)
            break
        except Exception as error:
            log.warning(f"error in {survey} catalogue downloading... - {error}")
            ntrial += 1
    if ntrial > maxtrial:
        coord = None; ra_err = None; dec_err = None
    
    return coord, ra_err, dec_err
    

def clean_racs_catlogue(racs_raw):
    racs_coord = SkyCoord(racs_raw['ra'], racs_raw['dec'], unit=units.degree)
    _, sep, _ = racs_coord.match_to_catalog_sky(racs_coord, nthneighbor=2)
    compactness = racs_raw['total_flux_gaussian'] / racs_raw['peak_flux']
    snr = racs_raw['peak_flux'] / racs_raw['e_peak_flux']
    racs = racs_raw[(sep > 30*units.arcsec) & (compactness < 1.5) & (snr > 10)].reset_index(drop=True)

    # racs_coord = SkyCoord(racs['ra'], racs['dec'], unit=u.degree)
    # racs_ra_err = racs['e_ra']
    # racs_dec_err = racs['e_dec']

    return racs

def clean_pybdsf_catalogue(cat, threshold=0.003):
    cat = cat[(cat["E_RA"] < threshold) & (cat["E_DEC"] < threshold)].copy()
    ### add snr
    cat["Peak_SNR"] = cat["Peak_flux"] / cat["E_Peak_flux"]
    return cat.reset_index(drop=True)

def get_bootstrap_offset(dra, ddec, raerr, decerr, num=1000):

    mean_ra_list = []; mean_dec_list = []
    
    for i in range(num):
        indlist = np.random.randint(low=0, high=dra.shape[0], size=dra.shape[0])
        mean_ra_list.append(np.average(dra[indlist], weights=1/raerr[indlist]**2))
        mean_dec_list.append(np.average(ddec[indlist], weights=1/decerr[indlist]**2))

    return mean_ra_list, mean_dec_list

def plot_bootstrap_result(ra_offset, dec_offset, title=""):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6, 3), dpi=100)

    ax1.hist(ra_offset)
    ax1.axvline(np.mean(ra_offset), color='red', ls='--', 
                label=r'$\bar\Delta RA={:.2f}\pm{:.2f}$"'.format(np.mean(ra_offset), np.std(ra_offset)))

    ax2.hist(dec_offset)
    ax2.axvline(np.mean(dec_offset), color='red', ls='--', 
                label=r'$\bar\Delta DEC={:.2f}\pm{:.2f}$"'.format(np.mean(dec_offset), np.std(dec_offset)))

    ax1.legend()
    ax2.legend()

    ax1.set_xlabel("RA offset (arcsec)")
    ax2.set_xlabel("DEC offset (arcsec)")

    ax1.set_title(title)
    ax2.set_title(title)
    
    return fig

### plotting fits image...
def fits_cutout(data, wcs, ra, dec, radius=60.):
    centcoord = SkyCoord(ra, dec, unit=units.degree)
    cutoutsize = (2 * radius * units.arcsec, 2 * radius * units.arcsec)
    return Cutout2D(data, centcoord, cutoutsize, wcs=wcs)

def plot_fits_data(data, wcs, norm=None):
    fig = plt.figure(figsize=(4, 4), facecolor="w")
    ax = fig.add_subplot(1, 1, 1, projection=wcs)
    
    if norm is None: norm = ImageNormalize(data, interval=ZScaleInterval())
        
    im = ax.imshow(data, norm=norm, cmap="gray_r", interpolation="none")
    ax.coords[0].set_axislabel("R.A.")
    ax.coords[1].set_axislabel("Decl.")
    
    return fig, ax

class CorrSrcPos:
    def __init__(
        self, fieldfitspath, burstfitspath,
        fieldcatpath, burstcatpath, srcra, srcdec, 
        workdir = "./", summary=""
    ):
        self.fieldfitspath = fieldfitspath
        self.burstfitspath = burstfitspath
        self.fieldcatpath = fieldcatpath
        self.burstcatpath = burstcatpath
        self.coord = (srcra, srcdec)
        self.workdir = workdir
        self.summary = summary

        ### make workdir
        if not os.path.exists(self.workdir):
            os.makedirs(self.workdir)
        
        self._load_data()
        
    def _load_data(self, threshold=0.003):
        self.fieldhdul = fits.open(self.fieldfitspath)
        self.bursthdul = fits.open(self.burstfitspath)
        self.fieldcatraw = Table.read(self.fieldcatpath).to_pandas()
        self.burstcatraw = Table.read(self.burstcatpath).to_pandas()
        
        ### do some initial cleaning...
        self.fieldcat = clean_pybdsf_catalogue(self.fieldcatraw, threshold=threshold)
        self.burstcat = clean_pybdsf_catalogue(self.burstcatraw, threshold=threshold)
        
        self.field_coord = SkyCoord(self.fieldcat["RA"], self.fieldcat["DEC"], unit=units.degree)
        self.field_ra_err = self.fieldcat["E_RA"].to_numpy() * 3600.
        self.field_dec_err = self.fieldcat["E_DEC"].to_numpy() * 3600.
        self.burst_coord = SkyCoord(self.burstcat["RA"], self.burstcat["DEC"], unit=units.degree)
        self.burst_ra_err = self.burstcat["E_RA"].to_numpy() * 3600.
        self.burst_dec_err = self.burstcat["E_DEC"].to_numpy() * 3600.
        
        ### load field center...
        self.centcoord = (self.fieldhdul[0].header['CRVAL1'], self.fieldhdul[0].header['CRVAL2'])
        
    def _get_racs(self, radius=1.0):
        log.info(f"downloading racs catalogue centred at {self.centcoord} with a radius {radius} deg")
        racscat = get_racs_catalogue(*self.centcoord, radius=radius)
        log.info(f"got {len(racscat)} sources from racs catalogue")
        
        if racscat is None:
            log.warning("cannot get racs catalogue... will not do any correction...")
            self.racscat = None; self.racs_ra_err = None; self.racs_dec_err = None
            return
        
        ### if you need to do any correction to the catalogue itself, do it here...
        racs_raw = racscat.to_pandas()
        self.racscat = clean_racs_catlogue(racs_raw)
        log.info(f"got {len(self.racscat)} sources after cleaning...")
        
        self.racs_coord = SkyCoord(self.racscat["ra"], self.racscat["dec"], unit=units.degree)
        self.racs_ra_err = self.racscat["e_ra"].to_numpy()
        self.racs_dec_err = self.racscat["e_dec"].to_numpy()
        
    def _get_ref(self, radius=1.0):
        surveys_order = ["WISE", "2MASS", "VLASS"]
        for survey in surveys_order:
            self.ref_coord, self.ref_ra_err, self.ref_dec_err = vizier_catalogue(
                *self.centcoord, survey=survey, radius=radius, maxtrial=5
            )
            if self.ref_coord is None:
                log.info(f"cannot get {survey} catalogue... try the next one...")
            else:
                log.info(f"use {survey} catalogue as the reference...")
                self.ref_survey = survey # get the reference survey...
                break
        
        if self.ref_coord is None:
            self.ref_survey = None
            log.warning("cannot get reference catalogue... will only correct for the racs offset...")
        
    ### several functions on selecting burst sources, work out the distance, and then determining the radius...
    def _select_sources(self, snrcut=8.):
        srccoord = SkyCoord(*self.coord, unit=units.degree)
        pccoord = SkyCoord(*self.centcoord, unit=units.degree)
        ### calculate the distance
        pc_src_sep = srccoord.separation(pccoord).degree
        log.info(f"separation between the candidate and phase center is {pc_src_sep:.2f} deg")
        sep_thres = 40 * pc_src_sep + 40 # corresponding to 60 arcsec for 0.5 deg
        log.info(f"use {sep_thres} arcsec as the threshold to filter sources...")
        self.nearbycat = self.burstcat[
            (self.burst_coord.separation(srccoord).degree < sep_thres / 3600.) &
            (self.burstcat["Peak_SNR"] >= snrcut)
        ].copy()
        self.nearbycat = self.nearbycat.sort_values("Peak_SNR", ascending=False)
        self.nearbycat = self.nearbycat.reset_index(drop=True)
        log.info(f"found {len(self.nearbycat)} sources within the error circle and brighter than SNR = {snrcut}")
        self.sep_thres = sep_thres
        
    
    
    def _compare_two_surveys(
        self, coord1, coord2, ra_err1=None, dec_err1=None,
        ra_err2 = None, dec_err2 = None,
        title="compare", radius=30*units.arcsec
    ):
        
        ### this is mainly for the dignose plot
        fig, ax = plt.subplots(figsize=(4, 4))
        _ = compare_survey(
            coord1, coord2, ra_err1, dec_err1, 
            ra_err2, dec_err2,
            survey=title, ax=ax, 
            radius=radius
        )
        fig.savefig(f"{self.workdir}/{title}.png", bbox_inches="tight")
        plt.close()
        ### this is for the calculation
        survey_offset = get_survey_offset(
            coord1, coord2, ra_err1, dec_err1, 
            ra_err2, dec_err2,
            radius=radius,
        )
        
        return survey_offset
    
    def _compare_all_surveys(self, nbootstrap=1000):
        default_offset = np.zeros(nbootstrap)
        if self.racs_coord is None:
            self.craco_racs_ra_offset = default_offset.copy()
            self.craco_racs_dec_offset = default_offset.copy()
            self.racs_ref_ra_offset = default_offset.copy()
            self.craco_racs_dec_offset = default_offset.copy()
            return
        
        ### compare between field and racs...
        dra, ddec, raerr, decerr = self._compare_two_surveys(
            self.field_coord, self.racs_coord, self.field_ra_err, self.field_dec_err,
            self.racs_ra_err, self.racs_dec_err, title="field_racs", radius=10*units.arcsec,
        )
        self.craco_racs_ra_offset, self.craco_racs_dec_offset = get_bootstrap_offset(
            dra, ddec, raerr, decerr, num=nbootstrap,
        )
        
        
        if self.ref_coord is None:
            self.racs_ref_ra_offset = default_offset.copy()
            self.racs_ref_dec_offset = default_offset.copy()
            return
        ### comapre between racs and ref...
        dra, ddec, raerr, decerr = self._compare_two_surveys(
            self.racs_coord, self.ref_coord, self.racs_ra_err, self.racs_dec_err,
            self.ref_ra_err, self.ref_dec_err, title="racs_ref", radius=2.5*units.arcsec,
        )
        self.racs_ref_ra_offset, self.racs_ref_dec_offset = get_bootstrap_offset(
            dra, ddec, raerr, decerr, num=nbootstrap,
        )
        
    def _plot_offsets(self,):
        fig = plot_bootstrap_result(
            self.craco_racs_ra_offset, self.craco_racs_dec_offset, title="field_racs"
        )
        fig.savefig(f"{self.workdir}/bootstrap.field_racs.png", bbox_inches="tight")
        plt.close()
        
        fig = plot_bootstrap_result(
            self.racs_ref_ra_offset, self.racs_ref_dec_offset, title="racs_ref"
        )
        fig.savefig(f"{self.workdir}/bootstrap.racs_ref.png", bbox_inches="tight")
        plt.close()
        
    
    def correct_src_coord(self, dump=True):
        # get catalogues...
        self._get_racs(); self._get_ref()
        # get offsets
        self._compare_all_surveys(nbootstrap=1000)
        self._plot_offsets()
        ### get source nearby the target...
        self._select_sources(snrcut=8.)
        ### start to apply the offset...
        self.ra_offset = np.mean(self.craco_racs_ra_offset) + np.mean(self.racs_ref_ra_offset)
        self.dec_offset = np.mean(self.craco_racs_dec_offset) + np.mean(self.racs_ref_dec_offset)
        self.ra_match_uncertainty = np.std(self.craco_racs_ra_offset) ** 2 + np.std(self.racs_ref_ra_offset) ** 2
        self.dec_match_uncertainty = np.std(self.craco_racs_dec_offset) ** 2 + np.std(self.racs_ref_dec_offset) ** 2
        ### overall note
        note = f"{self.summary}\n\n"
        note += f"Using {self.ref_survey} as the reference catalogue\n"
        note += f"Offsets between CRACO field image and RACS:\n"
        note += f"\tOffset {np.mean(self.craco_racs_ra_offset):.3f} arcsec, {np.mean(self.craco_racs_dec_offset):.3f} arcsec\n"
        note += f"\tUncertainty {np.std(self.craco_racs_ra_offset):.3f} arcsec, {np.std(self.craco_racs_dec_offset):.3f} arcsec\n"
        note += f"Offsets between RACS and Reference catalogue:\n"
        note += f"\tOffset {np.mean(self.racs_ref_ra_offset):.3f} arcsec, {np.mean(self.racs_ref_dec_offset):.3f} arcsec\n"
        note += f"\tUncertainty {np.std(self.racs_ref_ra_offset):.3f} arcsec, {np.std(self.racs_ref_dec_offset):.3f} arcsec\n"
        notes = [note]
        
        ### loop through the source
        for i, row in self.nearbycat.iterrows():
            title = f"Source {i+1} with a SNR of {row['Peak_SNR']}"
            ra = row["RA"]; dec = row["DEC"]
            ra_err = row["E_RA"] * 3600.; dec_err = row["E_DEC"] * 3600.
            ### source flux
            peak_flux = row["Peak_flux"] * 1e3; peak_flux_err = row["E_Peak_flux"] *1e3
            notes.append(self.single_src_corr(
                ra, dec, ra_err, dec_err, title,
                extra = f"Peak flux {peak_flux:.2f}±{peak_flux_err:.2f} mJy"
            ))
            
        ###
        if dump:
            with open(f"{self.workdir}/coord_correct.txt", "w") as fp:
                fp.write("\n-----------\n".join(notes))
        return notes
        
        
    def single_src_corr(self, ra, dec, ra_err, dec_err, title="Source", extra=""): # errors are all in arcsec...
        coord = SkyCoord(ra, dec, unit=units.degree)
        coordcorr = coord.spherical_offsets_by(self.ra_offset*units.arcsec, self.dec_offset*units.arcsec)
        ra_uncertainty = np.sqrt(ra_err ** 2 + self.ra_match_uncertainty)
        dec_uncertainty = np.sqrt(dec_err ** 2 + self.dec_match_uncertainty)
        
        ra_str = coord.ra.to_string(unit=units.hourangle, sep=':', precision=3, pad=True)
        dec_str = coord.dec.to_string(unit=units.degree, sep=':', precision=3, pad=True)
        corr_ra_str = coordcorr.ra.to_string(unit=units.hourangle, sep=':', precision=3, pad=True)
        corr_dec_str = coordcorr.dec.to_string(unit=units.degree, sep=':', precision=3, pad=True)
        
        note = f"{title}\n"
        note += f"Raw source position in CASA {ra_str} {dec_str}\n"
        note += f"Corrected source position {corr_ra_str} {corr_dec_str}\n"
        note += f"RA, DEC uncertainty is {ra_uncertainty:.3f} arcsec, {dec_uncertainty:.3f} arcsec\n"
        note += f"{extra}\n"
        
        return note
    
    def plot_burst_field(self, ):
        wcs = WCS(self.bursthdul[0].header).celestial
        data = np.squeeze(self.bursthdul[0].data)

        cutout = fits_cutout(data, wcs, *self.coord, radius=2 * self.sep_thres)

        fig, ax = plot_fits_data(cutout.data, cutout.wcs)

        for i, row in self.nearbycat.iterrows():
            ra = row["RA"]; dec = row["DEC"]
            ax.scatter(
                ra, dec, marker="x", s=50,
                transform=ax.get_transform("icrs"),
                c = "cyan"
            )
            ax.text(
                ra, dec, f"Source {i+1}", color="cyan",
                transform=ax.get_transform("icrs"),
                va="bottom", ha="center"
            )
            
        fig.savefig(f"{self.workdir}/burstfield.png", bbox_inches="tight")
        plt.close()
        
