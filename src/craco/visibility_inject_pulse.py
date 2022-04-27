from __future__ import division, print_function, absolute_import
import numpy as NP
from astropy.coordinates import SkyCoord
from astropy import units as U
from astropy import constants as FC
from astropy.wcs import WCS

from Furby_p3.sim_furby import get_furby
from craft.craco_plan import PipelinePlan
from craft import uvfits
from craft.craco import bl2ant

################################################################################

def dircos_from_wcs(coords, wcs):
    
    """
    ----------------------------------------------------------------------------
    Compute direction cosines given coordinates in (RA, Dec) and a WCS
    
    Inputs:
    
    coords   [instance of class astropy.coordinates.SkyCoord] Single or M-array 
             of SkyCoord type
             
    wcs      [instance of class astropy.wcs.WCS] World coordinate system, an 
             instance of class WCS in astropy
             
    Outputs:
    
    dircos   [numpy array] (M,3)-shaped array of direction cosine coordinates
    ----------------------------------------------------------------------------
    """
    
    if not isinstance(wcs, WCS):
        raise TypeError('Input wcs must be an instance of class astropy.wcs.WCS')
    if not isinstance(coords, SkyCoord):
        raise TypeError('Input coords must be an instance of class astrop.coordinates.SkyCoord')
        
    xy_pix = wcs.all_world2pix(NP.hstack([coords.ra.deg.reshape(-1,1), coords.dec.deg.reshape(-1,1)]), 1)
    xy_pix = xy_pix.reshape(-1,2)
    x = xy_pix[:,0]
    y = xy_pix[:,1]
    theta_x = (x-wcs.wcs.crpix[0]) * wcs.wcs.cdelt[0]
    theta_y = (y-wcs.wcs.crpix[1]) * wcs.wcs.cdelt[1]
    l = NP.sin(NP.radians(theta_x))
    m = NP.sin(NP.radians(theta_y))
    n = NP.sqrt(1-(l**2+m**2))
    dircos = NP.hstack([l.reshape(-1,1), m.reshape(-1,1), n.reshape(-1,1)])
    
    return dircos

################################################################################

def rime_propagator_term(blvec, lvec, fvec):
    
    """
    ----------------------------------------------------------------------------
    Compute the complex exponential in the Radio Interferometer Measurement 
    Equation (RIME) given source locations, baseline vectors, and frequencies.
    
    propagator = exp(-2*pi*i*(f/c)*b.l)

    Inputs:
    
    blvec   [numpy array] Numpy array of baseline vectors (in units of m). 
            Shape=(...,nbl,3)
    
    lvec    [numpy array] Numpy array of direction cosine vectors. 
            Shape=(nsrc,3)
             
    fvec    [numpy array] Numpy array of frequencies (in units of Hz). 
            Shape=(nspw,nfreqs) or just (nfreqs,). 
            nspw = number of spectral windows (default=1)
             
    Outputs:
    
    propagator   
            [numpy array] (...,nbl,nspw,nfreq,nsrc) shaped array of propagating
            term in the RIME
    ----------------------------------------------------------------------------
    """
    
    if not isinstance(blvec, NP.ndarray):
        raise TypeError('Input blvec must be a numpy array')
    if blvec.ndim < 1:
        raise ValueError('Input blvec must have at least two dimensions')
    if blvec.ndim == 1:
        blvec = blvec[NP.newaxis,...]
    if blvec.shape[-1] != 3:
        raise ValueError('Input blvec must have shape (...,3)')

    if not isinstance(lvec, NP.ndarray):
        raise TypeError('Input lvec must be a numpy array')
    if lvec.ndim < 1:
        raise ValueError('Input lvec must have at least one dimension')
    if lvec.ndim == 1:
        lvec = lvec[NP.newaxis,...] 
    if lvec.shape[-1] != 3:
        raise ValueError('Input lvec must have shape (...,3)')
 
    if not isinstance(fvec, (int,float,NP.ndarray)):
        raise TypeError('Input fvec must be a scalar or numpy array')
    fvec = NP.asarray(fvec)
    if fvec.ndim == 1:
        fvec = fvec.reshape(1,-1)
        
    propagator = NP.exp(-1j * 2 * NP.pi * NP.einsum('...ik,...jk->...ij', blvec, lvec)[...,NP.newaxis,NP.newaxis,:] * fvec[...,NP.newaxis] / FC.c.si.value) # shape=(...,nbl,nspw,nfreq,nsrc)
    return propagator

################################################################################

def get_bl_from_plan(plan):
    
    """
    ----------------------------------------------------------------------------
    Get baseline information from a CRACO plan
    
    Inputs:
    
    plan    [instance of class CRACO PipelinePlan]
    
    Outputs:
    
    (blid, antpair, blvec)   
            [3-element tuple] 
            blid:    [numpy array] Array of baseline ids. Shape=(nbl,)
            antpair  [numpy recArray] RecArray of antenna pairs, where each 
                     element is a tuple of integers consisting of antenna 
                     numbers. Shape=(nbl,).
            blvec    [numpy array] Baseline vectors relative to phase center. 
                     Units in m. Shape=(nbl,3)
    ----------------------------------------------------------------------------
    """
    
    blvec = []
    antpair = []
    blid = []
    for baseline_id, bldata in list(plan.baselines.items()):
        u = bldata['UU'] * FC.c.si.value # convert seconds to metres
        v = bldata['VV'] * FC.c.si.value # convert seconds to metres
        w = bldata['WW'] * FC.c.si.value # convert seconds to metres
        blvec += [(u, v, w)]
        antpair += [bl2ant(baseline_id)]
        blid += [baseline_id]
    antpair = NP.array(antpair, dtype=[('A1', '<i8'), ('A2', '<i8')])
    blvec = NP.asarray(blvec)
    blid = NP.asarray(blid)
    return (blid, antpair, blvec)

################################################################################

def genvis_1PS(blvec, src_dircos, fvec, apparent_stokes_intensity=None, ignore_w=False):
    
    """
    ----------------------------------------------------------------------------
    Generate visibility for a point source given its arbitrary location (in 
    direction cosine coordinates), baseline vector, frequencies, and the source 
    intensity
    
    Inputs:
    
    blvec   [numpy array] Numpy array of baseline vectors (in units of m). 
            Shape=(...,nbl,3)
    
    src_dircos    
            [numpy array] Numpy array of direction cosine vectors. 
            Shape=(nsrc,3)
             
    fvec    [numpy array] Numpy array of frequencies (in units of Hz). 
            Shape=(nspw,nfreqs) or just (nfreqs,). 
            nspw = number of spectral windows (default=1)
            
    apparent_stokes_intensity
            [NoneType or numpy array] A numpy array of point source Stokes 
            intensity. Shape=(nspw,nfreq,nStokesPol). If ndim=2, it will be
            recast into shape (nspw=1,nfreq,nStokesPol). nStokesPol=1 (Stokes I) 
            or 4 (I,Q,U,V). If ndim<2, an error will be raised. If set to None, 
            it will be set to a value of 1 with shape=(1,1,1).
            
    ignore_w
            [boolean] If set to True, the w-values in the baseline vector will
            be set to zero. If False (default), w-values will be used in the 
            computation of visibilities.
             
    Outputs:
    
    vis     [numpy array] (...,nbl,nspw,nfreq,nStokesPol) shaped array of 
            visibilities
    ----------------------------------------------------------------------------
    """
        
    if apparent_stokes_intensity is None:
        apparent_stokes_intensity = NP.array([1.0]).reshape(1,1,1) # shape=(nspw=1,nfreq=1,nStokesPol=1)
    if not isinstance(apparent_stokes_intensity, (int,float,NP.ndarray)):
        raise TypeError('Input apparent_stokes_intensity must be a scalar or numpy array')
    if isinstance(apparent_stokes_intensity, (int,float)):
        apparent_stokes_intensity = NP.asarray(apparent_stokes_intensity).reshape(1,1,1)
    if apparent_stokes_intensity.ndim < 2:
        raise ValueError('Input apparent_stokes_intensity must have two dimensions')
    if apparent_stokes_intensity.ndim == 2:
        apparent_stokes_intensity = apparent_stokes_intensity[NP.newaxis,...] # shape=(nspw=1,nfreq,nStokesPol)
    nStokesPol = apparent_stokes_intensity.shape[-1]
    if (nStokesPol != 1) and (nStokesPol != 4):
        raise ValueError('Input apparent_stokes_intensity must be a 1- or 4-element array')
    
    if not isinstance(src_dircos, NP.ndarray):
        raise TypeError('Input src_dircos must be a numpy array')
    src_dircos = src_dircos.reshape(-1)
    if src_dircos.shape[-1] != 3:
        raise ValueError('Input src_dircos must be a 3-element array')
        
    if ignore_w: # If set, ignore w-term
        blvec[:,2] = 0.0
    
    propagator = rime_propagator_term(blvec, src_dircos, fvec)
    vis = apparent_stokes_intensity * propagator[...,0][...,NP.newaxis] # shape=(...,nbl,nspw,nfreq,nStokesPol)
    
    return vis

################################################################################

def genvis_1PS_at_phase_center(blvec, fvec, apparent_stokes_intensity=None, ignore_w=False):
    
    """
    ----------------------------------------------------------------------------
    Generate visibility for a point source at phase center, baseline vector, 
    frequencies, and the source intensity
   
    Inputs:
    
    blvec   [numpy array] Numpy array of baseline vectors (in units of m). 
            Shape=(...,nbl,3)
    
    fvec    [numpy array] Numpy array of frequencies (in units of Hz). 
            Shape=(nspw,nfreqs) or just (nfreqs,). 
            nspw = number of spectral windows (default=1)
            
    apparent_stokes_intensity
            [NoneType or numpy array] A numpy array of point source Stokes 
            intensity. Shape=(nspw,nfreq,nStokesPol). If ndim=2, it will be
            recast into shape (nspw=1,nfreq,nStokesPol). nStokesPol=1 (Stokes I) 
            or 4 (I,Q,U,V). If ndim<2, an error will be raised. If set to None, 
            it will be set to a value of 1 with shape=(1,1,1).
            
    ignore_w
            [boolean] If set to True, the w-values in the baseline vector will
            be set to zero. If False (default), w-values will be used in the 
            computation of visibilities.
             
    Outputs:
    
    vis     [numpy array] (...,nbl,nspw,nfreq,nStokesPol) shaped array of 
            visibilities
    ----------------------------------------------------------------------------
    """
        
        
    src_dircos_pc = NP.array([0.0, 0.0, 1.0])
    return genvis_1PS(blvec, src_dircos_pc, fvec, apparent_stokes_intensity=apparent_stokes_intensity, ignore_w=ignore_w)

################################################################################

def genvis_1PS_from_plan(plan, src_ra_deg, src_dec_deg, apparent_stokes_intensity=None, ignore_w=False):
    
    """
    ----------------------------------------------------------------------------
    Generate visibility for a point source given its arbitrary location (in 
    RA, Dec), its Stokes intensity, and a CRACO plan from which the baseline 
    vector, and frequencies
    
    Inputs:
    
    plan    [instance of class CRACO PipelinePlan]
    
    src_ra_deg    
            [numpy array] Numpy array of source RA (in deg). 
            Shape=(nsrc,3)
             
    src_dec_deg    
            [numpy array] Numpy array of source Dec (in deg). 
            Shape=(nsrc,3)
             
    apparent_stokes_intensity
            [NoneType or numpy array] A numpy array of point source Stokes 
            intensity. Shape=(nspw,nfreq,nStokesPol). If ndim=2, it will be
            recast into shape (nspw=1,nfreq,nStokesPol). nStokesPol=1 (Stokes I) 
            or 4 (I,Q,U,V). If ndim<2, an error will be raised. If set to None, 
            it will be set to a value of 1 with shape=(1,1,1).
            
    ignore_w
            [boolean] If set to True, the w-values in the baseline vector will
            be set to zero. If False (default), w-values will be used in the 
            computation of visibilities.
             
    Outputs:
    
    vis     [numpy array] (...,nbl,nspw,nfreq,nStokesPol) shaped array of 
            visibilities
    ----------------------------------------------------------------------------
    """
        
    blid, antpair, blvec = get_bl_from_plan(plan)
    src_coord = SkyCoord(ra=NP.asarray(src_ra_deg).reshape(-1)*U.deg, dec=NP.asarray(src_dec_deg).reshape(-1)*U.deg, frame='icrs')
    src_dircos = dircos_from_wcs(src_coord, plan.wcs)
    fvec = plan.freqs
    vis = genvis_1PS(blvec, src_dircos, fvec, apparent_stokes_intensity=apparent_stokes_intensity, ignore_w=ignore_w)
    return vis

################################################################################

def genvis_1PS_from_dynamic_spectrum(vis_1ps_static, dynamic_spectrum, spwind=0, polind=0, outfmt='craco'):
    
    """
    ----------------------------------------------------------------------------
    Generate visibility dynamic spectrum for a point source given its static 
    visibility and a dynamic spectrum simulated for a FRB candidate. 
    
    Inputs:
    
    vis_1ps_static
            [numpy array] Static visibility of a single point source of shape 
            (nbl,nspw,nfreq,nStokesPol) 
            
    dynamic_spectrum
            [numpy array] Dynamic spectrum of a FRB candidate intensity of shape
            (ntimes,nfreq)
            
    spwind  [integer] Index of spectral window (default=0)
    
    polind  [integer] Index of polarization (default=0 => Stokes I)
    
    outfmt  [string] Accepted values are 'craco' and 'uvdata'. Depending on this
            parameter, the output shape is different
            
    Outputs:
    
    vis_dynamic     
            [numpy array] Dynamic spectrum of visibilities of FRB candidate. 
            Depending on the value of parameter outfmt, the returned shape will 
            be (ntimes,nbl,nfreq) for outfmt='craco' and (nbl,nfreq,ntimes) for
            outfmt='uvdata'
    ----------------------------------------------------------------------------
    """
            
    if not isinstance(vis_1ps_static, NP.ndarray):
        raise TypeError('Input vis_1ps_static must be a numpy array')
    if vis_1ps_static.ndim != 4:
        raise ValueError('Input vis_1ps_static must have shape (nbl,nspw,nfreq,nStokesPol)')
    if (vis_1ps_static.shape[-1] != 1) and (vis_1ps_static.shape[-1] != 4):
        raise ValueError('Input vis_1ps_static must have shape (...,1) or (...,4)')

    if not isinstance(spwind, int):
        raise TypeError('Input spwind must be an integer')
        
    if not isinstance(polind, int):
        raise TypeError('Input polind must be an integer')
        
    if not isinstance(outfmt, str):
        raise TypeError('Input outfmt must be a string')
    if outfmt.lower() not in ['craco', 'uvdata']:
        raise ValueError('Value specified for outfmt not supported')
            
    vis_dynamic = vis_1ps_static[:,spwind,:,polind][NP.newaxis,...] * dynamic_spectrum[:,NP.newaxis,:] # shape=(ntimes,nbl,nfreqs)
    
    if outfmt.lower() == 'craco':
        return NP.moveaxis(vis_dynamic, 0, -1) # shape=(nbl,nfreqs,ntimes) 
    if outfmt.lower() == 'uvdata':
        return vis_dynamic # shape=(ntimes,nbl,nfreqs)

################################################################################

def gen_dispersed_vis_1PS_from_plan(plan, src_ra_deg, src_dec_deg, dynamic_spectrum, apparent_stokes_intensity=None, ignore_w=False, spwind=0, polind=0, outfmt='craco'):

    """
    ----------------------------------------------------------------------------
    Generate visibility dynamic spectrum for a point source given a CRACO plan,
    an arbitrary source location, and a dynamic spectrum simulated for a FRB 
    candidate. 
    
    Inputs:
    
    plan    [instance of class CRACO PipelinePlan]
    
    src_ra_deg    
            [numpy array] Numpy array of source RA (in deg). 
            Shape=(nsrc,3)
             
    src_dec_deg    
            [numpy array] Numpy array of source Dec (in deg). 
            Shape=(nsrc,3)
             
    ignore_w
            [boolean] If set to True, the w-values in the baseline vector will
            be set to zero. If False (default), w-values will be used in the 
            computation of visibilities.
             
    apparent_stokes_intensity
            [NoneType or numpy array] A numpy array of point source Stokes 
            intensity. Shape=(nspw,nfreq,nStokesPol). If ndim=2, it will be
            recast into shape (nspw=1,nfreq,nStokesPol). nStokesPol=1 (Stokes I) 
            or 4 (I,Q,U,V). If ndim<2, an error will be raised. If set to None, 
            it will be set to a value of 1 with shape=(1,1,1).
            
    dynamic_spectrum
            [numpy array] Dynamic spectrum of a FRB candidate intensity of shape
            (ntimes,nfreq)
            
    spwind  [integer] Index of spectral window (default=0)
    
    polind  [integer] Index of polarization (default=0 => Stokes I)
    
    outfmt  [string] Accepted values are 'craco' and 'uvdata'. Depending on this
            parameter, the output shape is different
            
    Outputs:
    
    vis_dynamic     
            [numpy array] Dynamic spectrum of visibilities of FRB candidate. 
            Depending on the value of parameter outfmt, the returned shape will 
            be (ntimes,nbl,nfreq) for outfmt='craco' and (nbl,nfreq,ntimes) for
            outfmt='uvdata'
    ----------------------------------------------------------------------------
    """            
    
    vis_1ps_static = genvis_1PS_from_plan(plan, src_ra_deg, src_dec_deg, apparent_stokes_intensity=apparent_stokes_intensity, ignore_w=ignore_w)
    vis_dynamic = genvis_1PS_from_dynamic_spectrum(vis_1ps_static, dynamic_spectrum, spwind=spwind, polind=polind, outfmt=outfmt)
    return vis_dynamic

################################################################################

"""
Example:

from craco import visibility_inject_pulse as VIP

vis_1ps_disp = VIP.gen_dispersed_vis_1PS_from_plan(plan, src_ra_deg, src_dec_deg, dynamic_spectrum, apparent_stokes_intensity=None, ignore_w=True, spwind=0, polind=0, outfmt='craco')
"""
