#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Friday March 10 2020

Test the mapplots.py module using the skymap package

@author: qbaghi

"""

import ligo.skymap.plot
from matplotlib import pyplot as plt
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy import units as u
import ligo.skymap.plot
import astropy_healpix as ah
import tempfile
from ligo.skymap import version, kde, io
import pickle
import os
from astropy.time import Time
from optparse import OptionParser


if __name__ == '__main__':

    # -------------------------------------------------------------------------
    # Load simulation informations
    # -------------------------------------------------------------------------
    parser = OptionParser(usage="usage: %prog [options] YYY.txt",
                          version="12.02.2020, Quentin Baghi")
    (options, args) = parser.parse_args()
    if not args:
        fpath = "/Users/qbaghi/Codes/python/pylisa/pylisa-dev/data/"
        input_prefix = "2020-07-13_21h09-54_04_signal_delay_tdi_unequalnoises"
        input_path = fpath + input_prefix
    else:
        input_path = args[0]
    # File paths
    # input_prefix = "2020-07-22_09h53-46_04_signal_delay_pci_equalnoises"
    chain_path = input_path + '_chain.p'
    lprob_path = input_path + '_lnprob.p'
    # Load sample chains
    chain_file = open(chain_path, 'rb')
    chain = pickle.load(chain_file)
    chain_file.close()
    # Load log-posterior values
    lprob_file = open(lprob_path, 'rb')
    lprob = pickle.load(lprob_file)
    lprob_file.close()
    # # Load true source parameters
    # config_file = fpath + input_prefix + "_config.ini"
    # param_dic = pyloadings.load_source_params(config_file)
    # Index of sky location parameter in the chains
    i_sky = 6

    # -------------------------------------------------------------------------
    # Process sample data
    # -------------------------------------------------------------------------
    n_burn = 3000
    n_thin = 8
    print("Shape of sample array: " + str(chain.shape))
    chain_eff = chain[0, :, n_burn::n_thin, :]
    lprob_eff = lprob[0, :, n_burn::n_thin]
    inds_nonzero = np.where(chain_eff[0, :, i_sky + 0] != 0)[0]
    chain_eff = chain_eff[:, inds_nonzero, :]
    lprob_eff = lprob_eff[:, inds_nonzero]
    # Conversion to Ecliptic latitude and longitude
    latitude = chain_eff[:, :, i_sky + 0].flatten() - np.pi / 2
    longitude = chain_eff[:, :, i_sky + 1].flatten() + np.pi

    # -------------------------------------------------------------------------
    # Convert to celestial (equatorial) coordinates
    # -------------------------------------------------------------------------    
    # Conversion to SkyCoord object
    coords = SkyCoord(longitude*u.rad, latitude*u.rad, 
                      frame='barycentrictrueecliptic')
    # Conversion to celestial (equatorial) coordinates
    coords_cel = coords.icrs
    # Get the location by name
    source = SkyCoord.from_name("HM Cnc")
    print(source.icrs)

    # -------------------------------------------------------------------------
    # Write the skymap
    # -------------------------------------------------------------------------    
    # Prepare output
    # outdir = '../data/'
    # fitsoutname = input_prefix + "_skymap.fits.gz"
    fitsoutname = input_path + "_skymap.fits.gz"
    enable_multiresolution = True
    # convert angles in right ascension and declination
    pts = np.column_stack((coords_cel.ra.rad, coords_cel.dec.rad))

    # Create instance of 2D KDE class
    skypost = kde.Clustered2DSkyKDE(pts)
    
    # Pickle the skyposterior object
    # with open(os.path.join(outdir, 'skypost.obj'), 'wb') as out:
    #     pickle.dump(skypost, out)
    with open(input_path + 'skypost.obj', 'wb') as out:
        pickle.dump(skypost, out)

    # Making skymap
    hpmap = skypost.as_healpix()

    # if not enable_multiresolution:
    #     hpmap = bayestar.rasterize(hpmap)

    # Include metadata
    hpmap.meta.update(io.fits.metadata_for_version_module(version))
    hpmap.meta['creator'] = 'Q. Baghi'
    hpmap.meta['origin'] = 'LISA'
    hpmap.meta['gps_creation_time'] = Time.now().gps

    # Write skymap to file if needed
    # f_name = os.path.join(outdir, fitsoutname)
    f_name = fitsoutname
    io.fits.write_sky_map(f_name, 
                          hpmap, nest=True,
                          vcs_version='foo 1.0', vcs_revision='bar',
                          build_date='2018-01-01T00:00:00')
    for card in fits.getheader(f_name, 1).cards:
        print(str(card).rstrip())

    # # -------------------------------------------------------------------------
    # # Plot map
    # # -------------------------------------------------------------------------
    # # Parameters
    # # frame = 'barycentrictrueecliptic'
    # frame = 'galactic'
    # # frame = 'ICRS'
    # cmap = 'cylon'
    # projection = 'astro hours mollweide'
    # # Load custom presets
    # presets.plotconfig(lbsize=14, lgsize=12, autolayout=True, figsize=(8, 4),
    #                    ticklabelsize=12)
    # # True values
    # # lat_true = param_dic.get('theta') - np.pi / 2
    # # lon_true = param_dic.get('phi') + np.pi
    # # source = SkyCoord((param_dic.get('phi') + np.pi)*u.rad, 
    # #                   (param_dic.get('theta') - np.pi / 2)*u.rad, 
    # #                 frame='barycentrictrueecliptic')
    # source = SkyCoord.from_name("HM Cnc")
    # source_gal = source.galactic 
    # # Load skymap
    # skymap, metadata = io.fits.read_sky_map(os.path.join(outdir, fitsoutname),
    #                                         nest=nest)
    # # Convert sky map from probability to probability per square degree.
    # nside = hp.npix2nside(len(skymap))
    # deg2perpix = hp.nside2pixarea(nside, degrees=True)
    # probperdeg2 = skymap / deg2perpix
    # # Projection type
    # ax = plt.axes(projection=projection)
    # ax.grid(color='gray', linestyle='dotted')
    # # Plot sky map.
    # vmax = probperdeg2.max()
    # # frame = 'ICRS'
    # img = ax.imshow_hpx((probperdeg2, frame), 
    #                     nested=metadata['nest'],
    #                     vmin=0., vmax=vmax,
    #                     cmap=cmap)
    # # Add colorbar.
    # cb = plot.colorbar(img)
    # cb.set_label(r'Probability per deg²')
    # # Add a white outline to all text to make it stand out from the background.
    # # plot.outline_text(ax)
    # plt.show()
    
    # mapplots.load_healpix_map(outdir,
    #                           fitsoutname=fitsoutname,
    #                           radecs=[],
    #                           contour=None,
    #                           annotate=False,
    #                           inset=False)
