import healpy as hp
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, LogNorm
from matplotlib.ticker import MaxNLocator
from matplotlib import rcParams
import numpy as np
from ligo.skymap import io, kde, bayestar, version, plot, postprocess
from astropy.time import Time
from astropy.coordinates import SkyCoord
from astropy import units as u
import os


def rectangular_plot(x_arr, y_arr, z, xlabel='', ylabel='', log_norm=True):

    # generate 2 2d grids for the x & y bounds
    y, x = np.meshgrid(y_arr, x_arr)

    # x and y are bounds, so z should be the value *inside* those bounds.
    # Therefore, remove the last value from the z array.
    # z = z[:-1, :-1]
    #

    # pick the desired colormap, sensible levels, and define a normalization
    # instance which takes data values and translates those into levels.
    # cmap = plt.get_cmap('PiYG')
    cmap = plt.get_cmap()
    if not log_norm:
        levels = MaxNLocator(nbins=15).tick_values(z.min(), z.max())
        norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)
    else:
        norm = LogNorm(vmin=z.min(), vmax=z.max())

    fig, ax = plt.subplots(nrows=1)
    # im = ax0.pcolormesh(x, y, z, cmap=cmap, norm=norm)
    im = ax.pcolormesh(x, y, z, norm=norm)
    fig.colorbar(im, ax=ax)
    # adjust spacing between subplots so `ax1` title and `ax0` tick labels
    # don't overlap
    fig.tight_layout()

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    return fig, ax


def write_healpix_map(skylocs, nside=16, outdir='./',
                      enable_multiresolution=False,
                      fitsoutname='skymap.fits.gz',
                      nest=True):
    """
    Construct a sky map of the sky localization posterior distribution
    using Healpix


    Parameters
    ==========
    skylocs : 2d numpy array
        sky location angle matrix of size Nsamples x 2 [radians]
    logpvals : array_like
        log posterior probability values, vector of size Nsamples
    contour : float or None
        plot contour enclosing this percentage of probability mass [may be
        specified multiple times, default: none]

    theta = EclipticLatitude + np.pi / 2,
    phi = EclipticLongitude - np.pi


    """

    # Conversion to Ecliptic latitude and longitude
    latitude = skylocs[:, 0] - np.pi / 2
    longitude = skylocs[:, 1] + np.pi
    # Should we concert into degrees?
    # latitude *= 180 / np.pi
    # longitude *= 180 / np.pi

    # convert angles in right ascension and declination
    pts = np.column_stack((longitude, latitude))

    # Create instance of 2D KDE class
    skypost = kde.Clustered2DSkyKDE(pts)

    # # Pickle the skyposterior object
    # with open(os.path.join(outdir, 'skypost.obj'), 'wb') as out:
    #     pickle.dump(skypost, out)

    # Making skymap
    hpmap = skypost.as_healpix()

    if not enable_multiresolution:
        hpmap = bayestar.rasterize(hpmap)

    # Include metadata
    hpmap.meta.update(io.fits.metadata_for_version_module(version))
    hpmap.meta['creator'] = 'Q. Baghi'
    hpmap.meta['origin'] = 'LISA'
    hpmap.meta['gps_creation_time'] = Time.now().gps

    # Write skymap to file if needed
    io.write_sky_map(os.path.join(outdir, fitsoutname), hpmap, nest=nest)


def load_healpix_map(outdir, fitsoutname='skymap.fits.gz', radecs=[],
                     contour=None, annotate=True, inset=False):
    # Load skymap
    skymap, metadata = io.fits.read_sky_map(os.path.join(outdir, fitsoutname),
                                            nest=None)

    # Convert sky map from probability to probability per square degree.
    nside = hp.npix2nside(len(skymap))
    deg2perpix = hp.nside2pixarea(nside, degrees=True)
    probperdeg2 = skymap / deg2perpix

    # Projection type
    ax = plt.axes(projection='astro hours mollweide')
    ax.grid()

    # Plot sky map.
    vmax = probperdeg2.max()
    img = ax.imshow_hpx((probperdeg2, 'ICRS'), nested=metadata['nest'],
                        vmin=0., vmax=vmax)

    # Add colorbar.
    cb = plot.colorbar(img)
    cb.set_label(r'Prob. per deg²')

    if contour:
        # Add contours.
        cls = 100 * postprocess.find_greedy_credible_levels(skymap)
        cs = ax.contour_hpx(
            (cls, 'ICRS'), nested=metadata['nest'],
            colors='k', linewidths=0.5, levels=contour)
        fmt = r'%g\%%' if rcParams['text.usetex'] else '%g%%'
        plt.clabel(cs, fmt=fmt, fontsize=6, inline=True)

    # # Add markers (e.g., for injections or external triggers).
    # for ra, dec in radecs:
    #     ax.plot_coord(
    #         SkyCoord(ra, dec, unit='deg'), '*',
    #         markerfacecolor='white', markeredgecolor='black', markersize=10)

    # Try to add a zoom inset
    if inset:
        ra, dec = radecs
        center = SkyCoord(ra*u.deg, dec*u.deg)
        ax_inset = plt.axes(
            [0.59, 0.3, 0.4, 0.4],
            projection='astro zoom',
            center=center,
            radius=10 * u.deg)
        for key in ['ra', 'dec']:
            ax_inset.coords[key].set_ticklabel_visible(False)
            ax_inset.coords[key].set_ticks_visible(False)
        ax.grid()
        ax.mark_inset_axes(ax_inset)
        ax.connect_inset_axes(ax_inset, 'upper left')
        ax.connect_inset_axes(ax_inset, 'lower left')
        ax_inset.scalebar((0.1, 0.1), 5 * u.deg).label()
        ax_inset.compass(0.9, 0.1, 0.2)

        ax_inset.imshow_hpx((probperdeg2, 'ICRS'), nested=metadata['nest'],
                            vmin=0., vmax=vmax)  # , cmap='cylon')
        ax_inset.plot(
            center.ra.deg, center.dec.deg,
            transform=ax_inset.get_transform('world'),
            marker=plot.reticle(),
            color='white',
            markersize=30,
            markeredgewidth=3)

    # Add a white outline to all text to make it stand out from the background.
    plot.outline_text(ax)
    ax.grid()

    if annotate:
        text = []
        try:
            objid = metadata['objid']
        except KeyError:
            pass
        else:
            text.append('event ID: {}'.format(objid))
        if contour:
            pp = np.round(contour).astype(int)
            ii = np.round(np.searchsorted(np.sort(cls),
                                          contour) * deg2perpix).astype(int)
            for i, p in zip(ii, pp):
                # FIXME: use Unicode symbol instead of TeX '$^2$'
                # because of broken fonts on Scientific Linux 7.
                text.append(
                    u'{:d}% area: {:d} deg²'.format(p, i, grouping=True))
        ax.text(1, 1, '\n'.join(text), transform=ax.transAxes, ha='right')

    plt.show()
