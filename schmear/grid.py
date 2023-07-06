import logging
import os
import numpy as np
import astropy.units as u

import webbpsf
from webbpsf import gridded_library, setup_logging

from astropy.nddata import CCDData
from photutils.psf import GriddedPSFModel

setup_logging(level="ERROR")


def _grid_model(
    filt, filename, detector, oversample,
    fov_pixels, n_psfs, visualize=False, overwrite=False,
    expected_filename=''
):
    # Choose pixel boundaries for the grid of PSFs:
    start_pix = 0
    stop_pix = 4000
    buffer_pix = 100

    # Choose locations on detector for each PSF:
    pixel_range = np.linspace(
        start_pix + buffer_pix,
        stop_pix - buffer_pix,
        int(n_psfs ** 0.5)
    )
    location_list = [
        (int(x), int(y)) for y in pixel_range for x in pixel_range
    ]

    if not os.path.exists(expected_filename) or overwrite:
        wfi = webbpsf.roman.WFI()
        wfi.filter = filt
        # wfi.options['jitter'] = None
        # wfi.options['jitter_sigma'] = 0
        filter_to_wl = (int(wfi.filter[1:]) / 100 * u.um).to_value(u.m)

        # Initialize the PSF library
        inst = gridded_library.CreatePSFLibrary(
            instrument=wfi,
            filter_name=wfi.filter,
            detectors=detector,
            num_psfs=n_psfs,
            monochromatic=filter_to_wl,
            oversample=oversample,
            fov_pixels=fov_pixels,
            add_distortion=False,
            crop_psf=False,
            save=True,
            filename=filename,
            overwrite=True,
            verbose=False
        )

        inst.location_list = location_list

        # Create the PSF grid:
        gridmodel = inst.create_grid()

        if visualize:
            webbpsf.gridded_library.display_psf_grid(gridmodel)
    elif os.path.exists(expected_filename):
        logging.log(logging.INFO, "Loading existing gridded PSF model")
        ndd = CCDData.read(expected_filename, unit=u.electron/u.s, ext=0)
        ndd.data = np.flip(np.flip(ndd.data, axis=2), axis=1)
        ndd.meta = dict(ndd.meta)
        ndd.meta['oversampling'] = oversample
        ndd.meta['grid_xypos'] = np.array(
            [list(tup)[::-1] for tup in location_list]
        )
        gridmodel = GriddedPSFModel(ndd)

    return gridmodel, location_list
