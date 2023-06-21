import numpy as np

from astropy.io import fits
from astropy.wcs import WCS
from stips.scene_module import SceneModule
from stips.observation_module import ObservationModule


def _synthesize_image(obs_prefix, obs_ra, obs_dec, n_stars, filt, extra_kwargs):
    scm = SceneModule(out_prefix=obs_prefix, ra=obs_ra, dec=obs_dec, **extra_kwargs)

    stellar_parameters = {
        'n_stars': n_stars,
        'age_low': 7.5e12,
        'age_high': 7.5e12,
        'z_low': -2.0,
        'z_high': -2.0,
        'imf': 'salpeter',
        'alpha': -2.35,
        'binary_fraction': 0.01,
        # 'clustered': True,
        # 'distribution': 'invpow',
        'clustered': False,
        'distribution': 'uniform',
        'radius': 1.0,
        'radius_units': 'pc',
        'distance_low': 1.0,
        'distance_high': 1.0,
        'offset_ra': 0.0,
        'offset_dec': 0.0
    }

    offset = {
        'offset_id': 1,
        'offset_centre': False,
        'offset_ra': 0.0,
        'offset_dec': 0.0,
        'offset_pa': 0.0
    }

    residuals = {
        'residual_flat': False,
        'residual_dark': False,
        'residual_cosmic': False,
        'residual_poisson': True,
        'residual_readnoise': True
    }

    observation_parameters = {
        'instrument': 'WFI',
        'filters': [filt],
        'detectors': 1,
        'distortion': False,
        'background': 0.01,
        'observations_id': 1,
        'exptime': 1000,
        'offsets': [offset]
    }

    stellar_cat_file = scm.CreatePopulation(stellar_parameters)

    obm = ObservationModule(
        observation_parameters, out_prefix=obs_prefix,
        ra=obs_ra, dec=obs_dec, residual=residuals,
        **extra_kwargs
    )

    obm.nextObservation()

    obm.addCatalogue(stellar_cat_file)

    obm.addError()

    synth_fits_path, mosaic_file, params = obm.finalize(mosaic=False)

    coords = [
        list(map(float, line.split(' ')[-1].split(',')))
        for line in fits.getheader(synth_fits_path, ext=1)['HISTORY']
        if line.startswith('Adding point')
    ]

    true_pixel_coords = np.array(coords) - 22

    wcs = WCS(fits.getheader(synth_fits_path, ext=1))

    true_sky_coords = wcs.pixel_to_world(*true_pixel_coords.T)

    return synth_fits_path, true_pixel_coords, true_sky_coords

