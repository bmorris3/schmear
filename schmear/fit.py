import asdf
import numpy as np

from astropy.modeling.fitting import LevMarLSQFitter
from astropy.nddata import CCDData, StdDevUncertainty
from astropy.table import Table
import astropy.units as u

from photutils.psf import SourceGrouper, PSFPhotometry
from photutils.background import LocalBackground
from photutils.detection import DAOStarFinder


def _fit_psf(crit_separation, gridmodel, coords, asdf_file,
             cls, fit_shape, progress_bar=False):
    grouper = SourceGrouper(min_separation=crit_separation)
    fitter = LevMarLSQFitter(calc_uncertainties=True)

    if cls is PSFPhotometry:
        extras = {}
    else:
        extras = dict(
            finder=DAOStarFinder(
                # these defaults extracted from the
                # romancal SourceDetectionStep
                fwhm=2.0,
                threshold=2.0,
                sharplo=0.0,
                sharphi=1.0,
                roundlo=-1.0,
                roundhi=1.0,
                peakmax=1000.0,
            ),
        )

    photometry = cls(
        grouper=grouper,
        localbkg_estimator=LocalBackground(
            inner_radius=10, outer_radius=30
        ),
        psf_model=gridmodel,
        fitter=fitter,
        fit_shape=fit_shape,
        aperture_radius=15,
        progress_bar=progress_bar,
        **extras
    )

    guesses = Table(coords, names=['x_init', 'y_init'])

    with asdf.open(asdf_file) as file:
        data = np.array(file.tree['roman']['data'])
        error = np.array(file.tree['roman']['err'])
        mask = np.array(file.tree['roman']['dq']) != 0

    error = np.clip(error, 1e-2, None)
    
    shape = data.shape
    buffer = 0
    in_range = (
        (guesses['x_init'] > buffer) &
        (guesses['x_init'] < shape[0] - buffer) &
        (guesses['y_init'] > buffer) &
        (guesses['y_init'] < shape[1] - buffer)
    )
    guesses = guesses[in_range]

    result_tab = photometry(
        data=data, error=10 * error, init_params=guesses, mask=mask
    )

    return result_tab
