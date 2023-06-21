import os
import pickle
import numpy as np

from astropy.nddata import CCDData
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.stats import mad_std
from photutils.psf import PSFPhotometry, GriddedPSFModel, IterativePSFPhotometry

from .grid import _grid_model
from .fit import _fit_psf
from .romanisim import _synthesize_image, pixel_scale as PIX_SCALE
from .plots import _plot_residuals


__all__ = ['Trial']


class Trial:
    def __init__(self):
        # WebbPSF parameters:
        self.n_psfs = 3**2
        self.filter = 'F087'
        self.detector = 'SCA01'
        self.wavelength = (int(self.filter[1:]) / 100 * u.um).to_value(u.m)
        self.oversample = 10
        self.fov_pixels = 50

        # # STIPS parameters:
        # self.obs_ra = 150.0
        # self.obs_dec = -2.5
        # self.n_stars = 100
        # self.exptime = 1000

        # romanisim parameters:
        self.fov_coord = SkyCoord.from_name("M13")
        self.n_stars = 500
        self.faintmag = 26
        self.image_dir = 'images'
        self.append_to_file_path = ''

        # photutils parameters:
        self.fit_shape = (29, 29)
        self.crit_separation = 5
        self.pixel_scale = PIX_SCALE
        self.photometry_cls = PSFPhotometry

        # attrs we'll assign later:
        self.location_list = None
        self.grid_model = None
        self.true_pixel_coords = None
        self.true_sky_coords = None
        self.true_fluxes = None
        self.fit_results = None
        self.astrometric_residuals = None
        self.astrometric_rms = None
        self.relative_flux_residuals = None
        self.elapsed_time = None

    @property
    def path_grid(self):
        return (
            f'grids/psf_grid_model_{self.filter}_{self.detector.lower()}.fits'
        )

    @property
    def path_image(self):
        return (
            f'{self.image_dir}/synthetic_image_{self.filter}_{self.detector}_' +
            f'{self.n_stars:.0f}_{self.faintmag:.0f}_' + 
            f'{self.oversample:.0f}{self.append_to_file_path}.asdf'
        )
    
    @property
    def path_catalog(self):
        return (
            f'{self.image_dir}/synthetic_catalog_{self.filter}_{self.detector}_' +
            f'{self.n_stars:.0f}_{self.faintmag:.0f}_{self.oversample:.0f}' + 
            f'{self.append_to_file_path}.ecsv'
        )

    def construct_grid_model(self, visualize=False, overwrite=True):
        self.grid_model, self.location_list = _grid_model(
            filt=self.filter,
            filename=f'grids/psf_grid_model_{self.filter}',
            expected_filename=self.path_grid,
            detector=self.detector,
            oversample=self.oversample,
            fov_pixels=self.fov_pixels,
            n_psfs=self.n_psfs,
            visualize=visualize,
            overwrite=overwrite
        )

    def construct_image(self, overwrite=False):
        (
            self.true_pixel_coords,
            self.true_fluxes
        ) = _synthesize_image(
            input_coord=self.fov_coord,
            filt=self.filter,
            n_sources=self.n_stars,
            sca=int(self.detector[-2:]),
            output_path=self.path_image,
            catalog_path=self.path_catalog,
            overwrite=overwrite,
            faintmag=self.faintmag,
        )

    def fit_psf(self, progress_bar=False):
        self.fit_results = _fit_psf(
            crit_separation=5,
            gridmodel=self.grid_model,
            coords=self.true_pixel_coords,
            asdf_file=self.path_image,
            cls=self.photometry_cls,
            progress_bar=progress_bar
        )

    def stats(self):
        self.astrometric_residuals = np.array(np.hypot(
            self.fit_results['x_init'] - self.fit_results['x_fit'],
            self.fit_results['y_init'] - self.fit_results['y_fit']
        ))
        self.relative_flux_residuals = np.array((
            (self.fit_results['flux_init'] -
             self.fit_results['flux_fit']) /
            self.fit_results['flux_init']
        ))
        self.astrometric_rms = mad_std(
            (
                self.astrometric_residuals * u.pix * self.pixel_scale
            ).to(u.mas)
        )

    def plot_residuals(self):
        return _plot_residuals(
            result_tab=self.fit_results,
            pixel_scale=self.pixel_scale
        )

    def __repr__(self):
        s = "<Trial\n"

        print_attrs = [
            'filter',
            'detector',
            'wavelength',
            'n_stars',
            'astrometric_rms'
        ]

        for attr in print_attrs:
            res = getattr(self, attr)
            if not attr.startswith('_') and not callable(res) and res is not None:
                s += f'\t{attr}: {repr(res)}\n'
        return s + ">"

    def __getstate__(self):
        state = self.__dict__.copy()
        # Don't pickle these attrs:
        del state["grid_model"]
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        # Add attrs back that weren't in pickle:
        self.grid_model = None

    def save(self, path, overwrite=True):
        if not os.path.exists(path) or overwrite:
            with open(path, 'wb') as pkl:
                pickle.dump(self, pkl)

    @classmethod
    def load(cls, path):

        new = cls()

        with open(path, 'rb') as pkl:
            loaded = pickle.load(pkl)

        for attr in dir(loaded):
            res = getattr(loaded, attr)
            if not attr.startswith('_') and not callable(res):
                try:
                    setattr(new, attr, res)
                except AttributeError:
                    # triggered for attrs decorated with property:
                    pass

        ndd = CCDData.read(new.path_grid, unit=u.ct, ext=0)
        ndd.meta = dict(ndd.meta)
        ndd.meta['oversampling'] = new.oversample
        ndd.meta['grid_xypos'] = np.array([
            [list(tup) for tup in new.location_list]
            for _ in range(ndd.shape[0])
        ])
        new.grid_model = GriddedPSFModel(ndd)

        return new
