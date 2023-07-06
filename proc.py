import os
import sys
import logging
from itertools import product
import numpy as np
from schmear import Trial
from time import time
from photutils.psf import PSFPhotometry, IterativePSFPhotometry

ris_logger = logging.getLogger('romanisim')
ris_logger.setLevel(logging.ERROR)

n_pix = 4096 ** 2

# estimate the number of pixels per star in the SCA. Near galactic
# center ~ 10, and near the pole ~ 400
# pixels_per_star_grid = [500]
# n_stars_grid = [int(n_pix / pps) for pps in pixels_per_star_grid]

n_stars_grid = [500]
oversample_grid = [4, 8, 12]
fov_pixels_grid = [9, 15, 21, 27]
fit_shape_grid = [5, 7, 11, 13, 15, 25]

faintmag_grid = [21]
phot_cls_grid = [PSFPhotometry]  #, IterativePSFPhotometry]
n_psfs_grid = [8**2] # np.power(np.arange(3, 8), 2).tolist()

filter_grid = "F087, F106, F129, F158, F184".split(', ')
# Excluded filters listed in webbpsf but not in galsim: 'F062', 'F146', 'F213'

args = [
    n_stars_grid, 
    oversample_grid, 
    fov_pixels_grid, 
    fit_shape_grid,
    phot_cls_grid,
    faintmag_grid,
    n_psfs_grid,
    filter_grid
]

parameters = list(product(*args))
n_threads_candidates = np.arange(1, 3)
iters_per_thread = len(parameters) / n_threads_candidates
max_ind = np.argwhere(np.mod(iters_per_thread, 1) == 0).max()
n_threads = n_threads_candidates[max_ind]
n_iters_per_thread = int(iters_per_thread[max_ind])

if __name__ == '__main__':

    edges = n_iters_per_thread * np.arange(n_threads + 1)
    iter_index = int(sys.argv[-1])
    overwrite = False

    for params in parameters[edges[iter_index]:edges[iter_index+1]]:
        n_stars, oversample, fov_pixels, fit_shape, phot_cls, faintmag, n_psf, use_filter = params

        strings = []
        
        for p in params: 
            if not isinstance(p, tuple) and p not in phot_cls_grid:
                strings.append(f"{p}".replace(' ', ''))
            elif p in phot_cls_grid:
                strings.append(p.__name__)
            else: 
                strings.append(f"{p[0]}")

        save_path = f"outputs/trial_{'_'.join(strings)}.pkl"

        if not os.path.exists(save_path):

            trial = Trial()
            trial.n_stars = n_stars
            trial.filter = use_filter
            trial.oversample = oversample
            trial.fov_pixels = fov_pixels
            trial.fit_shape = fit_shape
            trial.n_psfs = n_psf
            trial.faintmag = faintmag
            trial.photometry_cls = phot_cls
            # pbar.set_description('constructing synthetic image')
            trial.construct_image(overwrite=overwrite)
            # pbar.set_description('constructing PSF grid model')
            trial.construct_grid_model(overwrite=overwrite)
            # pbar.set_description('fitting PSF model to image')
            start_time = time()
            trial.fit_psf(progress_bar=False)
            stop_time = time()

            trial.stats()
            trial.elapsed_time = stop_time - start_time

            trial.save(save_path)