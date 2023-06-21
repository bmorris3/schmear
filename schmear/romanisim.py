import os
from copy import deepcopy

import numpy as np
import asdf
import astropy.units as u
from galsim import UniformDeviate
from romanisim import persistence, wcs
from romanisim.catalog import make_stars
from romanisim.image import simulate
from romanisim.parameters import default_parameters_dictionary


pixel_scale = 0.11 * u.arcsec / u.pix
fov_corner_to_center = (
    4092 * u.pix * pixel_scale
).to(u.deg)


def _synthesize_image(
        input_coord,
        filt='F087', n_sources=500,
        output_path='synthetic_image.asdf', sca=7,
        catalog_path='star-catalog.ecsv',
        seed=0,
        radius=2 * fov_corner_to_center.to_value(u.deg),
        overwrite=False,
        faintmag=26,
        scale_up_n_sources=13,
    ):
    if not os.path.exists(output_path) or overwrite:
        cat = make_stars(
            coord=input_coord, n=int(n_sources * scale_up_n_sources),
            radius=radius, bandpasses=[filt],
            faintmag=faintmag,
            index=3/5
        )
        cat.write(catalog_path, overwrite=True)

        # prepare inputs for the `romanisim.image.simulate` method:
        metadata = deepcopy(default_parameters_dictionary)
        metadata['instrument']['detector'] = f'WFI{sca:02d}'
        metadata['instrument']['optical_element'] = filt
        metadata['exposure']['ma_table_number'] = 1
        metadata['pointing'] = {
            'ra_v1': input_coord.ra.degree,
            'dec_v1': input_coord.dec.degree
        }
        metadata['wcsinfo'] = {
            'ra_ref': input_coord.ra.degree,
            'dec_ref': input_coord.dec.degree,
            'v2_ref': 0,
            'v3_ref': 0,
            'roll_ref': 0,
        }

        wcs.fill_in_parameters(
            metadata, input_coord, boresight=False
        )
        # run the simulation:
        im, simcatobj = simulate(
            metadata, cat, webbpsf=True, level=2,
            persistence=persistence.Persistence(),
            rng=UniformDeviate(seed), usecrds=False
        )

        af = asdf.AsdfFile()
        romanisimdict = {'simcatobj': simcatobj}
        af.tree = {'roman': im, 'romanisim': romanisimdict}
        af.write_to(output_path)

    with asdf.open(output_path) as asdf_file:
        simcatobj = np.array([list(x) for x in asdf_file.tree['romanisim']['simcatobj']])

    pixel_coords = simcatobj[:, :2]
    true_flux = simcatobj[:, 2]

    return pixel_coords, true_flux