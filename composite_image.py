import sys
import os
from schmear import Trial

n_stars_total = int(1.6e6)
os.makedirs(f'n_stars_{n_stars_total}', exist_ok=True)
number_of_iterations = 10

i = int(sys.argv[-1])

trial = Trial()
trial.n_stars = int(n_stars_total / number_of_iterations)
trial.oversample = 4
trial.fov_pixels = 20
trial.fit_shape = (29, 29)
trial.image_dir = 'n_stars_1600000'
trial.append_to_file_path = f'_{i:02d}'
trial.construct_image()