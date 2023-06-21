import numpy as np
import astropy.units as u
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from astropy.stats import mad_std


def _plot_residuals(result_tab, pixel_scale):
    err_x = (result_tab['x_fit'] - (result_tab['x_init'])) * u.pix
    err_y = (result_tab['y_fit'] - (result_tab['y_init'])) * u.pix

    transformations = [
        lambda x: (x * pixel_scale).to_value(u.mas),
        #lambda x: x.value
    ]
    labels = ['mas']#, 'pix']

    for pixel_to_world, label in zip(transformations, labels):
        fig, ax = plt.subplots(2, 2, figsize=(6, 6), sharex='col', sharey='row')
        ax[1, 0].errorbar(
            pixel_to_world(err_x), pixel_to_world(err_y),
            xerr=pixel_to_world(result_tab['x_err'] * u.pix),
            yerr=pixel_to_world(result_tab['y_err'] * u.pix),
            fmt='none', ecolor='silver', zorder=-10, alpha=0.2
        )
        c = ax[1, 0].scatter(
            pixel_to_world(err_x), pixel_to_world(err_y),
            c=result_tab['flux_fit'], zorder=10, marker='o'
        )
        cbaxes = inset_axes(
            ax[1, 0], width="80%", height="3%",
            loc='upper center', borderpad=0.4
        )
        cbar = plt.colorbar(
            c, cax=cbaxes, orientation='horizontal',
            ax=ax[1, 0], location='bottom'
        )
        plt.setp(cbar.ax.get_xticklabels(), fontsize=8)

        ax[1, 0].set(
            xlabel=f'resid$_x$ [{label}]',
            ylabel=f'resid$_y$ [{label}]',
        )

        ax[0, 1].axis('off')
        msx = pixel_to_world(mad_std(err_x))
        width = 5
        lims = np.array([-width * msx, width * msx]) + pixel_to_world(np.median(err_x))
        ax[0, 0].hist(pixel_to_world(err_x), bins=10, range=lims)
        ax[0, 0].set(
            title=f'${{\\rm std}}(\Delta x) = {msx:.2f}$ {label}',
            xlim=lims
        )

        msy = pixel_to_world(mad_std(err_y))
        lims = np.array([-width * msy, width * msy]) + pixel_to_world(np.median(err_y))
        ax[1, 1].hist(pixel_to_world(err_y), bins=10, range=lims, orientation='horizontal')
        ax[1, 1].set(
            title=f'${{\\rm std}}(\Delta y) = {msy:.2f}$ {label}',
            ylim=lims
        )
    return fig, ax