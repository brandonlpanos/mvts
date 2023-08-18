import os
import sys
import torch
import pickle
import numpy as np
sys.path.insert(0, "../../src")
from models import CNNModel
import matplotlib.pyplot as plt
from datasets import MVTSDataset
import matplotlib.colors as mcolors
from captum.attr import GuidedGradCam
import matplotlib.gridspec as gridspec
from torch.utils.data import DataLoader
from matplotlib.ticker import MultipleLocator
import torch.nn.functional as F
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from astropy.io import fits
from astropy.wcs import WCS
from astropy import units as u

import sunpy.map
import sunpy.data.sample  # This line is only necessary if you want to use sample data
from sunpy.visualization.colormaps import color_tables as ct

def plot_attributions(mvts, attribution_mask, obs_date_time, max_val=None, name=None):

    plt.rcParams['font.size'] = 8
    global mean_probability
    

    dtypes = ['94', '131', '171', '193', '211', '304', '335', '1600', '1700', 'M_45s', 'V_45s',
            'TOTUSJH', 'TOTBSQ', 'TOTPOT', 'TOTUSJZ', 'ABSNJZH', 'SAVNCPP', 'USFLUX', 'TOTFZ',
            'MEANPOT', 'EPSZ', 'MEANSHR', 'SHRGT45', 'MEANGAM', 'MEANGBT', 'MEANGBZ', 'MEANGBH',
            'MEANJZH', 'TOTFY', 'MEANJZD', 'MEANALP', 'TOTFX', 'EPSY', 'EPSX', 'XR_MAX']

    mvts = mvts.T
    attribution_mask = attribution_mask.T

    mvts[mvts==0] = np.nan
    mvts = unity_based_normalization(mvts) # normalize the data so easier to see each feature in the original data

    n_void = np.isnan(mvts[0]).sum()

    percent_point = max_val / 40
    mx = int(180 * percent_point)
    mx = 180 - mx

    fig = plt.figure(figsize=(6, 25))
    gs = gridspec.GridSpec(5, 3, height_ratios=[1, 1, 1, 1, 4], width_ratios=[1, 1, 1], hspace=0.2, wspace=0.3)
    with open('/home/panosb/custom_cmap.pkl', 'rb') as f: custom_cmap = pickle.load(f)

    # plot input mvt
    ax1 = fig.add_subplot(gs[0:1, :])
    ax1.set_title(f'Date-Time: {obs_date_time}  mvts prob: {mean_probability:.2f}')
    ax1.set_ylabel('Features')
    ax1.set_xlabel('Time (min) before flare start')
    im1 = ax1.imshow(mvts, aspect='auto', cmap='gray', interpolation='spline16', vmin=0, vmax=np.nanmax(mvts), extent=[180, 0, 35, 0], alpha=0.5)
    ax1.yaxis.grid(True, "major")
    ax1.xaxis.grid(True, "major")
    ax1.xaxis.set_major_locator(MultipleLocator(20))
    ax1.xaxis.set_minor_locator(MultipleLocator(10))
    ax1.yaxis.set_major_locator(MultipleLocator(5))
    ax1.yaxis.set_minor_locator(MultipleLocator(1))
    ax1.tick_params(which='major', length=5,width=1)
    ax1.tick_params(which='minor', length=3,width=1)
    # Plot vertical line at max value of attribution mask
    ax1.axvline(x=mx, color='w', linestyle='--', linewidth=1)
    cbar1 = plt.colorbar(im1, cax=ax1.inset_axes([1.01, 0, 0.01, 1]), aspect=10)
    cbar1.set_label('Intensity')
    cbar1.ax.set_yticks([]) 

    # Plot saliency map
    ax2 = fig.add_subplot(gs[1, :])
    ax2.set_title('saliency map')
    ax2.set_ylabel('Features')
    ax2.set_xlabel('Time (min) before flare start')
    attribution_mask[:,0:n_void] = np.nan
    im2 = ax2.imshow(attribution_mask, aspect='auto', cmap=custom_cmap, interpolation='spline16', extent=[180, 0, 35, 0], alpha=1)
    ax2.yaxis.grid(True, "major")
    ax2.xaxis.grid(True, "major")
    ax2.xaxis.set_major_locator(MultipleLocator(20))
    ax2.xaxis.set_minor_locator(MultipleLocator(10))
    ax2.yaxis.set_major_locator(MultipleLocator(5))
    ax2.yaxis.set_minor_locator(MultipleLocator(1))
    ax2.tick_params(which='major', length=5,width=1)
    ax2.tick_params(which='minor', length=3,width=1)
    # Plot vertical line at max value of attribution mask
    ax2.axvline(x=mx, color='w', linestyle='--', linewidth=1)
    cbar2 = plt.colorbar(im2, cax=ax2.inset_axes([1.01, 0, 0.01, 1]), aspect=10)
    cbar2.set_label('Intensity')
    cbar2.ax.set_yticks([]) 

    # Plot the light curves
    gs2 = gridspec.GridSpecFromSubplotSpec(6, 6, subplot_spec=gs[2, 0:], wspace=0, hspace=0)
    ax3 = fig.add_subplot(gs2[:, :])
    ax3.set_title('features (light curves)')
    for i in range(6):
        for j in range(6):
            ind = (i * 6) + j
            if ind < len(dtypes):  # Check if index is within the range of dtypes
                ax = fig.add_subplot(gs2[i, j])
                # Plot vertical line at max value of attribution mask
                ax.axvline(x=max_val, color='k', linestyle='--', linewidth=0.5)
                # Handle NaN values in attribution_mask
                masked_mask = np.ma.masked_invalid(attribution_mask)
                # Normalize the masked_mask to range [0, 1]
                norm = mcolors.Normalize(vmin=masked_mask.min(), vmax=masked_mask.max())
                normalized_mask = norm(masked_mask)
                # Convert the normalized mask to RGB
                cmap = custom_cmap
                rgb_colors = cmap(normalized_mask)
                light_curve = mvts[ind]
                light_curve_colors = rgb_colors[ind, :, :]
                for k, clr in enumerate(light_curve_colors):
                    if k + 1 == 40: break
                    x = [k, k+1]
                    y = [light_curve[k], light_curve[k+1]]
                    plt.plot(x, y, color=clr, linewidth=1.5)
                    plt.text(0.02, 0.85, dtypes[ind], fontfamily='Arial', fontsize=5, fontweight='ultralight', horizontalalignment='left', verticalalignment='center', transform=ax.transAxes)
                    plt.xticks([])
                    plt.yticks([])
                    plt.xlim(0,40)
    ax3.set_xticks([])
    ax3.set_yticks([])
    plt.tight_layout()
    if name is not None:
        plt.savefig(f'/home/panosb/sml/bpanos/mvts/figs/grad_cam/{name}.pdf', bbox_inches='tight')
    plt.show()
    plt.close(fig)

    return None

def unity_based_normalization(data):
    '''
    Normalize each row of the data matrix by subtracting its minimum value, dividing by its range, and scaling to a range of 0-1
    Takes in arrays of shape (features, time)
    '''
    # Normalize each row by subtracting its minimum value, dividing by its range, and scaling to a range of 0-1
    # Get the maximum and minimum values of each row
    max_vals = np.nanmax(data, axis=1)
    min_vals = np.nanmin(data, axis=1)
    # Compute the range of each row, and add a small constant to avoid division by zero
    ranges = max_vals - min_vals
    eps = np.finfo(data.dtype).eps  # machine epsilon for the data type
    ranges[ranges < eps] = eps
    # Normalize each row by subtracting its minimum value, dividing by its range, and scaling to a range of 0-1
    data = (data - min_vals[:, np.newaxis]) / ranges[:, np.newaxis]
    data = data + np.nanmax(data)
    data *= (1 / np.nanmax(data, axis=1)[:, np.newaxis])
    return data

def plot_fits_images(fits_paths, save_name=None):
    '''
    fits_paths is a list of paths to fits files
    Plots a grid of images from the fits files
    '''
    
    plt.rcParams['font.size'] = 5 
    
    fig = plt.figure(figsize=(15, 12))
    axs = []

    for idx, fits_path in enumerate(fits_paths):
        with fits.open(fits_path) as hdul:
            header = hdul[1].header
            data = hdul[1].data

        wcs_sliced = WCS(header, naxis=[1, 2])

        if idx < len(axs):
            ax = axs[idx]
        else:
            ax = fig.add_subplot(3, 4, idx+1, projection=wcs_sliced)
            axs.append(ax)

        obs_date_time = header.get('DATE-OBS', "Unknown Date")
        vmin_val, vmax_val = np.percentile(data, [1, 99])

        if 'aia' in fits_path:
            # data = data * header['DN_GAIN'] 
            vmin_val, vmax_val = np.percentile(data, [1, 99])
            aia_map = sunpy.map.Map(data, header)
            channel = int(aia_map.meta['wavelnth'])
            channel_wavelength = channel * u.angstrom
            cmap = ct.aia_color_table(channel_wavelength)
            title = f'AIA {channel} Date-Time: {obs_date_time}'
            c_bar_label = 'Intensity [DN/s]'
        elif 'magnetogram' in fits_path:
            cmap = 'binary_r'
            title = f'Magnetogram Date-Time: {obs_date_time}'
            c_bar_label = 'Magnetic Field Strength [Gauss]'
        elif 'Dopplergram' in fits_path:
            cmap = 'coolwarm'
            title = f'Dopplergram Date-Time: {obs_date_time}'
            data = data - np.mean(data)
            vmin_val, vmax_val = np.percentile(data, [1, 99])
            c_bar_label = 'Velocity [m/s]'

        ax.imshow(data, origin='lower', cmap=cmap, vmin=vmin_val, vmax=vmax_val, alpha=0.7, aspect='equal')
        ax.set_title(title)
        ax.coords.grid(True, color='white', ls='--')
        ax.coords[0].set_axislabel('Solar X [arcsec]')
        ax.coords[1].set_axislabel('Solar Y [arcsec]')

        # Using inset_axes to create space for the colorbar within the ax
        cbar_ax = inset_axes(ax, width='3%', height='100%', loc='lower left',
                             bbox_to_anchor=(1.05, 0., 1, 1), bbox_transform=ax.transAxes, borderpad=0)
        cbar = plt.colorbar(ax.images[0], cax=cbar_ax, orientation='vertical')
        cbar.set_label(c_bar_label)

    plt.subplots_adjust(hspace=-0.7, wspace=0.5)  # adjust these values as needed
    if save_name is not None:
        plt.savefig(f'/home/panosb/sml/bpanos/mvts/figs/collage/{save_name}.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return obs_date_time


if __name__ == '__main__':

    # get index from unshuffled data loader

    # load csv to fits map (fix path)
    with open('/home/panosb/sml/bpanos/mvts/csv_to_fits_map.p', 'rb') as f:
        csv_to_fits_map = pickle.load(f)

    # Load all data
    indices = np.arange(0, 485, 1)
    dataloader = DataLoader(MVTSDataset(indices, norm_type='standard'), batch_size=len(indices), shuffle=False, drop_last=False)
    data, _, labels = next(iter(dataloader))

    # Calculate mean and std probability for each sample
    csv_indx = 0
    for mvts, y in zip(data, labels):
        mvts = mvts.unsqueeze_(0)
        mvts = mvts.unsqueeze_(1)
        mvts = mvts.requires_grad_()
        mvts = torch.nan_to_num(mvts)

        # Itterate over all models from each 50 folds #? I may have to push these to the server
        probability_for_all_models = []
        attribution_masks = []
        for model_indx in range(50):
            if model_indx in [27,37]: continue
            # Load best model for a specific fold
            model = CNNModel()
            model.load_state_dict(torch.load(f'../../models/cnn_std/{model_indx}.pth', map_location=torch.device('cpu')))
            model.eval();

            # Create a GuidedGradCam object based on the model and the desired layer
            guided_grad_cam = GuidedGradCam(model, model.conv3)
            # Compute the attribution mask for the desired class
            attribution_mask = guided_grad_cam.attribute(mvts, target=y)
            attribution_mask = attribution_mask.squeeze().detach().numpy()
            # attribution_mask = attribution_mask / np.nanmax(attribution_mask)
            attribution_mask = unity_based_normalization(attribution_mask)
            attribution_masks.append(attribution_mask)

            # get prediction score
            pred = model(mvts)
            probabilities = F.softmax(pred, dim=1)
            positive_class_probability = probabilities[0][1].item()
            probability_for_all_models.append(positive_class_probability)
            del model

        # store the mean probability for all models #? I need to include these values in my plot title
        mean_probability = torch.mean(torch.tensor(probability_for_all_models))
        std_probability = torch.std(torch.tensor(probability_for_all_models))

        # Plot the attribution mask
        mvts = mvts.squeeze().detach().numpy()
        attribution_masks = np.array(attribution_masks)
        attribution_mask = np.nanmean(attribution_masks, axis=0)
        # Find max value in attribution mask as a mean along the features
        max_val = np.nanmax(attribution_mask, axis=1)
        max_val_t_loc = np.nanargmax(max_val)

        # find location to fits files #? could fix the acuracy on the time here
        noaa = csv_to_fits_map['csv_'+str(csv_indx)][0]
        deep_path = f'/home/panosb/sml/bpanos/old/mvts/downloads_lowcad/{noaa}/'
        files_for_active_region = [file for file in os.listdir(deep_path)]
        max_loc = csv_to_fits_map['csv_' + str(csv_indx)][1]

        for file in files_for_active_region:
            if str(max_loc) in file:
                leaf_name = file
        
        deep_path = deep_path + leaf_name + '/'

        # plot aia and hmi image collage
        fits_paths = []
        for subdir in os.listdir(deep_path):
            if subdir == 'ndarrays' or subdir == '.DS_Store' or subdir == 'vids': continue
            files_in_sub = sorted(os.listdir(deep_path + subdir))

            file_loc = max_val_t_loc - (40 - len(files_in_sub))

            fits_paths.append(deep_path + subdir + '/' + files_in_sub[file_loc])

        obs_date_time = plot_fits_images(fits_paths, save_name=f'collage_{csv_indx}_prob_{mean_probability:.2f}')

        # plot agrigate heatmap and data for the sample
        plot_attributions(mvts, attribution_mask, obs_date_time, max_val=max_val_t_loc, name=f'heatmap_{csv_indx}_prob_{mean_probability:.2f}')

        csv_indx += 1