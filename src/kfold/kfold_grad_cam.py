import sys
import torch
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
plt.rcParams['font.size'] = 8

'''
This script is used to generate aggregate attributions for the CNN model over all 50 folds.
'''

def plot_attributions(mvts, attribution_mask, name=None):

    dtypes = ['94', '131', '171', '193', '211', '304', '335', '1600', '1700', 'M_45s', 'V_45s',
            'TOTUSJH', 'TOTBSQ', 'TOTPOT', 'TOTUSJZ', 'ABSNJZH', 'SAVNCPP', 'USFLUX', 'TOTFZ',
            'MEANPOT', 'EPSZ', 'MEANSHR', 'SHRGT45', 'MEANGAM', 'MEANGBT', 'MEANGBZ', 'MEANGBH',
            'MEANJZH', 'TOTFY', 'MEANJZD', 'MEANALP', 'TOTFX', 'EPSY', 'EPSX', 'XR_MAX']

    mvts = mvts.T
    attribution_mask = attribution_mask.T


    mvts[mvts==0] = np.nan
    mvts = unity_based_normalization(mvts) # normalize the data so easier to see each feature in the original data

    n_void = np.isnan(mvts[0]).sum()

    fig = plt.figure(figsize=(10, 25))
    gs = gridspec.GridSpec(5, 3, height_ratios=[1, 1, 1, 1, 4], width_ratios=[1, 1, 1], hspace=0.15, wspace=0.3)

    # plot input mvt
    ax1 = fig.add_subplot(gs[0:1, :])
    ax1.set_title(f'mvts obs: {name}')
    ax1.imshow(mvts, aspect='auto', cmap='gray', interpolation='spline16', vmin=0, vmax=np.nanmax(mvts), alpha=0.5)
    ax1.yaxis.grid(True, "major")
    ax1.xaxis.grid(True, "major")
    ax1.xaxis.set_major_locator(MultipleLocator(2))
    ax1.xaxis.set_minor_locator(MultipleLocator(1))
    ax1.yaxis.set_major_locator(MultipleLocator(5))
    ax1.yaxis.set_minor_locator(MultipleLocator(1))
    ax1.tick_params(which='major', length=5,width=1)
    ax1.tick_params(which='minor', length=3,width=1)

    # Plot saliency map
    ax2 = fig.add_subplot(gs[1, :])
    ax2.set_title('saliency map')
    attribution_mask[:,0:n_void] = np.nan
    ax2.imshow(attribution_mask, aspect='auto', cmap='jet', alpha=0.5, interpolation='spline16')
    ax2.yaxis.grid(True, "major")
    ax2.xaxis.grid(False, "major")
    ax2.xaxis.set_major_locator(MultipleLocator(2))
    ax2.xaxis.set_minor_locator(MultipleLocator(1))
    ax2.yaxis.set_major_locator(MultipleLocator(5))
    ax2.yaxis.set_minor_locator(MultipleLocator(1))
    ax2.tick_params(which='major', length=5,width=1)
    ax2.tick_params(which='minor', length=3,width=1)

    # Plot the light curves
    gs2 = gridspec.GridSpecFromSubplotSpec(6, 6, subplot_spec=gs[2, 0:], wspace=0, hspace=0)
    for i in range(6):
        for j in range(6):
            ind = (i * 6) + j
            if ind < len(dtypes):  # Check if index is within the range of dtypes
                ax = fig.add_subplot(gs2[i, j])
                # Handle NaN values in attribution_mask
                masked_mask = np.ma.masked_invalid(attribution_mask)
                # Normalize the masked_mask to range [0, 1]
                norm = mcolors.Normalize(vmin=masked_mask.min(), vmax=masked_mask.max())
                normalized_mask = norm(masked_mask)
                # Convert the normalized mask to RGB
                cmap = plt.cm.get_cmap('jet')
                rgb_colors = cmap(normalized_mask)
                light_curve = mvts[ind]
                light_curve_colors = rgb_colors[ind, :, :]
                for k, clr in enumerate(light_curve_colors):
                    if k + 1 == 40: break
                    x = [k, k+1]
                    y = [light_curve[k], light_curve[k+1]]
                    plt.plot(x, y, color=clr, linewidth=0.7)
                    plt.text(0.02, 0.9, dtypes[ind], fontsize=5, horizontalalignment='left', verticalalignment='center', transform=ax.transAxes)
                    plt.xticks([])
                    plt.yticks([])
                    plt.xlim(0,40)
    plt.tight_layout()
    plt.savefig(f'../../kfold_results/figs/{name}.png', bbox_inches='tight')
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

if __name__ == '__main__':

    # Load all data
    indices = np.arange(0, 485, 1)
    dataloader = DataLoader(MVTSDataset(indices, norm_type='standard'), batch_size=len(indices), shuffle=False, drop_last=False)
    data, _, labels = next(iter(dataloader))

    # Itterate over all mvts
    for indx, (x, y) in enumerate(zip(data, labels)):
        if y.item() == 0: continue
        name = indices[indx]
        x = x.unsqueeze_(0)
        x = x.unsqueeze_(1)
        x = x.requires_grad_()
        x = torch.nan_to_num(x)
        attribution_masks = []
        # Itterate over all models from each 50 folds 
        for model_indx in range(50):
            if model_indx in [27,37]: continue
            # Load best model for a specific fold
            model = CNNModel()
            model.load_state_dict(torch.load(f'../../models/cnn_model_standard_{model_indx}.pth'))
            model.eval();
            # Create a GuidedGradCam object based on the model and the desired layer
            guided_grad_cam = GuidedGradCam(model, model.conv3)
            # Compute the attribution mask for the desired class
            attribution_mask = guided_grad_cam.attribute(x, target=y)
            attribution_mask = attribution_mask.squeeze().detach().numpy()
            # attribution_mask = attribution_mask / np.nanmax(attribution_mask)
            attribution_mask = unity_based_normalization(attribution_mask)
            attribution_masks.append(attribution_mask)
            # delete model
            del model
        x = x.squeeze().detach().numpy()
        attribution_masks = np.array(attribution_masks)
        attribution_mask = np.nanmean(attribution_masks, axis=0)
        plot_attributions(x, attribution_mask, name=name)
        np.save(f'../../kfold_results/attributions/{name}.npy', attribution_mask)
