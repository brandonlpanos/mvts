import os 
import pickle
import numpy as np
import pandas as pd
import sunpy.timeseries as ts
import matplotlib.pyplot as plt
from sunpy.net import attrs as a
from sunpy.net import Fido as fido
from scipy.interpolate import interp1d
from matplotlib.ticker import MultipleLocator
from scipy.interpolate import pchip_interpolate

'''
pipeline downloads GOES X-ray flux data for each flare in the SWAN-SF dataset
'''

def plot_goes(flux_low_cad, save_goes_path):
    fig, axes = plt.subplots(figsize=(12,4))
    plt.plot(flux_low_cad, c='r', label='1.0 -- 8.0 Å')
    axes.set_yscale("log")
    axes.set_ylim(1e-7, 1e-3)
    axes.set_ylabel("Watts m$^{-2}$")
    labels = ['B', 'C', 'M', 'X']
    centers = np.logspace(-6.5, -3.5, len(labels))
    for value, label in zip(centers, labels):
        axes.text(1.02, value, label, transform=axes.get_yaxis_transform(), horizontalalignment='center')
    for i in range(len(flux_low_cad)):
        plt.axvline(i, c='k', linestyle='--', alpha=.2)
    axes.yaxis.grid(True, "major")
    axes.xaxis.grid(False, "major")
    axes.xaxis.set_major_locator(MultipleLocator(2))
    axes.xaxis.set_minor_locator(MultipleLocator(1))
    axes.tick_params(which='major', length=5,width=1)
    axes.tick_params(which='minor', length=3,width=1)
    axes.legend()
    plt.xlim(0,39)
    plt.tight_layout()
    if save_goes_path:
        plt.savefig(save_goes_path + '.png', bbox_inches='tight')
    plt.show()
    plt.close(fig)
    return None

def get_flare_locs(df_mvts):
    '''
    Returns a dictionary {'M' : arr[20, 103, ...], 'X': [40, 50...] }
    with keys indicating flare class and ndarrays the location of the flare max along the time series
    If no M or X, then key will not exist
    '''
    flare_loc_dic = {}
    pos_cls = ['MFLARE_LOC', 'XFLARE_LOC']
    for cls in pos_cls:
        flare_locs = df_mvts[cls].to_numpy()
        locs = np.squeeze(np.where(flare_locs==1))
        locs = np.atleast_1d(locs)
        if locs.size > 0:
            flare_loc_dic[cls[0]] = locs
    return flare_loc_dic

def extend(short_lightcurve, required_length):
    # Probably because of slight readout timings one of the channels is missing a few data points on the tail.
    n_missing = required_length - len(short_lightcurve) # number of missing values
    fill_value = short_lightcurve[-1] # repeat last value until reaching required_length
    extended_light_curve = np.concatenate( (short_lightcurve, np.full((n_missing,), fill_value)) ) # fill
    return extended_light_curve

def truncate(short_lightcurve, required_length):
    # If light curve slonger than suposed to be, then remove indicies from end of vector untill correct size
    remove_inds = np.arange(-1, -1 * required_length -1, -1)
    short_lightcurve = np.delete(short_lightcurve, remove_inds)
    return short_lightcurve

def degrade_xray(xray):
    '''
    GOES has a high cadance, this function degrades to the resolution of the mvts while 
    maintaining important peak infomation 
    This method creates a continuous curve through the data points using cubic polynomials,
    while ensuring that the resulting curve passes through all of the original data points.
    '''
    y = xray
    x = np.array(range(len(xray)))
    x_new = np.linspace(x.min(), x.max(), num=40)
    degraded_xray = pchip_interpolate(x, y, x_new)
    return degraded_xray

def fetch_goes_data(start_time, end_time):
    '''
    Fetches GOES X-ray flux in the 1-8 Å channel
    There are often multiple GOES instruments and sometimes multiple files for each instrument
    in-order to satisfy the time interval.
    This function creates a list of X-ray flux for each instrument over the given time interval
    '''
    # define the query to search for GOES XRS data
    query = fido.search(a.Time(start_time, end_time), a.Instrument('XRS'))
    # Download the data
    files = fido.fetch(query)
    # Create a dictionary to hold the data frames for each instrument
    my_dic = {}
    for file in files:
        # Extract the instrument name from the file name
        instrument = file.split('_')[2]
        # Check if the instrument is already in the dictionary
        if instrument not in my_dic:
            my_dic[instrument] = []
        # Load the data into a TimeSeries object
        ts_goes = ts.TimeSeries(file)
        # Convert the TimeSeries object to a pandas dataframe
        df = ts_goes.to_dataframe()
        # Append the dataframe to the list for this instrument
        my_dic[instrument].append(df)
    # Concatenate the data frames for each instrument into a single dataframe
    instrument_xray_flux = []
    for instrument in my_dic:
        df_instrument = pd.concat(my_dic[instrument], axis=0)
        cropped_df = df_instrument[start_time:end_time]
        flux_b = cropped_df['xrsb'].to_numpy()
        instrument_xray_flux.append(flux_b)
    return instrument_xray_flux


# Set time to 4 hours before flare max and derive quantities
time_interval = 4 # (hrs)
n_points = time_interval * 5 # because SWAN-SF has a cadence of 12 minutes
clean_window_thresh = 5 # define a clean window threshold (hrs)
base_cad = 6 * 60 # in seconds (the cadance of the final mvts series, i.e., 6 minutes)

# must only derive GOES curve for the obs below
pos_obs = os.listdir('/Users/brandonlpanos/gits/mvts/downloads_lowcad/my_mvts/')

# Iterate over NOAA active regions here
swan_original_data = '../SWAN/'
for partition in os.listdir(swan_original_data):
    if partition == '.DS_Store': continue
    for csv_file in os.listdir(f'../SWAN/{partition}/'):

        path_to_data = f'../SWAN/{partition}/{csv_file}'
        NOAA_AR = path_to_data.split('/')[-1][:-4]

        # Load SWAN-SF data series for a NOAA AR
        df_mvts = pd.read_csv(path_to_data, sep='\t')

        # Get locations of M- and X-class flares along the time series
        flare_loc_dic = get_flare_locs(df_mvts)

        if 'X' not in flare_loc_dic.keys(): continue
        # Get time for each point in the SWAN series
        time_strings = df_mvts['Timestamp'].to_numpy()

        # function to calculate times between SWAN-data points in hrs
        delta_t = lambda d1, d2: (abs(d1 - d2) * 12) / 60.
        # get all large flare locations on SWAN-SF data grid
        all_large_flare_locs = [ vals for _, vals in flare_loc_dic.items() ]
        all_large_flare_locs = [item for sublist in all_large_flare_locs for item in sublist] # flatten nested list
        for flare_cls, max_locs in flare_loc_dic.items():
            if flare_cls == 'M': continue
            # iterate through single flare locs
            for ii, max_loc in enumerate(max_locs):
                save_name = f'{NOAA_AR}_{max_loc}.npy'
                # if save_name not in pos_obs: continue
                try:
                    skip_flag = False
                    # compare distance in hrs between all other points
                    t_between_all_obs = np.array([ delta_t(max_loc, all_locs) for all_locs in all_large_flare_locs ])
                    # find where obs close together
                    within_interval = np.where(np.array(t_between_all_obs) < clean_window_thresh)[0]
                    # if close event comes before current event then skip obs
                    if len(within_interval > 1):
                        for ob in within_interval:
                            if all_large_flare_locs[ob] < max_loc:
                                skip_flag = True
                                break
                    # skip flare if there is another large flare within the threshold window "clean_window_thresh"
                    if skip_flag: continue 

                    loc = max_loc - n_points # select data unit in SWAN-SF dataset 3 hours before flare 

                    # Set the start and end times
                    start_time = time_strings[loc] 
                    end_time = time_strings[max_loc]
                    
                    # Derive X-ray flux for all avalable GOESE satalites
                    instrument_xray = fetch_goes_data(start_time, end_time)
                    
                    final_xray_list = []
                    for x_ray in instrument_xray:
                        try:
                            degraded_xray = degrade_xray(x_ray)
                        except:
                            continue
                        final_xray_list.append(degraded_xray)
                        
                    print(len(final_xray_list))
                        
                    if len(final_xray_list) > 1:
                        final_xray = np.vstack(final_xray_list)
                        # add all rows while keeping the maximum value of each index
                        final_xray = np.maximum.reduce(final_xray, axis=0)
                        
                    if len(final_xray_list) == 1:
                        final_xray = np.array(final_xray_list)
                        
                    np.save(f'../goes/{save_name}', final_xray)

                except: continue