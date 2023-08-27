import os
import drms
import types
import sunpy
import numpy as np
import pandas as pd
import astropy.units as u
from astropy.io import fits
from astropy.time import Time
from datetime import datetime
from scipy import interpolate
from matplotlib import rcParams
import matplotlib.pyplot as plt
from sunpy.coordinates import frames
from astropy.coordinates import SkyCoord
from IPython.display import display, HTML
import matplotlib.animation as manimation
from matplotlib.ticker import MultipleLocator
from matplotlib.animation import FuncAnimation


def get_bounding_box(df_mvts: pd.DataFrame, loc: int) -> (str, int, int, int, int):
    '''
    Takes in SWAN-SF DataFrame and a row index loc
    outputs tuple with time, x_cent, y_cent, box_width, and box_height all in arcseconds
    This is a necessary set of inputs for the process argument that allows us to track patches of the sun
    '''
    t_ref = df_mvts['Timestamp'][loc]
    # coords in degrees
    lat_max = df_mvts['LAT_MAX'][loc]
    lat_min = df_mvts['LAT_MIN'][loc]
    lon_max = df_mvts['LON_MAX'][loc]
    lon_min = df_mvts['LON_MIN'][loc]
    # defined in original coord system
    c = SkyCoord( lon_min, lat_min, unit='deg', frame=frames.HeliographicStonyhurst, observer='earth', obstime=t_ref)
    c = c.transform_to(frames.Helioprojective) # convert into arcsec
    x1, y1 = c.Tx.value, c.Ty.value
    c = SkyCoord( lon_max, lat_max, unit='deg', frame=frames.HeliographicStonyhurst, observer='earth', obstime=t_ref)
    c = c.transform_to(frames.Helioprojective) # convert into arcsec
    x2, y2 = c.Tx.value, c.Ty.value
    x_cent = (x1 + x2) / 2.
    y_cent = (y1 + y2) / 2.
    box_width = abs(x2 - x1)
    box_height = abs(y2 - y1)
    return (t_ref, x_cent, y_cent, box_width, box_height)

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

def animate(data_cube, paths, name=377, segment='AIA171',save_path='~/vid.mp4', figsize=(8,8), use_log=1):
    '''
    Function that takes as input a datacube and produces an animation
    input: data_cube --> numpy array (index, x, y)
           name --> SHARP #
           segment --> observable, i.e., line-of-sight magnetic field (Br)
           save_path --> path to save the animation
    output: saved animation of datacube 
    '''
    # get header infomation
    hdul = fits.open(path)
    hdr = hdul[1].header
    
    rcParams['font.family'] = 'serif'
    rcParams['font.size'] = 15
    fig, axis = plt.subplots( figsize=figsize )
    NOAA = str(name)
    Frame = 0
    time = paths[0].split('.')[3][:-1]
    plt.title(f"{segment} (NOAA: {NOAA}) Frame: {Frame} Time: {time}", fontsize=10, c='white', y=.92)
    if use_log == 1:
        image = np.log(data_cube[0,:,:])
    if use_log == 0:
        image = data_cube[0,:,:]
    im = plt.imshow( image, cmap="binary")
    fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None)
    plt.tight_layout()
    plt.axis('off')
    plt.close( fig )
    def init():
        return im,
    def update(i):
        time = paths[i].split('.')[3][:-1]
        if use_log == 1:
            im_data = np.log(data_cube[i,:,:])
        if use_log == 0:
            im_data = data_cube[i,:,:]
        im.set_data( im_data )
        im.axes.set_title(f"{segment} (NOAA: {NOAA}) Frame: {i} Time: {time}", fontsize=10, c='white', y=.92)
        fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None)
        return im,
    anim = FuncAnimation( fig, lambda i: update(i),
                          init_func=init,
                          frames=len(data_cube),
                          interval=200, blit=True )
    if save_path is not None: anim.save( save_path )
    plt.close(fig)
    return None

def interpolate_missing_vals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Interpolates the missing values (i.e., NaN's) using linear interpolation. That
    is, for any group of consecutive missing values, it treats the values as equally
    spaced numbers between the present values before and after the gap.
    This does not impact non-numerical values.
    :return: the interpolated version of the given dataframe.
    """
    if df.isna().sum().sum() != 0:
        return df.interpolate(method='linear', axis=0, limit_direction='both')
    else:
        return df

def extend(short_lightcurve, required_length):
    # Probably because of slight readout timings one of the channels is missing a few data points on the tail.
    n_missing = required_length - len(short_lightcurve) # number of missing values
    fill_value = short_lightcurve[-1] # repeat last value until reaching required_length
    extended_light_curve = np.concatenate( (short_lightcurve, np.full((n_missing,), fill_value)) ) # fill
    return extended_light_curve

def truncate(short_lightcurve, required_length):
    # If light curve is longer than suposed to be, then remove indicies from end of vector untill correct size
    remove_inds = np.arange(-1, -1 * required_length -1, -1)
    short_lightcurve = np.delete(short_lightcurve, remove_inds)
    return short_lightcurve

def interp_to_high_cad(t_old, y_old, t_grid_high_cad):
    # Linearly interpolate light curve onto high cadence time grid set by the euv channel
    f = interpolate.interp1d(t_old, y_old, fill_value="extrapolate")
    return f(t_grid_high_cad)

# Start here for the velocity correction
def use_bitmap(data, bitmap):
    bitmap = torch.where(bitmap <= 16, 1, 0)
    conv = torch.mul(data,bitmap)
    mask = conv != 0 # mask all the zeros
    norm = torch.mean(conv[mask])
    data = torch.subtract(data,norm)
    return data


if __name__ == '__main__':

    # Select the NOAA active region of interest, as well as the partition it belongs to
    NOAA_AR = '3364'
    partition = 'partition3'

    # All SDO instruments of interest and their associated channels
    euv_data_types = {'lev1_euv_12s': ['94','131','171','193','211','304','335']}
    uv_data_types = {'lev1_uv_24s': ['1600', '1700']}
    hmi_data_m = {'M_45s': 'mag'}
    hmi_data_v = {'V_45s': 'vel'}

    # Set time to 4 hours before flare max and derive quantities
    time_interval = 210 # original, changed for long obs 4 # (hrs)
    n_points = time_interval * 5 # because SWAN-SF has a cadence of 12 minutes
    clean_window_thresh = 5 # define a clean window threshold (hrs)
    base_cad = 6 * 60 # in seconds (the cadance of the final mvts series, i.e., 6 minutes)

    # Number of actual data points
    euv_cad_n = int((time_interval * 60 * 60 ) / base_cad)
    uv_cad_n = int((time_interval * 60 * 60 ) / base_cad)
    mag_cad_n = int((time_interval * 60 * 60 ) / base_cad)
    sharp_cad_n = int((time_interval * 60 * 60 ) / 720)

    # Define time grids
    t_uv_grid = np.linspace(1, euv_cad_n, uv_cad_n)
    t_mag_grid = np.linspace(1, euv_cad_n, mag_cad_n)
    t_sharp_grid = np.linspace(1, euv_cad_n, sharp_cad_n)

    # This is the cadence all observations are interpolated to
    t_grid_high_cad = np.linspace(1, euv_cad_n, euv_cad_n)

    # Make directory for final mvts
    save_vids_folder = f'../downloads_lowcad/my_mvts'
    out_dir = os.path.join(save_vids_folder)
    if not os.path.exists(out_dir): os.makedirs(out_dir)

    # Iterate over NOAA active regions here
    base_save_dir = '../downloads_lowcad/'
    swan_original_data = '../SWAN/'

    save_dir_ar = f'{base_save_dir}/{NOAA_AR}/'
    path_to_data = f'../SWAN/{partition}/{NOAA_AR}.csv'
    # Load SWAN-SF data series for a NOAA AR
    df_mvts = pd.read_csv(path_to_data, sep='\t')
    # Get locations of M- and X-class flares along the time series
    flare_loc_dic = get_flare_locs(df_mvts)
    # Get time for each point in the SWAN series
    time_strings = df_mvts['Timestamp'].to_numpy()
    # Create DRMS client
    client = drms.Client()

    flare_cls = 'M'
    # max_loc = 1543
    max_loc = 1600

    save_dir_ar_obs = f'{save_dir_ar}{flare_cls}_{max_loc}' # unique save directory e.g., X_420
    loc = max_loc - n_points # select data unit in SWAN-SF dataset 3 hours before flare
    print(f'loc: {loc}', f'max_loc: {max_loc}', f'n_points: {n_points}')
    print(save_dir_ar_obs)

    # Make a directory for pre-flare obs ndarrays after converting from fits 
    save_ndarray_folder = f'{save_dir_ar_obs}/ndarrays'
    out_dir = os.path.join(save_ndarray_folder)
    if not os.path.exists(out_dir): os.makedirs(out_dir)

    # Make a directory for videos
    save_vids_folder = f'{save_dir_ar_obs}/vids'
    out_dir = os.path.join(save_vids_folder)
    if not os.path.exists(out_dir): os.makedirs(out_dir)

    # Derive the bounding box parameters and time for the initial frame set three hours before flare maximum
    t_ref, x_cent, y_cent, box_width, box_height = get_bounding_box(df_mvts, loc)
    t_reform1 = t_ref.replace(' ', '_') # need to change time format for function
    print(f"t_ref: {t_ref}, x_cent: {x_cent}, y_cent: {y_cent}, box_width: {box_width}, box_height: {box_height}, t_reform1: {t_reform1}")

    mvts_light_curves = [] # container to assemble all light curves for a particular obs

    #? euv channel (12s) --> 6 minutes
    for data_type, channels in euv_data_types.items():
        # Itterate over bands
        for band in channels:
            print(band)
            # Make a directory for saved filter images
            band_save_dir = f'{save_dir_ar_obs}/{band}/'
            out_dir = os.path.join(band_save_dir)
            if not os.path.exists(out_dir): os.makedirs(out_dir)
            # Construct the drms query string: either "series[timespan][wavelength]"
            qstr = 'aia.lev1_euv_12s['+t_reform1+'/264h@6m]['+band+']{image}' # =changed from the original 4h@6m to 264h@6m
            # Reformulate the time argument
            t_reform2 = t_ref.replace(' ', 'T')
            # Download a seiries of images given the bounding boxes
            process = {'im_patch': {'t_ref': t_reform2,
                                    't': 0,
                                    'r': 0,
                                    'c': 0,
                                    'locunits': 'arcsec',
                                    'boxunits': 'arcsec',
                                    'x': x_cent,
                                    'y': y_cent,
                                    'width': box_width,
                                    'height': box_height}}
            result = client.export(qstr, method='url',protocol='fits',email='brandonlpanos@gmail.com', process=process)
            result.download(out_dir)
            
            # Convert all images to numpy arrays and save
            files = os.listdir(band_save_dir)
            files.sort() # Time order fits files
            paths = [ band_save_dir + '/' + file for file in files if '.fits' in file ]
            cube = [] # Create cube of images
            for path in paths:
                # get header infomation
                hdul = fits.open(path)
                hdr = hdul[1].header
                exposure_time = hdr['EXPTIME']
                image = fits.getdata(path, ext=1) / exposure_time
                cube.append(image)
            cube = np.array(cube)
            ndarray_save_path = f'{save_ndarray_folder}/{band}'
            np.save(ndarray_save_path, cube) # saves array (time, x, y)
            
            # Create a video of the datacube by calling the animation function
            save_vid_path = f'{save_vids_folder}/{band}.mp4'
            ar = f'{NOAA_AR}_{flare_cls}_{max_loc}'
            animate(cube, paths, name=ar, segment=band, save_path=save_vid_path, figsize=(8,8))
            
            # Turn images into light curves
            light_curve = np.array([ np.nanmean(im) for im in cube ])
            # Fill in missing tails with last real value
            if len(light_curve) < euv_cad_n: light_curve = extend(light_curve, euv_cad_n)
            if len(light_curve) > euv_cad_n: light_curve = truncate(light_curve, euv_cad_n)
            # Append all SDO light curves and turn into a numpy array
            mvts_light_curves.append(light_curve)


    #? uv channel (24s)
    for data_type, channels in uv_data_types.items():
        # Iterate over bands
        for band in channels:
            # Make a directory for saved filter images
            band_save_dir = f'{save_dir_ar_obs}/{band}/'
            out_dir = os.path.join(band_save_dir)
            if not os.path.exists(out_dir): os.makedirs(out_dir)
            # Construct the drms query string: either "series[timespan][wavelength]"
            qstr = 'aia.lev1_uv_24s['+t_reform1+'/264h@6m]['+band+']{image}'
            # Download a seiries of images given the bounding boxes
            process = {'im_patch': {'t_ref': t_reform2,
                                    't': 0,
                                    'r': 0,
                                    'c': 0,
                                    'locunits': 'arcsec',
                                    'boxunits': 'arcsec',
                                    'x': x_cent,
                                    'y': y_cent,
                                    'width': box_width,
                                    'height': box_height}}
            result = client.export(qstr, method='url',protocol='fits',email='brandon.panos@fhnw.ch', process=process)
            result.download(out_dir)
            
            # Convert all images to numpy arrays and save
            files = os.listdir(band_save_dir)
            files.sort() # Time order fits files
            paths = [ band_save_dir + '/' + file for file in files if '.fits' in file ]
            cube = [] # Create cube of images
            for path in paths:
                # get header infomation
                hdul = fits.open(path)
                hdr = hdul[1].header
                exposure_time = hdr['EXPTIME']
                image = fits.getdata(path, ext=1) / exposure_time
                cube.append(image)
            cube = np.array(cube)
            ndarray_save_path = f'{save_ndarray_folder}/{band}'
            np.save(ndarray_save_path, cube) # saves array (time, x, y)

            # Create a video of the datacube by calling the animation function
            save_vid_path = f'{save_vids_folder}/{band}.mp4'
            ar = f'{NOAA_AR}_{flare_cls}_{max_loc}'
            animate(cube, paths, name=ar, segment=band, save_path=save_vid_path, figsize=(8,8))
            
            # Turn images into light curves
            light_curve = np.array([ np.nanmean(im) for im in cube ])
            # Fill in missing tails with last real value
            if len(light_curve) < uv_cad_n: light_curve = extend(light_curve, uv_cad_n)
            if len(light_curve) > uv_cad_n: light_curve = truncate(light_curve, uv_cad_n)
            # Linearly interpolate onto high cadnce grid set by the euv channels
            light_curve = interp_to_high_cad(t_uv_grid, light_curve, t_grid_high_cad)
            # Append all SDO light curves and turn into a numpy array
            mvts_light_curves.append(light_curve)

    #? hmi magnetic field channel (45s)
    band = hmi_data_m.get('M_45s', None)
    # Make a directory for saved filter images
    band_save_dir = f'{save_dir_ar_obs}/{band}/'
    out_dir = os.path.join(band_save_dir)
    if not os.path.exists(out_dir): os.makedirs(out_dir)
    # Construct the drms query string: either "series[timespan][wavelength]"
    qstr = 'hmi.M_45s['+t_reform1+'/264h@6m][2]{image}'
    # Download a seiries of images given the bounding boxes
    process = {'im_patch': {'t_ref': t_reform2,
                            't': 0,
                            'r': 0,
                            'c': 0,
                            'locunits': 'arcsec',
                            'boxunits': 'arcsec',
                            'x': x_cent,
                            'y': y_cent,
                            'width': box_width,
                            'height': box_height}}
    result = client.export(qstr, method='url',protocol='fits',email='brandon.panos@fhnw.ch', process=process)
    result.download(out_dir)

    # Convert all images to numpy arrays and save
    files = os.listdir(band_save_dir)
    files.sort() # Time order fits files
    paths = [ band_save_dir + '/' + file for file in files if '.fits' in file ]
    cube = [] # Create cube of images
    for path in paths:
        # get header information
        hdul = fits.open(path)
        image = fits.getdata(path, ext=1)
        cube.append(image)
    cube = np.array(cube)
    ndarray_save_path = f'{save_ndarray_folder}/{band}'
    np.save(ndarray_save_path, cube) # saves array (time, x, y)

    # Create a video of the datacube by calling the animation function
    save_vid_path = f'{save_vids_folder}/{band}.mp4'
    ar = f'{NOAA_AR}_{flare_cls}_{max_loc}'
    animate(cube, paths, name=ar, segment=band, save_path=save_vid_path, figsize=(8,8), use_log=0)

    # Turn images into light curves
    light_curve = np.array([ np.nanmean(im) for im in cube ])
    # Fill in missing tails with last real value
    if len(light_curve) < mag_cad_n: light_curve = extend(light_curve, mag_cad_n)
    if len(light_curve) > mag_cad_n: light_curve = truncate(light_curve, mag_cad_n)
    # Linearly interpolate onto high cadnce grid set by the euv channels
    light_curve = interp_to_high_cad(t_mag_grid, light_curve, t_grid_high_cad)
    # Append all SDO light curves and turn into a numpy array
    mvts_light_curves.append(light_curve)


    #? hmi bitmap channel (45s)
    # Need to get the bitmap for the dopplergram channel to subtract the velocity from the solar rotation by removing the mean of the quiet sun
    band = 'bit'
    # Make a directory for saved filter images
    band_save_dir = f'{save_dir_ar_obs}/{band}/'
    out_dir = os.path.join(band_save_dir)
    if not os.path.exists(out_dir): os.makedirs(out_dir)
    # Construct the drms query string: either "series[timespan][wavelength]"
    qstr = f'hmi.sharp_cea_720s[{NOAA_AR}]['+t_reform1+'/264h@6m]{bitmap}'
    result = client.export(qstr, method='url',protocol='fits',email='brandon.panos@fhnw.ch')
    result.download(out_dir)

    # Convert all images to numpy arrays and save
    files = os.listdir(band_save_dir)
    files.sort() # Time order fits files
    paths = [ band_save_dir + '/' + file for file in files if '.fits' in file ]
    cube = [] # Create cube of images
    for path in paths:
        # get header infomation
        hdul = fits.open(path)
        image = fits.getdata(path, ext=1)
        cube.append(image)
    cube = np.array(cube)
    ndarray_save_path = f'{save_ndarray_folder}/{band}'
    np.save(ndarray_save_path, cube) # saves array (time, x, y)

    # Create a video of the datacube by calling the animation function
    save_vid_path = f'{save_vids_folder}/{band}.mp4'
    ar = f'{NOAA_AR}_{flare_cls}_{max_loc}'
    animate(cube, paths, name=ar, segment=band, save_path=save_vid_path, figsize=(8,8), use_log=0)

    dir_binary_fits_files = out_dir # Keep track of file with bitmapp header information

    #? hmi dopplergram channel (45s)
    hmi_data_v = {'V_45s': 'vel'}
    band = hmi_data_v.get('V_45s', None)
    # Make a directory for saved filter images
    band_save_dir = f'{save_dir_ar_obs}/{band}/'
    out_dir = os.path.join(band_save_dir)
    if not os.path.exists(out_dir): os.makedirs(out_dir)
    # Construct the drms query string: either "series[timespan][wavelength]"
    qstr = 'hmi.V_45s['+t_reform1+'/264h@6m][2]{image}'
    # Download a seiries of images given the bounding boxes
    process = {'im_patch': {'t_ref': t_reform2,
                            't': 0,
                            'r': 0,
                            'c': 0,
                            'locunits': 'arcsec',
                            'boxunits': 'arcsec',
                            'x': x_cent,
                            'y': y_cent,
                            'width': box_width,
                            'height': box_height}}
    result = client.export(qstr, method='url',protocol='fits',email='brandon.panos@fhnw.ch', process=process)
    result.download(out_dir)

    dir_velocity_fits_files = out_dir

    # velocity correction (ensure it works, will be only semi accurate, room for improvment)
    # Order the paths to the fits files by time
    all_bit_paths = sorted(os.listdir(dir_binary_fits_files))
    all_bit_paths = [dir_binary_fits_files + all_bit_paths[i] for i in range(len(all_bit_paths))]
    all_vel_paths = sorted(os.listdir(dir_velocity_fits_files))
    all_vel_paths = [dir_velocity_fits_files + all_vel_paths[i] for i in range(len(all_vel_paths))]

    # collect all times for each bitmap
    time_format = '%Y.%m.%d_%H:%M:%S.%f_TAI'
    bit_time_list = []
    for path_to_bit in all_bit_paths:
        with fits.open(path_to_bit) as hdul:
            bit_header = hdul[0].header
            t_obs_bit = bit_header['T_OBS']
            bit_time_list.append(t_obs_bit)
    # Convert time strings to datetime objects - Moved outside the loop
    bit_time_list = [datetime.strptime(time_str, time_format) for time_str in bit_time_list]
                    
    cube = [] # Create cube of images
    for path_to_vel in all_vel_paths:
        with fits.open(path_to_vel) as hdul:
            vel_im = hdul[1].data  # Accessing the image is in the primary HDU

            vel_header = hdul[1].header
            t_obs_vel = vel_header['T_OBS']

            # Convert time strings to datetime objects
            target_datetime = datetime.strptime(t_obs_vel, time_format)

            # Find the index of the time closest to the target time
            closest_index = min(range(len(bit_time_list)), key=lambda i: abs(bit_time_list[i] - target_datetime))
            print(bit_time_list[closest_index])
            print(target_datetime)

            # Get the closest time bit image
            with fits.open(all_bit_paths[closest_index]) as hdul:
                bit_im = hdul[0].data
                # resize bit image to match velocity image
                bit_im = np.resize(bit_im, (vel_im.shape[0], vel_im.shape[1]))

                # Correct velocity image by suptracting mean of the quit sun isolated by the closest bitmap image
                corrected_vel_im = use_bitmap(vel_im, bit_im)
                cube.append(corrected_vel_im)
                
    cube = np.array(cube)
    # velocity correction end

    ndarray_save_path = f'{save_ndarray_folder}/{band}'
    np.save(ndarray_save_path, cube) # saves array (time, x, y)

    # Create a video of the datacube by calling the animation function
    save_vid_path = f'{save_vids_folder}/{band}.mp4'
    ar = f'{NOAA_AR}_{flare_cls}_{max_loc}'
    animate(cube, paths, name=ar, segment=band, save_path=save_vid_path, figsize=(8,8), use_log=0)

    # Turn images into light curves
    light_curve = np.array([ np.nanmean(im) for im in cube ])
    # Fill in missing tails with last real value
    if len(light_curve) < mag_cad_n: light_curve = extend(light_curve, mag_cad_n)
    if len(light_curve) > mag_cad_n: light_curve = truncate(light_curve, mag_cad_n)
    # Linearly interpolate onto high cadnce grid set by the euv channels
    light_curve = interp_to_high_cad(t_mag_grid, light_curve, t_grid_high_cad)
    # Append all SDO light curves and turn into a numpy array
    mvts_light_curves.append(light_curve)

    #? Sharps channel (720s)
    df_mvt = df_mvts[int(loc) : int(max_loc)] # slice the SHARP dataframe to the three hour interval
    # 1) interpolate missing values
    df_mvt = interpolate_missing_vals(df_mvt)
    # 2) skip mvt if B data poor for more than 10 consecutive datapoints (i.e., 2hrs) or a total of 4hrs
    var = df_mvt['IS_TMFI']
    s = var != 1
    number_of_consec_bad_b = (~s).cumsum()[s].value_counts().max()
    tot_bad_b = var[var < 1].count()
    # 3) skip mvt if X-ray quality is poor (<2) for more than 10 consecutive points (i.e., 2hrs)
    var = df_mvt['XR_QUAL']
    s = var < 2
    number_of_consec_bad_X = (~s).cumsum()[s].value_counts().max()
    # Remove unwanted series
    to_drop = ['Timestamp',
            'LAT_MIN','LON_MIN','LAT_MAX','LON_MAX','HC_ANGLE',
            'BFLARE','BFLARE_LABEL','BFLARE_LOC','BFLARE_LABEL_LOC',
            'CFLARE','CFLARE_LABEL','CFLARE_LOC','CFLARE_LABEL_LOC',
            'MFLARE','MFLARE_LABEL','MFLARE_LOC','MFLARE_LABEL_LOC',
            'XFLARE','XFLARE_LABEL','XFLARE_LOC','XFLARE_LABEL_LOC',
            'QUALITY','XR_QUAL','CRVAL1','CRVAL2','CRLN_OBS','CRLT_OBS','SPEI','IS_TMFI']
    df_mvt.drop(to_drop, axis=1, inplace=True)
    array = df_mvt.to_numpy().T
    # Save sharps 
    np.save(f'{save_ndarray_folder}/sharps', array) # ndarray (feature, time) --> (25 mag, 3hrs)

    # Interpolate onto high cadence grid and join to SDO lightcurves for single flare
    mag_fet_high_res = []
    for mag_fet in array:
        # Fill in missing tails with last real value
        if len(mag_fet) < sharp_cad_n: mag_fet = extend(mag_fet, sharp_cad_n)
        if len(mag_fet) > sharp_cad_n: mag_fet = truncate(mag_fet, sharp_cad_n)
        # Linearly interpolate onto high cadnce grid set by the euv channels
        mag_fet = interp_to_high_cad(t_sharp_grid, mag_fet, t_grid_high_cad)
        mag_fet_high_res.append(mag_fet)
    mag_fet_high_res = np.vstack(mag_fet_high_res) # (25 magnetic features, time)

    # concatenate SDO and SHARP high res light curves into single array 
    final_obs_mvt = np.concatenate( (mvts_light_curves, mag_fet_high_res), axis=0 )
    # Save obs data
    np.save(f'../downloads_lowcad/my_mvts/{NOAA_AR}_{flare_cls}_{max_loc}', final_obs_mvt)