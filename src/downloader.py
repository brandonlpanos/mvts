import drms
import h5py
import torch
import requests
import numpy as np
from tqdm import tqdm
from astropy.io import fits
from pandas import Series as s
from datetime import datetime as dt_obj
from concurrent.futures import ThreadPoolExecutor

class data_aq():

    """
    Download data from JSOC database. 
    For details see https://github.com/mbobra/SHARPs and links therein.
    """

    def __init__(self, series='hmi.sharp_cea_720s', harp_num=377, time='2011.02.14_15:00:00/12h', filter=None, keywords='T_OBS, USFLUX', segment='Br, continuum'):

        """
        Example:
        #? keys, segments = c.query('hmi.sharp_cea_720s[377][2011.02.14_15:00:00/12h][? (QUALITY<65536) ?]', key='T_REC, USFLUX, ERRVF', segment='Br, continuum')

        Arguments:
        series --> e.g. hmi.sharp_cea_720s, aia.lev1_euv_12s
        harp_num: e.g. 720, 721
        time: e.g. 2011.02.14_15:00:00/12h, collect data from 15:00:00 to 03:00:00 the next day, or 2011.02.14_15:00:00/10h@6m, collect data every 6 minutes for 10 hours, or 2017.09.03_00:00_TAI-2017.09.06_00:00_TAI@6h from this time to that time at a 6 hour cadence.
        filter: e.g. [? (CRLN_OBS < 1) AND (USFLUX > 4e22) ?]
        keywords: e.g. T_OBS, HARPNUM, R_VALUE
        segment: e.g. magnetogram, continuum
        """

        self.series = series
        self.harp_num = harp_num
        self.time = time
        self.filter = filter
        self.keywords = keywords
        self.segment = segment

        self.c = drms.Client()

        if filter != None: name = f"{series}[{harp_num}][{self.time}][{filter}]"
        else: name = f"{series}[{harp_num}][{self.time}][]"
        self.named_seg = [f"{k.strip()}" for k in segment.split(",")]
        self.keys, _ = self.c.query(f"{name}", key=f"{keywords}", segment=f"{segment}")

    @staticmethod
    def parse_tai_string(tstr,datetime=True):
        """function to convert T_OBS into a datetime object"""
        year   = int(tstr[:4])
        month  = int(tstr[5:7])
        day    = int(tstr[8:10])
        hour   = int(tstr[11:13])
        minute = int(tstr[14:16])
        if datetime: return dt_obj(year,month,day,hour,minute)
        else: return year,month,day,hour,minute
         
    def download(self, filename="Brandon"):
        """Download data from JSOC database and save it to a H5PY file"""
        data = h5py.File(f"/Users/brandonlpanos/gits/mvts/data/new_data/{filename}", "w") # create data file

        HARPS = [f"H_{i}" for i in self.harp_num]

        progress_bar = tqdm(total=len(self.harp_num), desc="Overall Progress", position=0)
        with requests.Session() as session:
            for h, N in zip(HARPS, self.harp_num):
                HARP_group = data.create_group(f"{h}") # create group for saving everything with certain harp number

                if self.filter is not None:
                    name = f"{self.series}[{N}][{self.time}][{self.filter}]"
                else:
                    name = f"{self.series}[{N}][{self.time}]"

                keys_H, segments_H = self.c.query(name, key=self.keywords, segment=self.segment) # Download Data stepwise for each HARP

                t_obs = HARP_group.create_dataset("t_obs", shape=(keys_H.T_OBS.size,), dtype="S23", chunks=True, maxshape=(None,))
                datasets = {dataset_name: HARP_group.create_dataset(dataset_name, shape=(keys_H.T_OBS.size,) + fits.open('http://jsoc.stanford.edu' + getattr(segments_H, dataset_name)[0])[-1].data.shape, dtype="f", chunks=True) for dataset_name in self.named_seg}

                def download_dataset(dataset_name, i):
                    url = 'http://jsoc.stanford.edu' + getattr(segments_H, dataset_name)[i]
                    dataset = datasets[dataset_name]
                    dataset[i] = torch.from_numpy(fits.open(url)[-1].data)

                def download_t_obs(i):
                    t_obs[i] = self.parse_tai_string(keys_H.T_OBS[i], datetime=True).isoformat()

                with ThreadPoolExecutor() as executor:
                    for i in range(keys_H.T_OBS.size):
                        executor.submit(download_t_obs, i)
                        for dataset_name in self.named_seg:
                            executor.submit(download_dataset, dataset_name, i)

                progress_bar.update(1)

        progress_bar.close()
        data.close()
    
    def get_data(self, filename = "Data.h5"):
        h5_file = h5py.File(f"/Users/brandonlpanos/gits/mvts/data/new_data/{filename}", 'r') # open H5PY file

        group_names = list(h5_file.keys())

        datasets = {}

        for group_name in group_names:
            group = h5_file[group_name] # Access the group
            datasets[group_name] = {}

            datasets[group_name]["t_obs"] = group["t_obs"][:]

            for dataset_name in self.named_seg:
                dataset = group[dataset_name][:]
                datasets[group_name][dataset_name] = torch.from_numpy(dataset)
        return datasets

