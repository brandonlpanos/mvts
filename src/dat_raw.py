import h5py
import requests
import drms
from tqdm import tqdm
import torch
from astropy.io import fits
from datetime import datetime as dt_obj
from pandas import Series as s
from concurrent.futures import ThreadPoolExecutor

import numpy as np

class data_aq():
    def __init__(self, series, keywords, seg, Harp_NUM = None, time = None, filter = None):
        super().__init__()
        self.time = time
        self.series = series
        self.keywords = keywords
        self.seg = seg
        self.filter = filter

        y = 0
        n = 0

        self.Dat_len = 0

        self.c = drms.Client()

        if time == None:
            self.time = f"0-3000"

        if type(Harp_NUM) == int or Harp_NUM == None:
            if filter != None and Harp_NUM != None:
                name = f"{series}[{Harp_NUM}][{self.time}][{filter}]"
                self.Harp_NUM = [Harp_NUM]

            elif filter != None and Harp_NUM == None:
                name = f"{series}[][{self.time}][{filter}]"
                self.Harp_NUM = None

            elif filter == None and Harp_NUM != None:
                name = f"{series}[{Harp_NUM}][{self.time}][]"
                self.Harp_NUM = [Harp_NUM]

            else:
                assert "More Information is need, e.g. HARP-Numbers, or filter!"

            self.named_seg = [f"{k.strip()}" for k in seg.split(",")]

            self.keys, _ = self.c.query(f"{name}", key=f"{keywords}", seg=f"{seg}")

            if hasattr(self.keys, "T_OBS"):
                print(f"Download data in range {min(self.keys.T_OBS)}-{max(self.keys.T_OBS)}")
                self.Dat_len += self.keys.T_OBS.size

            else:
                print("Attribute 'T_OBS' does not exist in 'keys'")
                
        
        else:
            self.keys = []
            No_T_Obs = []
            HARPS = np.arange(Harp_NUM[0],Harp_NUM[-1]+1,1)
            self.Harp_NUM = []

            self.named_seg = [f"{k.strip()}" for k in seg.split(",")]
            
            for N in tqdm(HARPS):
                if filter == None:
                    name = f"{series}[{N}][{self.time}][]"
                else:
                    name = f"{series}[{N}][{self.time}][{filter}]"

                Keys, _ = self.c.query(f"{name}", key=f"{keywords}", seg=f"{seg}")

                if hasattr(Keys, 'T_OBS'):
                    # print(f"Download HARP_NUM {N} in range {min(Keys.T_OBS)}-{max(Keys.T_OBS)}")
                    self.Dat_len += Keys.T_OBS.size
                    self.Harp_NUM.append(N)
                    self.keys.append(Keys)
                    y +=1
                else:
                    # print(f"Attribute .T_OBS does not exist in 'keys' for HARP_NUM {N}")
                    n +=1
                    No_T_Obs.append(N)
            print(f"{y} HARPS to download")
            print(f"{n} HARPS to dismiss")

            with open("No_T_OBS.txt", 'w') as file:
                for item in No_T_Obs:
                    file.write(str(item) + '\n')
        
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

    def overview(self, N = None):
        t0 = (16*60+13.9)/2622 # t0 in seconds for 2622 fit files
        m0 = 15/2622 # memory in GB for 2622 fit files
        print(f"total number of fit files {self.Dat_len}")
        print(f"expected memory consumption: {round(self.Dat_len*m0,1)} GB")
        print(f"expected time to download: {round(self.Dat_len*t0/3600,1)} h")

        # si = self.c.info(self.series) # set the series
        # display(si.keywords)
        
        # if N != None and hasattr(data_aq, 'HARP_NUM') == True: # and type(self.Harp_NUM) == list
        #     if N in self.Harp_NUM:
        #         print(f"fit files for HARP_NUM {N}")
        #         display(self.keys[self.Harp_NUM.index(N)])
        #     else:
        #         print(f"HARP_NUM {N} is not in the queried data")

        # elif len(self.keys) >1 and N == None: 
        #     print("Don't diplay keys")
            
        # else:
        #     display(self.keys)  
            
         
    def download(self, filename="Brandon"):
        data = h5py.File(f"/sml/witmerj/Sunspot_Data/{filename}", "w") # create data file

        if self.Harp_NUM is None:
            self.Harp_NUM = s.unique(self.keys.HARPNUM)

        HARPS = [f"H_{i}" for i in self.Harp_NUM]

        progress_bar = tqdm(total=len(self.Harp_NUM), desc="Overall Progress", position=0)
        with requests.Session() as session:
            for h, N in zip(HARPS, self.Harp_NUM):
                HARP_group = data.create_group(f"{h}") # create group for saving everything with certain harp number

                if self.filter is not None:
                    name = f"{self.series}[{N}][{self.time}][{self.filter}]"
                else:
                    name = f"{self.series}[{N}][{self.time}]"

                keys_H, segments_H = self.c.query(name, key=self.keywords, seg=self.seg) # Download Data stepwise for each HARP

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
        h5_file = h5py.File(f"/sml/witmerj/Sunspot_Data/{filename}", 'r') # open H5PY file

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
    
if __name__ == "__main__":
    dat = data_aq('hmi.sharp_cea_720s', "T_OBS, HARPNUM, NOAA_AR, LONDTMIN, LONDTMAX", "continuum, magnetogram, bitmap, Dopplergram, Bp, Bt, Br, conf_disambig", Harp_NUM = 8923, filter = None )
    dat.overview()
    dat.download("Brandon_8923.h5")