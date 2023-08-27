import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.animation as manimation 
from random import randint
from datetime import datetime as dt_obj
import torch
import numpy as np

class visualize():
    def __init__(self, data: torch.float, time: str, bitmap: torch.float = None, vrange:list = None):
        super().__init__()
        self.time_str = time
        self.time = [self.parse_tai_string(t) for t in time]
        
        if bitmap is not None:
            self.data = self.use_bitmap(data,bitmap)
        else:
            self.data = data

        if vrange == None:
            self.vrange = [None, None]
        elif vrange == "max":
            self.vrange = [torch.min(data), torch.max(data)]
        else:
            self.vrange = vrange


    @staticmethod
    def parse_tai_string(tstr,datetime=True):
        """function to convert T_REC into a datetime object"""
        year   = int(tstr[:4])
        month  = int(tstr[5:7])
        day    = int(tstr[8:10])
        hour   = int(tstr[11:13])
        minute = int(tstr[14:16])
        if datetime: return dt_obj(year,month,day,hour,minute)
        else: return year,month,day,hour,minute
    
    @staticmethod
    def use_bitmap(data, bitmap):
        print("you're using Dopplergram")
        bitmap = torch.where(bitmap <= 16, 1, 0)
        conv = torch.mul(data,bitmap)
        for i in range(data.shape[0]):
            mask = conv[i,:,:] != 0 # mask all the zeros
            norm = torch.mean(conv[i, mask])
            data[i,:,:] = torch.subtract(data[i,:,:],norm)
        return data
        
    def image(self, timestamp: str = None):

        if timestamp != None:
            timestamp = self.parse_tai_string(timestamp)
            diff = []
            for x in (self.time):
                diff.append(abs(timestamp-x))
            index = np.argmin(diff)
        else:
            index = randint(0,len(self.time))
        
        data = self.data[index,:,:]

        fig, ax = plt.subplots(dpi = 200)
        im = ax.imshow(data, cmap='inferno', vmin = self.vrange[0], vmax = self.vrange[1], origin='lower')
        ax.set_title(f"{self.time[index]}")
        ax.xaxis.set_major_locator(MultipleLocator(200))
        ax.xaxis.set_minor_locator(MultipleLocator(100))
        ax.yaxis.set_major_locator(MultipleLocator(100))
        ax.yaxis.set_minor_locator(MultipleLocator(50))
        ax.tick_params(which='major', length=5,width=1)
        ax.tick_params(which='minor', length=3,width=1)
        # Create a divider for the existing axes instance
        divider = make_axes_locatable(ax)

        # Append a new axes to the right of the main axes with specified width
        cax = divider.append_axes("right", size="3%", pad=0.05)

        # Add a colorbar to the new axes
        cbar = plt.colorbar(im, cax=cax)
        plt.tight_layout()
        # ax.set_colorbar()
        plt.show()
        plt.close()

    def video(self):
        fig, ax = plt.subplots(dpi = 200)
        im = ax.imshow(self.data[0,:,:], cmap='inferno', vmin = self.vrange[0], vmax = self.vrange[1], origin='lower')
        ax.set_title(f"{self.time[0]} – {self.time[-1]}")
        ax.xaxis.set_major_locator(MultipleLocator(200))
        ax.xaxis.set_minor_locator(MultipleLocator(100))
        ax.yaxis.set_major_locator(MultipleLocator(100))
        ax.yaxis.set_minor_locator(MultipleLocator(50))
        ax.tick_params(which='major', length=5,width=1)
        ax.tick_params(which='minor', length=3,width=1)
        # Create a divider for the existing axes instance
        divider = make_axes_locatable(ax)
        # Append a new axes to the right of the main axes with specified width
        cax = divider.append_axes("right", size="3%", pad=0.05)

        # Add a colorbar to the new axes
        cbar = plt.colorbar(im, cax=cax)
        plt.tight_layout()
        # ax.set_colorbar()
        plt.close()

        ims = []
        for i in range(1,len(self.time)):
            ims.append([ax.imshow(self.data[i,:,:], cmap='inferno', vmin = self.vrange[0], vmax = self.vrange[1], origin='lower')])

        ani = manimation.ArtistAnimation(fig, ims, interval=len(self.data)/30, blit=True, repeat_delay=1000)
        return ani
