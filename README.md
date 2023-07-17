## Transformer-based sentiment analysis for flare prediction

In this repo, we use a hybrid state-of-the-art transformer model to discover which structures in our multivariate time series (mvts) are important for flare prediction.
The following figure shows the data acquisition process. We use the header information in the SWAN-SF [(R. A. Angryk et al. 2020)](https://doi.org/10.7910/DVN/EBCFKM) benchmark dataset to derive the flare maximum time and active region bounding box information. We then step four hours back in time and collect multi-channel image data from the Joint Science Operations Center (JSOC). This image data includes magnetic field information from the Solar Dynamic Observatory’s [(SDO, Lemen et al. 2012)](https://ui.adsabs.harvard.edu/abs/2012SoPh..275...17L/abstract), Helioseismic and Magnetic Imager [(HMI, Scherrer et al. 2012)](https://ui.adsabs.harvard.edu/abs/2012SoPh..275..207S/abstract) instrument as well as full atmospheric coverage of multiple scale heights from the AIA instrument.   

<p align="center">
  <img width="500" src="data_reduction.png">
</p>

Each image is converted into a single scalar summary statistic and concatenated into a tensor which becomes populated with spatiotemporal information as one moves forward in time toward flare onset. The resultant matrix is assigned a label $1$ if it terminates with a flare, and a label $0$ if it does not. Additionally, we append to the dataset a set of physical parameters inferred by magnetograms called Space-weather HMI Active Region Patches [(SHARPs, M.G. Bobra et al. 2014)](https://ui.adsabs.harvard.edu/abs/2014SoPh..289.3549B/abstract). Since a few data products have different cadences, all data are interpolated to a 12-minute cadence.   

To examine the relative importance of channel, time order, and intensity structure within the data, we devise a set of data augmentations that gradually relax these structures and reduce the amount of information in the input. We then evaluate the model on each new augmentation and monitor its performance to deduce that particular structure’s importance. The set of augmentations can be seen in the following image:  

<p align="center">
  <img width="700" src="structure_decay.png">
</p>


Each subplot is a different augmentation. The shapes represent different feature channels (such as magnetic field strength etc.) while the size of each shape represents the intensity of the feature at a particular time. The maximum structure is in the top left, while the minimum structure is in the bottom right, where the model can only leverage the relative power spectrum of the input. 