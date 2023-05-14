import h5py
import numpy as np
import matplotlib.pyplot as plt
from data_manipulation.dataset import Dataset
from PIL import Image
import os

data_dir = "/infodev1/non-phi-data/junjiang/Path_GAN/TSR_data/training"

"/infodev1/non-phi-data/junjiang/Path_GAN/TSR_data/training"
case_list = ["Wo-1-A5_RIO1338_HE", "Wo-1-C4_RIO1338_HE", "Wo-1-F1_RIO1338_HE",
             "Wo-2-B4_RIO1338_HE", "Wo-2-F1_RIO1338_HE"]

for c in case_list:
    hdf5_fn = os.path.join(data_dir, c + ".hd5")
    hdf5_file = h5py.File(hdf5_fn, 'r')
    print('Keys:', hdf5_file.keys())
    # Keys: <KeysViewHDF5 ['test_img', 'test_labels']>
    orig_imgs = hdf5_file["image"]
    lables = hdf5_file["tsr_scores"]
