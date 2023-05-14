import h5py
import numpy as np
from glob import glob
import os
from PIL import Image

dataset_name = "vgh_nki"  # default
marker = "he"  # default

real_hdf5 = "/infodev1/non-phi-data/junjiang/OvaryCancer/PathGAN-224_224/5045114_CR95-9505_B4_11-14-1995_HE.hd5"
hdf5_file = h5py.File(real_hdf5, 'r')
print('Keys:', hdf5_file.keys())
# Keys: <KeysViewHDF5 ['test_img', 'test_labels']>
orig_imgs = hdf5_file["image"]

data_out_dir = "/infodev1/non-phi-data/junjiang/Path_GAN/example_data/vgh_nki/he/patches_h224_w224"
hdf5_path_train = os.path.join(data_out_dir, "hdf5_%s_%s_test_my.h5" % (dataset_name, marker))
hdf5_file_train = h5py.File(hdf5_path_train, mode='w')

s_cnt = len(orig_imgs)
img_storage_train = hdf5_file_train.create_dataset(name='test_img', shape=[s_cnt, 224, 224, 3], dtype=int)
label_storage_train = hdf5_file_train.create_dataset(name='test_labels', shape=[s_cnt, 1], dtype=np.uint8)

img_storage_train[:, :, :, :] = (np.array(orig_imgs) * 255).astype(int)
label_storage_train[:, :] = np.ones([s_cnt, 1])

hdf5_file_train.close()
