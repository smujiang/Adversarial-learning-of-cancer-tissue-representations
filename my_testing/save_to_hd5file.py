import h5py
import numpy as np
from glob import glob
import os
from PIL import Image
import pandas as pd
import random
import glob

train_case_number = 60
img_dir = "/infodev1/non-phi-data/junjiang/OvaryCancer/PathGAN-224_224"

# case_list = os.listdir(img_dir)

case_list = ["OCMC-{:03d}".format(i) for i in range(1, 31)]

for case_id in case_list:
    # case_id = case_list[0][:-4]
    img_fn_list = glob.glob(os.path.join(img_dir, case_id, "*.png"))
    img_num = len(img_fn_list)
    key_shape = [img_num, 224, 224, 3]
    hdf5_path = os.path.join(img_dir, case_id + ".hd5")

    hdf5_file_w = h5py.File(hdf5_path, mode='w')
    img_storage = hdf5_file_w.create_dataset(name='image', shape=key_shape, dtype=np.uint8)
    for idx, fn in enumerate(img_fn_list):
        img = Image.open(fn, 'r')
        img_arr = np.array(img)[:, :, 0:3].astype(np.uint8)
        img_storage[idx] = img_arr

        if idx % 1000 == 0:
            print('Processed', idx + 1, 'images')

    hdf5_file_w.close()

    print("img data saved")
