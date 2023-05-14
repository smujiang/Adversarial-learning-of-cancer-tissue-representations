import h5py
import numpy as np
import matplotlib.pyplot as plt
from data_manipulation.dataset import Dataset
from PIL import Image
import os

# real_hdf5 = "/infodev1/non-phi-data/junjiang/Path_GAN/output_test/results/PathologyGAN/vgh_nki/h224_w224_n3_zdim200/hdf5_vgh_nki_he_test.h5"
# img_dir = "/infodev1/non-phi-data/junjiang/Path_GAN/example_data/debug_img"
# real_hdf5 = "/infodev1/non-phi-data/junjiang/Path_GAN/example_data/vgh_nki/he/patches_h224_w224/hdf5_vgh_nki_he_test.h5"
# test = Dataset(real_hdf5, 224, 224, 3, batch_size=2, labels=None, empty=False)
# images = test.images
#
# for i in range(100):
#     img_arr = np.squeeze(images[i, :, :, :]).astype(np.uint8)
#     img = Image.fromarray(img_arr)
#     save_to = os.path.join(img_dir, str(i)+".jpg")
#     img.save(save_to)
#
#
# with h5py.File(real_hdf5, mode='r') as hdf5_file:
#     # test = Dataset(hdf5_file, 224, 224, 3, batch_size=2, labels=None, empty=False)
#
#     for key in hdf5_file.keys():
#         print('\t Key: %s' % key)
#         key_shape = hdf5_file[key].shape
#         num_samples = key_shape[0]

###################################################################
# read testing data file. h5 format.
###################################################################
# real_hdf5 = "/infodev1/non-phi-data/junjiang/Path_GAN/example_data/vgh_nki/he/patches_h224_w224/hdf5_vgh_nki_he_test.h5"
# real_hdf5 = "/infodev1/non-phi-data/junjiang/Path_GAN/example_data/vgh_nki/he/patches_h224_w224/hdf5_vgh_nki_he_train.h5"
# real_hdf5 = "/infodev1/non-phi-data/junjiang/OvaryCancer/PathGAN-224_224/5045114_CR95-9505_B4_11-14-1995_HE.hd5"
# real_hdf5 = "/infodev1/non-phi-data/junjiang/Path_GAN/TSR_data/training/Wo-1-A5_RIO1338_HE.hd5"
real_hdf5 = "/infodev1/non-phi-data/junjiang/Path_GAN/TSR_data/hdf5_HGSOC_SBOT_he_test.h5"

hdf5_file = h5py.File(real_hdf5, 'r')
print('Keys:', hdf5_file.keys())
# Keys: <KeysViewHDF5 ['test_img', 'test_labels']>
# orig_imgs = hdf5_file["test_img"]
orig_imgs = hdf5_file["image"]

###################################################################
# read output file. h5 format.
###################################################################
# encode_fn = "/infodev1/non-phi-data/junjiang/Path_GAN/output_test/results/h224_w224_n3_zdim200/Wo-1-A5_RIO1338_HE.hd5"
# encode_fn = "/infodev1/non-phi-data/junjiang/Path_GAN/output_test/results/h224_w224_n3_zdim200/example_data/5045114_CR95-9505_B4_11-14-1995_HE.hd5"
encode_fn = "/infodev1/non-phi-data/junjiang/Path_GAN/output_test/results/h224_w224_n3_zdim200/hdf5_vgh_nki_he_test.h5"

hdf5_file = h5py.File(encode_fn, 'r')
print('Keys:', hdf5_file.keys())
# Keys: <KeysViewHDF5 ['test_img_prime', 'test_img_w_latent', 'test_labels']>

latent = hdf5_file["test_img_w_latent"]  # latent vector/discriptor encoded from real images
imgs = hdf5_file["test_img_prime"]  # images generated from W latent space.
# labels = hdf5_file["tsr_scores"]   # test labels are from the original file

for i in range(100):
    print(latent[i])
    img_in = np.squeeze(orig_imgs[i, :, :, :])
    img_out = np.squeeze(imgs[i, :, :, :] * 255).astype(np.uint8)
    show_img = np.concatenate((img_in, img_out), axis=1)
    plt.imshow(show_img)
    plt.axis("off")
    plt.show()
