import h5py
import numpy as np
from glob import glob
import os
from sklearn.manifold import TSNE
from PIL import Image
import matplotlib.pyplot as plt

case_list = ["5045114_CR95-9505_B4_11-14-1995_HE.svs"]
img_dir = "/infodev1/non-phi-data/junjiang/OvaryCancer/PathGAN-224_224"

case_id = case_list[0][:-4]
img_fn_list = glob(os.path.join(img_dir, case_id, "*.png"))

# input
real_hdf5 = "/infodev1/non-phi-data/junjiang/OvaryCancer/PathGAN-224_224/5045114_CR95-9505_B4_11-14-1995_HE.hd5"
hdf5_file = h5py.File(real_hdf5, 'r')
print('Keys:', hdf5_file.keys())
orig_imgs = hdf5_file["image"]

# output
encode_fn = "/infodev1/non-phi-data/junjiang/Path_GAN/output_test/results/h224_w224_n3_zdim200/5045114_CR95-9505_B4_11-14-1995_HE.hd5"
hdf5_file = h5py.File(encode_fn, 'r')
print('Keys:', hdf5_file.keys())

latent = hdf5_file["image_w_latent"]  # latent vector/discriptor encoded from real images
# imgs = hdf5_file["image_prime"]  # images generated from W latent space.
#
# for i in range(100):
#     img = Image.open(img_fn_list[i], 'r')
#     img_arr = np.array(img)[:, :, 0:3].astype(np.uint8)
#     img_in = np.squeeze(orig_imgs[i, :, :, :]*255).astype(np.uint8)
#     img_out = np.squeeze(imgs[i, :, :, :]*255).astype(np.uint8)
#     show_img = np.concatenate((img_arr, img_in, img_out), axis=1)
#     plt.imshow(show_img)
#     plt.show()

latent_arr = np.asarray(latent, dtype='float64')
tsne = TSNE(n_components=2, init='random')
embeddedings = tsne.fit_transform(latent_arr)
plt.scatter(x=embeddedings[:, 0], y=embeddedings[:, 1], s=0.5)
plt.title("TSR image data T-SNE projection")
plt.show()

# position 1:
idx = embeddedings[:, 0] > 20 and embeddedings[:, 0] < 40
embeddedings[:, 1] > -58 and embeddedings[:, 1] < -42

# position 2:


print(latent.shape)
