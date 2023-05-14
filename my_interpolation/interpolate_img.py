import numpy as np

from models.tools import linear_interpolation
from models.generative.gans.PathologyGAN_Encoder import PathologyGAN_Encoder as PathologyGAN
from models.evaluation.features import *
from data_manipulation.data import Data
from PIL import Image

# input_img1_fn = "/infodev1/non-phi-data/junjiang/Path_GAN/TSR_data/interpolation/img_pairs/5045114_CR95-9505_B4_11-14-1995_HE_81728_41152.png"
# input_img2_fn = "/infodev1/non-phi-data/junjiang/Path_GAN/TSR_data/interpolation/img_pairs/5045114_CR95-9505_B4_11-14-1995_HE_83072_45184.png"

input_img1_fn = "/infodev1/non-phi-data/junjiang/Path_GAN/example_data/debug_img/75.jpg"
input_img2_fn = "/infodev1/non-phi-data/junjiang/Path_GAN/example_data/debug_img/0.jpg"

image_width = 224  # default
image_height = 224  # default
image_channels = 3  # default
dataset = "TSR"  # default
# dataset = "vgh_nki"  # default
marker = "he"  # default

# save images into hd5 file
data_out_dir = "/infodev1/non-phi-data/junjiang/Path_GAN/TSR_data/interpolation"
hdf5_file_path = os.path.join(data_out_dir, "datasets", dataset, marker, "patches_h244_w244",
                              "hdf5_%s_%s_test.h5" % (dataset, marker))
hdf5_fp = h5py.File(hdf5_file_path, mode='w')

s_cnt = 2
img_storage_test = hdf5_fp.create_dataset(name='test_img', shape=[s_cnt, image_width, image_height, image_channels],
                                          dtype=int)
label_storage_test = hdf5_fp.create_dataset(name='test_labels', shape=[s_cnt, 1], dtype=np.uint8)

orig_img = Image.open(input_img1_fn, 'r')
a = (np.array(orig_img)[:, :, 0:3]).astype(int)
orig_img = Image.open(input_img2_fn, 'r')
b = (np.array(orig_img)[:, :, 0:3]).astype(int)
img_storage_test[:, :, :, :] = np.stack([a, b], axis=0)
label_storage_test[:, :] = np.ones([s_cnt, 1])

hdf5_fp.close()

# checkpoint = "/infodev1/non-phi-data/junjiang/Path_GAN/TSR_data_model/data_model_output/PathologyGAN_Encoder/TSR/h224_w224_n3_zdim200/checkpoints/PathologyGAN_Encoder.ckt"
checkpoint = "/infodev1/non-phi-data/junjiang/Path_GAN/pretrained_model/vgh_nki/checkpoints/PathologyGAN_Encoder.ckt"
batch_size = 2  #
z_dim = 200
model = "PathologyGAN"  # default
real_hdf5 = hdf5_file_path
dbs_path = "/infodev1/non-phi-data/junjiang/Path_GAN/TSR_data/interpolation"  # default
main_path = "/infodev1/non-phi-data/junjiang/Path_GAN/TSR_data/interpolation"
num_clusters = 100  # default
clust_percent = 1.0  # default
features = True
save_img = True  # default

# Paths for runs and datasets.
if dbs_path is None:
    dbs_path = os.path.dirname(os.path.realpath(__file__))
if main_path is None:
    main_path = os.path.dirname(os.path.realpath(__file__))

# Hyperparameters for training.
learning_rate_g = 1e-4
learning_rate_d = 1e-4
learning_rate_e = 1e-4
regularizer_scale = 1e-4
beta_1 = 0.5
beta_2 = 0.9
style_mixing = 0.5

# Model Architecture param.
layers_map = {224: 5, 112: 4, 56: 3, 28: 2}
layers = layers_map[image_height]
alpha = 0.2
n_critic = 5
gp_coeff = .65
use_bn = False
init = 'orthogonal'
loss_type = 'relativistic gradient penalty'
noise_input_f = True
spectral = True
attention = 28

path = os.path.join(main_path, 'results')
res = 'h%s_w%s_n%s_zdim%s' % (image_height, image_width, image_channels, z_dim)
path = os.path.join(path, res)
name_file = real_hdf5.split('/')[-1]
encode_hdf5_path = os.path.join(path, name_file)
print(encode_hdf5_path)


def make_gif(images, fname, duration=2, true_image=False):
    import moviepy.editor as mpy

    def make_frame(t):
        try:
            x = images[int(len(images) / duration * t)]
        except:
            x = images[-1]

        if true_image:
            return x.astype(np.uint8)
        else:
            return ((x + 1) / 2 * 255).astype(np.uint8)

    clip = mpy.VideoClip(make_frame, duration=duration)
    clip.write_gif(fname, fps=len(images) / duration)


# Instantiate and project images.
with tf.Graph().as_default():
    data = Data(dataset=dataset, marker=marker, patch_h=image_height, patch_w=image_width,
                n_channels=image_channels,
                batch_size=batch_size, project_path=dbs_path, labels=None)
    # Instantiate Model.
    pathgan = PathologyGAN(data=data, z_dim=z_dim, layers=layers, use_bn=use_bn, alpha=alpha, beta_1=beta_1, init=init,
                           regularizer_scale=regularizer_scale,
                           style_mixing=style_mixing, attention=attention, spectral=spectral,
                           noise_input_f=noise_input_f, learning_rate_g=learning_rate_g,
                           learning_rate_d=learning_rate_d, learning_rate_e=learning_rate_e, beta_2=beta_2,
                           n_critic=n_critic, gp_coeff=gp_coeff,
                           loss_type=loss_type, model_name=model)
    if not os.path.exists(encode_hdf5_path):
        encode_hdf5_path, num_samples = real_encode_from_checkpoint(model=pathgan, data=data, data_out_path=main_path,
                                                                    checkpoint=checkpoint, real_hdf5=real_hdf5,
                                                                    batches=batch_size, save_img=save_img)

    hdf5_fp = h5py.File(encode_hdf5_path, mode='r')
    print('Keys:', hdf5_fp.keys())
    pro_imgs = hdf5_fp['test_img_prime']
    pro_img_embs = hdf5_fp['test_img_w_latent']

    plt.imshow(np.squeeze(pro_imgs[0, :, :, :]))
    plt.show()

    n_images = 11
    # data_out_path = "/infodev1/non-phi-data/junjiang/Path_GAN/TSR_data_model/data_model_output/PathologyGAN_Encoder/TSR/h224_w224_n3_zdim200"
    data_out_path = "/infodev1/non-phi-data/junjiang/Path_GAN/pretrained_model/vgh_nki"
    images, sequence = linear_interpolation(pathgan, n_images, data_out_path, pro_img_embs[0, :], pro_img_embs[1, :])
    int_img_dir = "/infodev1/non-phi-data/junjiang/Path_GAN/TSR_data/interpolation"
    for i in range(n_images):
        img_storage = images[i, :, :, :]
        img_fn = os.path.join(int_img_dir, "int_" + str(i) + ".jpg")
        Img = Image.fromarray((img_storage * 255).astype(np.uint8))
        Img.save(img_fn)
        w_storage = sequence[i, :]
        plt.imshow(img_storage)
        plt.show()
        print(w_storage)

    images_all = np.concatenate(
        (np.expand_dims(a, axis=0).astype(np.uint8), images, np.expand_dims(b, axis=0).astype(np.uint8)), axis=0)
    gif_fn = "/infodev1/non-phi-data/junjiang/Path_GAN/TSR_data/interpolation/interpolation.gif"
    make_gif(images_all, gif_fn, duration=2, true_image=False)

    # TODO: try to interpolate along the graph?

    print("debug")
