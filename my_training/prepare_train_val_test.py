import random
from sklearn.model_selection import train_test_split
import h5py
import numpy as np
import matplotlib.pyplot as plt
from data_manipulation.dataset import Dataset
from PIL import Image
import os
from glob import glob

data_out_dir = "/infodev1/non-phi-data/junjiang/Path_GAN/TSR_data"
# data_out_dir = "/infodev1/non-phi-data/junjiang/Path_GAN/HGSOC_SBOT"
# case_list = ["Wo-1-A5_RIO1338_HE", "Wo-1-C4_RIO1338_HE", "Wo-1-F1_RIO1338_HE",
#                  "Wo-2-B4_RIO1338_HE", "Wo-2-F1_RIO1338_HE"]

img_dir = "/infodev1/non-phi-data/junjiang/OvaryCancer/PathGAN-224_224"

# case_list_tmp = os.listdir(img_dir)
# case_list = []
# for i in case_list_tmp:
#     if ".hd5" in i:
#         pass
#     else:
#         case_list.append(i)

case_list = ["OCMC-{:03d}".format(i) for i in range(1, 31)]

# dataset_name = "TSR"
dataset_name = "HGSOC_SBOT"
marker = "he"

hdf5_path_train = os.path.join(data_out_dir, "hdf5_%s_%s_train.h5" % (dataset_name, marker))
hdf5_file_train = h5py.File(hdf5_path_train, mode='w')

hdf5_path_val = os.path.join(data_out_dir, "hdf5_%s_%s_validation.h5" % (dataset_name, marker))
hdf5_file_val = h5py.File(hdf5_path_val, mode='w')

hdf5_path_test = os.path.join(data_out_dir, "hdf5_%s_%s_test.h5" % (dataset_name, marker))
hdf5_file_test = h5py.File(hdf5_path_test, mode='w')

# hdf5_path_all = os.path.join(data_out_dir, "hdf5_%s_all.h5" % dataset_name)
# hdf5_file_all = h5py.File(hdf5_path_all, mode='w')

#
# key_shape = [eligible_cnt_in_this_case, 224, 224, 3]
# img_storage = hdf5_file_w.create_dataset(name='image', shape=key_shape, dtype=np.float32)
# label_storage = hdf5_file_w.create_dataset(name='tsr_scores', shape=[eligible_cnt_in_this_case, 3], dtype=np.uint8)
#
# for idx, img in enumerate(img_data_list):
#     img_storage[idx] = img.astype(np.float32)/255.0
#     label_storage[idx] = labels_data_list[idx]
#
# hdf5_file_w.close()


case_hd5_dir = "/infodev1/non-phi-data/junjiang/OvaryCancer/PathGAN-224_224"
train_cnt = 0
val_cnt = 0
test_cnt = 0
for c in case_list:
    hdf5_fn = os.path.join(case_hd5_dir, c + ".hd5")
    hdf5_file = h5py.File(hdf5_fn, 'r')
    print('Keys:', hdf5_file.keys())
    # Keys: <KeysViewHDF5 ['test_img', 'test_labels']>
    orig_imgs = hdf5_file["image"]
    total_cnt = len(orig_imgs)
    labels = np.ones((total_cnt, 1)).astype(np.uint8)

    split_0_ = int(len(orig_imgs) * 0.6)
    split_1_ = int(len(orig_imgs) * 0.8)
    train_cnt += split_0_
    val_cnt += split_1_ - split_0_
    test_cnt += total_cnt - split_1_

    # train_orig_imgs = orig_imgs[0:split_0_,:,:,:]
    # val_orig_imgs = orig_imgs[split_0_:split_1_, :, :, :]
    # test_orig_imgs = orig_imgs[split_1_:, :, :, :]
    # train_labels = labels[0:split_0_, :, :, :]
    # val_labels = labels[split_0_:split_1_, :, :, :]
    # test_labels = labels[split_1_:, :, :, :]

img_storage_train = hdf5_file_train.create_dataset(name='train_img', shape=[train_cnt, 224, 224, 3], dtype=int)
label_storage_train = hdf5_file_train.create_dataset(name='train_labels', shape=[train_cnt, 1], dtype=np.uint8)
img_storage_val = hdf5_file_val.create_dataset(name='validate_img', shape=[val_cnt, 224, 224, 3], dtype=int)
label_storage_val = hdf5_file_val.create_dataset(name='validate_labels', shape=[val_cnt, 1], dtype=np.uint8)
img_storage_test = hdf5_file_test.create_dataset(name='test_img', shape=[test_cnt, 224, 224, 3], dtype=int)
label_storage_test = hdf5_file_test.create_dataset(name='test_labels', shape=[test_cnt, 1], dtype=np.uint8)
train_pt = 0
val_pt = 0
test_pt = 0
for c in case_list:
    hdf5_fn = os.path.join(case_hd5_dir, c + ".hd5")
    hdf5_file = h5py.File(hdf5_fn, 'r')
    print('Keys:', hdf5_file.keys())
    # Keys: <KeysViewHDF5 ['test_img', 'test_labels']>
    orig_imgs = np.array(hdf5_file["image"])
    total_cnt = len(orig_imgs)
    labels = np.ones((total_cnt, 1)).astype(np.uint8)

    X_train, X_tmp, y_train, y_tmp = train_test_split(orig_imgs, labels, test_size=0.4, shuffle=True)
    X_val, X_test, y_val, y_test = train_test_split(X_tmp, y_tmp, test_size=0.5, shuffle=True)
    img_storage_train[train_pt:train_pt + len(X_train)] = X_train
    label_storage_train[train_pt:train_pt + len(y_train)] = y_train
    img_storage_val[val_pt: val_pt + len(X_val)] = X_val
    label_storage_val[val_pt: val_pt + len(y_val)] = y_val
    img_storage_test[test_pt: test_pt + len(X_test)] = X_test
    label_storage_test[test_pt: test_pt + len(y_test)] = y_test

    train_pt += len(X_train)
    val_pt += len(X_val)
    test_pt += len(X_test)
    #
    # total_cnt = len(orig_imgs)
    # split_0_ = int(len(orig_imgs) * 0.6)
    # split_1_ = int(len(orig_imgs) * 0.8)
    #
    #
    # print(total_cnt, split_0_, split_1_)
    #
    # img_storage_train[train_pt:train_pt+split_0_] = orig_imgs[0:split_0_, :, :, :]
    # label_storage_train[train_pt:train_pt+split_0_] = labels[0:split_0_, :]
    # img_storage_val[val_pt: val_pt + split_1_ - split_0_] = orig_imgs[split_0_:split_1_, :, :, :]
    # label_storage_val[val_pt: val_pt + split_1_ - split_0_] = labels[split_0_:split_1_, :]
    # img_storage_test[test_pt: test_pt + total_cnt - split_1_] = orig_imgs[split_1_:, :, :, :]
    # label_storage_test[test_pt: test_pt + total_cnt - split_1_] = labels[split_1_:, :]

    # print(train_pt, train_pt + split_0_)
    # print(val_pt, val_pt + split_1_ - split_0_)
    # print(test_pt, test_pt + total_cnt - split_1_)
    #
    #
    # train_pt += split_0_
    # val_pt += split_1_ - split_0_
    # test_pt += total_cnt - split_1_

    print(train_pt, val_pt, test_pt)

hdf5_file_train.close()
hdf5_file_val.close()
hdf5_file_test.close()
# hdf5_file_all.close()
