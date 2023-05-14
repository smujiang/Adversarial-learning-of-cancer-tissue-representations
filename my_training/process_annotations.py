import os
import numpy as np
import io
import glob
import h5py
from skimage.color import rgb2hsv
from PIL import Image
from skimage.color import rgb2lab


def filter_by_content_area(rgb_image_array, area_threshold=0.5, brightness=85):
    """
    Takes an RGB image array as input,
        converts into LAB space
        checks whether the brightness value exceeds the threshold
        returns a boolean indicating whether the amount of tissue > minimum required

    :param rgb_image_array:
    :param area_threshold:
    :param brightness:
    :return:
    """
    # rgb_image_array[np.any(rgb_image_array == [0, 0, 0], axis=-1)] = [255, 255, 255]
    lab_img = rgb2lab(rgb_image_array)
    l_img = lab_img[:, :, 0]
    binary_img_array_1 = np.array(0 < l_img)
    binary_img_array_2 = np.array(l_img < brightness)
    binary_img = np.logical_and(binary_img_array_1, binary_img_array_2) * 255
    tissue_size = np.where(binary_img > 0)[0].size
    tissue_ratio = tissue_size * 3 / rgb_image_array.size  # 3 channels
    if tissue_ratio > area_threshold:
        return True
    else:
        return False


def get_mask_info(mask_img, background=(255, 255, 255)):
    """
    Takes an RGB mask image (PIL Image) as input,
        returns some information of the mask image
    :param mask_img:
    :param background: define the background color
    :return:
        n_lables:  how many labels in this mask image (data type: integer)
        colors: colors in this list
        areas: areas of each
    """
    cnt_colors = mask_img.getcolors()
    n_labels = 0
    colors = []
    color_areas_ratio = []
    for c in cnt_colors:
        if not background == c[1]:
            n_labels += 1
            colors.append(c[1])
            color_areas_ratio.append(c[0])
    color_areas_ratios = np.array(color_areas_ratio) / (mask_img.width * mask_img.height)
    return n_labels, colors, color_areas_ratios


def filter_by_mask_info(n_labels, colors, color_areas_ratios):
    if len(color_areas_ratios) == 0:  # empty
        return False, None
    if n_labels == 1:
        if color_areas_ratios > 0.5:
            return True, colors
        else:
            return False, None
    else:
        m = max(color_areas_ratios)  # select the one cover largest area as the label
        if m < 0.5:  # covers less than 50%
            return False, None
        else:
            idx = [i for i, j in enumerate(color_areas_ratios) if j == m]
            if len(idx) > 1:
                print("This patch share two labels or more ")
                #: TODO: this patch share two labels or more
                label_color = colors[idx[0]]
            else:
                label_color = colors[idx[0]]
            return True, label_color


'''
Sample a serials of patches from an image array
img_arr:    image array, type: numpy ndarray
patch_size: height and width, eg. [50,50]
step:       sampling step, eg. [50,50]
area_range: define the sampling start and end with where, [start_x_cor,end_x_cor,start_y_cor,end_y_cor],
            eg. [10,810,100,900]
# Reference: sklearn.feature_extraction.image.extract_patches_2d()            
'''


def slide_img_sampling(img_arr, patch_size, step, area_range=None):
    '''

    :param img_arr:
    :param patch_size:
    :param step:
    :param area_range:
    :return:
    '''
    img_size = img_arr.shape
    channels = img_size[2]
    if area_range is None:
        w_min = 0
        w_max = img_size[1]
        h_min = 0
        h_max = img_size[0]
    else:
        w_min = area_range[0]
        w_max = area_range[1]
        h_min = area_range[2]
        h_max = area_range[3]
    if w_min < 0 | h_min < 0 | w_max > img_size[1] | h_max > img_size[0]:
        print("area_range parameters error.")
        return False
    ex_w = range(w_min, w_max - patch_size[1] + step[1], step[1])
    ex_h = range(h_min, h_max - patch_size[0] + step[0], step[0])
    nd_array_patches = np.empty((len(ex_w) * len(ex_h), patch_size[0], patch_size[1], channels), dtype=np.uint8,
                                order='C')
    offsets = []
    patch_num = 0
    for h in ex_h:
        for w in ex_w:
            patch_arr = img_arr[h:(h + patch_size[0]), w:(w + patch_size[1]), :]
            if patch_arr.shape[0] == patch_size[0] and patch_arr.shape[1] == patch_size[1]:
                nd_array_patches[patch_num, :, :, :] = patch_arr
                offsets.append([w, h])
                patch_num += 1
    return nd_array_patches, offsets


def get_label_by_color(color_map, color):
    for i in range(color_map.shape[0]):
        for j in range(color_map.shape[1]):
            if all(color_map[i, j] == np.array(color)):
                return i, j
    return -1, -1


def write_to_csv(fp, img_fn_list, label_list):
    for idx, img_fn in enumerate(img_fn_list):
        labels = label_list[3 * idx:3 * (idx + 1)]
        Fibrosis_score = labels[0][1]
        Cellularity_score = labels[1][1]
        Orientation_score = labels[2][1]
        fp.write(
            str(Fibrosis_score) + "," + str(Cellularity_score) + "," + str(Orientation_score) + "," + img_fn + "\n")


# VALIDATION = True
SAVE_IMG = True
VALIDATION = False
# VALIDATION = True
# SAVE_IMG = False


count_all_samples = 0
count_eligible_samples = 0

all_encoded_label_list = []
eligible_encoded_label_list = []
eligible_img_fn_list = []

if __name__ == "__main__":
    patch_sz = [448, 448]
    rescale_to = [224, 224]
    stride = [112, 112]
    # stride = [32, 32]

    # data_root = "/Users/m192500/Dataset/OvaryCancer/StromaReactionAnnotation"
    # data_out_dir = "/Users/m192500/Dataset/OvaryCancer/StromaReactionAnnotation_pro"
    # data_val_out_dir = "/Users/m192500/Dataset/OvaryCancer/StromaReactionAnnotation_pro/val"

    data_root = "/infodev1/non-phi-data/junjiang/OvaryCancer/StromaReaction/Annotation"
    data_out_dir = "/infodev1/non-phi-data/junjiang/Path_GAN/TSR_data/training"
    data_val_out_dir = "/infodev1/non-phi-data/junjiang/Path_GAN/TSR_data/training"

    case_list = ["Wo-1-A5_RIO1338_HE", "Wo-1-C4_RIO1338_HE", "Wo-1-F1_RIO1338_HE",
                 "Wo-2-B4_RIO1338_HE", "Wo-2-F1_RIO1338_HE"]
    anno_list = ["Fibrosis", "Cellularity", "Orientation"]
    annotation_color_map = [[[100, 0, 0], [180, 0, 0], [255, 0, 0]],
                            [[0, 100, 0], [0, 180, 0], [0, 255, 0]],
                            [[0, 0, 100], [0, 0, 180], [0, 0, 255]]]

    for c in case_list:
        img_fn_list = glob.glob(os.path.join(data_root, c, "*.jpg"))

        img_num = len(img_fn_list)

        eligible_cnt_in_this_case = 0
        img_data_list = []
        labels_data_list = []

        for img_fn in img_fn_list:
            print("Processing %s" % img_fn)
            img_arr = np.array(Image.open(img_fn))
            mask_arr_list = []
            for _anno_ in anno_list:
                mask_fn = img_fn.replace(".jpg", "_" + _anno_ + "-mask.png")
                mask_arr = np.array(Image.open(mask_fn))
                mask_arr_list.append(mask_arr)

            img_arr_patches, offsets = slide_img_sampling(img_arr, patch_sz, stride, area_range=None)
            for patch, offset in zip(img_arr_patches, offsets):
                count_all_samples += 1
                # if count_all_samples > 100:  # For Debug
                #     break
                if filter_by_content_area(patch):  # Check if the image patch has enough (50%+) tissue or not
                    temp_fn = img_fn.replace(data_root, data_out_dir)
                    patch_img_fn = temp_fn.replace(".jpg", "_" + str(offset[0]) + "_" + str(offset[1]) + ".jpg")
                    patch_out_dir = os.path.split(patch_img_fn)[0]
                    if not os.path.exists(patch_out_dir):
                        os.makedirs(patch_out_dir)

                    ELIGIBLE_PATCH = []
                    mask_Img_list = []
                    mask_Img_fn_list = []
                    mask_encoded_label_list = []
                    for idx, _anno_ in enumerate(anno_list):
                        patch_mask_fn = patch_img_fn.replace(".jpg", "_" + _anno_ + "-mask.png")
                        patch_mask_arr = mask_arr_list[idx][offset[1]:(offset[1] + patch_sz[1]),
                                         offset[0]:(offset[0] + patch_sz[0]), :]
                        mask_Img = Image.fromarray(patch_mask_arr)
                        n_labels, colors, color_areas_ratios = get_mask_info(mask_Img, background=(255, 255, 255))
                        # filter by the characteristics of annotations
                        ELIGIBLE_PATCH_idx, label_color = filter_by_mask_info(n_labels, colors, color_areas_ratios)
                        if label_color is None:
                            encoded_label = (-1, -1)
                        else:
                            if isinstance(label_color, list):
                                encoded_label = get_label_by_color(np.array(annotation_color_map), label_color[0])
                            else:
                                encoded_label = get_label_by_color(np.array(annotation_color_map), label_color)
                        mask_encoded_label_list.append(encoded_label)

                        ELIGIBLE_PATCH.append(ELIGIBLE_PATCH_idx)
                        mask_Img_fn_list.append(patch_mask_fn)
                        mask_Img_list.append(mask_Img)

                    all_encoded_label_list += mask_encoded_label_list

                    patch_Img = Image.fromarray(patch)
                    # Only when all the annotation is eligible, the patch and mask will be saved.
                    if all(ELIGIBLE_PATCH):
                        count_eligible_samples += 1
                        eligible_encoded_label_list += mask_encoded_label_list
                        eligible_img_fn_list.append(patch_img_fn)

                        img_data = np.array(patch_Img.resize(rescale_to))
                        img_labels = np.array(mask_encoded_label_list)[:, 1]

                        eligible_cnt_in_this_case += 1
                        img_data_list.append(img_data)
                        labels_data_list.append(img_labels)

                    if VALIDATION:
                        temp_img = np.zeros([patch_sz[0], patch_sz[1] * (len(anno_list) + 1), 4], dtype=np.uint8)
                        temp_img[0:patch_sz[0], 0:patch_sz[1], 0:3] = patch
                        temp_img[0:patch_sz[0], 0:patch_sz[1], 3] = np.ones(patch_sz, dtype=np.uint8) * 255
                        for idx, _anno_ in enumerate(anno_list):
                            # patch_mask_fn = patch_img_fn.replace(".jpg", "_" + _anno_ + "-mask.png")
                            patch_mask_arr = mask_arr_list[idx][offset[1]:(offset[1] + patch_sz[1]),
                                             offset[0]:(offset[0] + patch_sz[0]), :]
                            temp_img[0: patch_sz[0], patch_sz[1] * (idx + 1):patch_sz[0] * (idx + 2),
                            0:3] = patch_mask_arr
                            temp_img[0: patch_sz[0], patch_sz[1] * (idx + 1):patch_sz[0] * (idx + 2), 3] = np.ones(
                                patch_sz, dtype=np.uint8) * 255
                        temp_patch_img_fn = patch_img_fn.replace(patch_out_dir, data_val_out_dir)
                        temp_patch_img_fn = temp_patch_img_fn.replace(".jpg", ".png")
                        temp_patch_img_dir = os.path.split(temp_patch_img_fn)[0]
                        if not os.path.exists(temp_patch_img_dir):
                            os.makedirs(temp_patch_img_dir)
                        Image.fromarray(temp_img).resize([rescale_to[1] * (len(anno_list) + 1), rescale_to[0]]).save(
                            temp_patch_img_fn)
            # break  # for debug
        # break  # for debug

        hdf5_path = os.path.join(data_out_dir, c + ".hd5")
        hdf5_file_w = h5py.File(hdf5_path, mode='w')

        key_shape = [eligible_cnt_in_this_case, 224, 224, 3]
        # img_storage = hdf5_file_w.create_dataset(name='image', shape=key_shape, dtype=np.float32)
        img_storage = hdf5_file_w.create_dataset(name='image', shape=key_shape, dtype=int)
        label_storage = hdf5_file_w.create_dataset(name='tsr_scores', shape=[eligible_cnt_in_this_case, 3],
                                                   dtype=np.uint8)

        for idx, img in enumerate(img_data_list):
            # img_storage[idx] = img.astype(np.float32)/255.0
            img_storage[idx] = img.astype(int)
            label_storage[idx] = labels_data_list[idx]

        hdf5_file_w.close()

'''
1. How many patches/annotations we can get, and how many are eligible(criteria?)
2. distributions of each class (count)

'''
print("There are %d patches in total, %d of which are eligible" % (count_all_samples, count_eligible_samples))


def get_distribution(score_list):
    Fib_scores = []
    Cell_scores = []
    Ori_scores = []
    for c in score_list:
        if c[0] == 0:
            Fib_scores.append(c[1])
        elif c[0] == 1:
            Cell_scores.append(c[1])
        elif c[0] == 2:
            Ori_scores.append(c[1])
        # else:
        #     raise Exception("unexpected score type")
    return Fib_scores, Cell_scores, Ori_scores


import matplotlib.pyplot as plt

all_Fib_scores, all_Cell_scores, all_Ori_scores = get_distribution(all_encoded_label_list)
fig, ax = plt.subplots(1, 3)
plt.suptitle("All Annotations")
ax[0].hist(all_Fib_scores, density=False, bins=3)
ax[0].set_ylabel('Frequency')
ax[0].set_xlabel('Fibrosis Score')

ax[1].hist(all_Cell_scores, density=False, bins=3)
ax[1].set_ylabel('Frequency')
ax[1].set_xlabel('Cellularity Score')

ax[2].hist(all_Ori_scores, density=False, bins=3)
ax[2].set_ylabel('Frequency')
ax[2].set_xlabel('Orientation Score')
plt.show()

Fib_scores, Cell_scores, Ori_scores = get_distribution(eligible_encoded_label_list)
fig, ax = plt.subplots(1, 3)
plt.suptitle("Eligible Annotations")
ax[0].hist(Fib_scores, density=False, bins=3)
ax[0].set_ylabel('Frequency')
ax[0].set_xlabel('Fibrosis Score')

ax[1].hist(Cell_scores, density=False, bins=3)
ax[1].set_ylabel('Frequency')
ax[1].set_xlabel('Cellularity Score')

ax[2].hist(Ori_scores, density=False, bins=3)
ax[2].set_ylabel('Frequency')
ax[2].set_xlabel('Orientation Score')
plt.show()
