import os
from PIL import Image
import staintools
import shutil

# Read data
target = staintools.read_image(
    "/infodev1/non-phi-data/junjiang/OvaryCancer/PathGAN-224_224/OCMC-001/OCMC-001_59584_54336.png")
RESULTS_DIR = "/infodev1/non-phi-data/junjiang/OvaryCancer/PathGAN-224_224_norm"
img_data_dir = "/infodev1/non-phi-data/junjiang/OvaryCancer/PathGAN-224_224"

case_id = "OCMC-005"
x = 98688
y = 82112

# case_id = "OCMC-012"
# x = 112960
# y = 30528

# case_id = "OCMC-017"
# x = 59008
# y = 61056

# case_id = "OCMC-016"
# x = 97344
# y = 44032


p_s = 448

tl_x, tl_y, bl_x, bl_y = (x - p_s, y + p_s, x - p_s, y - p_s)
tr_x, tr_y, br_x, br_y = (x + p_s, y + p_s, x + p_s, y - p_s)
t_x, t_y, b_x, b_y = (x, y + p_s, x, y - p_s)
l_x, l_y, r_x, r_y = (x - p_s, y, x + p_s, y)
center_img_fn = case_id + "_" + str(x) + "_" + str(y) + ".png"
img_fn1 = case_id + "_" + str(tl_x) + "_" + str(tl_y) + ".png"
img_fn2 = case_id + "_" + str(bl_x) + "_" + str(bl_y) + ".png"
img_fn3 = case_id + "_" + str(tr_x) + "_" + str(tr_y) + ".png"
img_fn4 = case_id + "_" + str(br_x) + "_" + str(br_y) + ".png"
img_fn5 = case_id + "_" + str(t_x) + "_" + str(t_y) + ".png"
img_fn6 = case_id + "_" + str(b_x) + "_" + str(b_y) + ".png"
img_fn7 = case_id + "_" + str(l_x) + "_" + str(l_y) + ".png"
img_fn8 = case_id + "_" + str(r_x) + "_" + str(r_y) + ".png"

output_dir = os.path.join(RESULTS_DIR, case_id)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

img_fn_list = [center_img_fn, img_fn1, img_fn2, img_fn3, img_fn4, img_fn5, img_fn6, img_fn7, img_fn8]
# Standardize brightness (optional, can improve the tissue mask calculation)
target = staintools.LuminosityStandardizer.standardize(target)
normalizer = staintools.StainNormalizer(method='vahadane')
for img_fn in img_fn_list:
    img_full_fn = os.path.join(img_data_dir, case_id, img_fn)
    print("Processing %s" % img_fn)
    if os.path.exists(img_full_fn):
        to_transform = staintools.read_image(img_full_fn)
        to_transform = staintools.LuminosityStandardizer.standardize(to_transform)

        # Stain normalize
        normalizer.fit(target)
        transformed = normalizer.transform(to_transform)

        img = Image.fromarray(transformed)
        save_to = os.path.join(output_dir, img_fn).replace(".png", "_norm.png")
        img.save(save_to)

        save_to = os.path.join(RESULTS_DIR, case_id, img_fn)
        shutil.copyfile(img_full_fn, save_to)
